import numpy as np
import torch
import torch.nn.functional as F
import cv2
from collections import deque, defaultdict

from utils.predictor import AsyncPredictor

from fastreid.config import get_cfg

class TargetState(object):
    New = 0
    Match = 1
    Lost = 2

class TargetNode:
    def __init__(self, origin, cls, track_id, frame_id, mtarget_id, xyxy, img=None, conf=None, feature=None):
        self.origin = origin
        self.cls = cls
        self.track_id = track_id
        self.frame_id = frame_id
        self.xyxy = xyxy
        self.img = img
        self.conf = conf
        self.feature = feature

        self.state = TargetState.New
        self.find_match_list = set()

        self.mtarget = MTargetNode(mtarget_id, self)
        self.match_conf = None

        self.area_match = set()

    def update_state(self, state):
        self.state = state
        if state == TargetState.Match:
            self.find_match_list.clear()

    def update_feature_img(self, img, conf, feature):
        self.feature = feature
        self.conf = conf
        self.img = img

    def update(self, frame_id, xyxy):
        self.frame_id = frame_id
        self.xyxy = xyxy
        if self.mtarget == None:
            self.update_state(TargetState.New)
        else:
            self.update_state(TargetState.Match)

class MTargetNode:
    def __init__(self, mTarget_id, target):
        self.mTarget_id = mTarget_id
        self.match_dict = defaultdict(dict)

        self.match_dict[target.origin][target.track_id] = target.frame_id
        self.min_frame_id = target.frame_id

    def match(self, target_node, frame_id, match_conf=None):
        self.match_dict[target_node.origin][target_node.track_id] = frame_id
        target_node.mtarget = self
        target_node.match_conf = match_conf if match_conf != None else target_node.match_conf
        target_node.update_state(TargetState.Match)

class MultiSourceTracker:
    def __init__(self, config, source_names, areas=[]):
        self.source_names = source_names
        self.config = config

        cfg = get_cfg()
        cfg.merge_from_file(self.config['predictor_config'])
        cfg.freeze()
        self.predictor = AsyncPredictor(cfg, num_gpus=1)
        _ = self.predictor([np.zeros((200, 200, 3), dtype=np.uint8)])

        self.frame_id  = 0
        self.count = 0
        self.max_target_lost = self.config['max_target_lost'] # 丟失多少幀後刪除
        self.match_thres = self.config['match_thres'] # 匹配閾值
        self.min_match_lost = self.config['min_match_lost'] # 最少lost多少幀後才匹配

        self.target_pool = {x:{} for x in self.source_names} # 用來存放target
        self.target_lost_deque = deque([[] for _ in range(self.max_target_lost)], maxlen=self.max_target_lost) # 用來存放丟失target的source, track_id
        self.source_match_list = {x:set() for x in self.source_names} # 用來存放每個source的匹配list

        areas = [('cam1', 'cam2')] # TODO: Test
        self.area_match = defaultdict(set)
        for area in areas:
            area_sort = sorted(area)
            for i, n in enumerate(area_sort):
                for n2 in area_sort[i+1:]:
                    self.area_match[n].add(n2)

    def update(self, data):
        self.frame_id += 1
        reslut_target = {}
        self.target_lost_deque.appendleft([])

        '''更新feature and frame_id'''
        # 遍歷source
        for source_name, (img, pred, targets) in data.items():
            reslut_target[source_name] = []
            imgs, img_info = [], []
            # 遍歷targets
            for *xyxy, conf, cls, track_id in targets:
                xyxy = list(map(lambda x: 0 if x < 0 else x, xyxy ))
                # 更新現有target
                if track_id in self.target_pool[source_name].keys():
                    old_deque_index = self.frame_id - self.target_pool[source_name][track_id].frame_id
                    self.target_lost_deque[old_deque_index].remove((source_name, track_id))
                    self.target_lost_deque[0].append((source_name, track_id))

                    self.target_pool[source_name][track_id].update(self.frame_id, xyxy)
                    self.remove_match_list(source_name, track_id)

                    # conf比較高, 更新future
                    if self.target_pool[source_name][track_id].conf < conf:
                        imgs.append(img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :])
                        img_info.append((track_id, conf))
                # 新target
                else:
                    imgs.append(img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :])
                    img_info.append((track_id, conf))
                    self.target_pool[source_name][track_id] = TargetNode(source_name, cls, track_id, self.frame_id, self.count, xyxy)
                    self.count += 1
                    self.target_lost_deque[0].append((source_name, track_id))
                
                reslut_target[source_name].append(self.target_pool[source_name][track_id])

            if len(imgs):
                features = F.normalize(self.predictor(imgs))

                for i, (img_id, conf) in enumerate(img_info):
                    self.target_pool[source_name][img_id].update_feature_img(imgs[i], conf, features[i])

        '''更新Lost target state'''
        for source_name, track_id in self.target_lost_deque[1]:
            if self.target_pool[source_name][track_id].state == TargetState.New:
                self.target_pool[source_name][track_id].update_state(TargetState.Lost)

        '''更新匹配list'''
        for source_name, track_id in self.target_lost_deque[self.min_match_lost]:
            if self.target_pool[source_name][track_id].state == TargetState.Lost:
                for s in self.source_names:
                    self.add_match_list(source_name, track_id, s)

        '''匹配target'''
        for source_name, track_id in self.target_lost_deque[0]:
            if not len(self.source_match_list[source_name]) and not len(self.area_match[source_name]):
                continue
            query_feature = self.target_pool[source_name][track_id].feature
            
            lost_features, lost_info = self.get_lost_feature(source_name, track_id) # lost匹配
            area_features, area_info = self.get_area_feature(source_name, track_id) # 重疊區域匹配

            features = torch.stack([query_feature, *lost_features, *area_features])
            distmat = 1 - torch.mm(features[:1], features[1:].t())
            distmat = distmat.numpy()[0]
            sort_index = np.argsort(distmat)
            lost_match = False
            for i, index in enumerate(sort_index):
                if distmat[index] > self.match_thres:
                    break

                # lost匹配
                if len(lost_features) > index:
                    if lost_match: # 限制重複匹配
                        continue
                    lost_match = True
                    s, t = lost_info[index]
                    self.remove_match_list(s, t)

                    mtarget = self.target_pool[s][t].mtarget
                    self.target_pool[s][t].update_state(TargetState.Match)
                    mtarget.match(self.target_pool[source_name][track_id], self.frame_id, distmat[index])

                    # cv2.imwrite(f'test/#{mtarget.mTarget_id}_{s}-{t}.jpg', self.target_pool[s][t].img)
                    # cv2.imwrite(f'test/#{mtarget.mTarget_id}-{distmat[index]:.4f}_{source_name}-{track_id}.jpg', self.target_pool[source_name][track_id].img)
                # 重疊區域匹配
                else:
                    s, t = area_info[index]
                    self.target_pool[source_name][track_id].area_match.add(s)

                    q_target = self.target_pool[source_name][track_id]
                    m_target = self.target_pool[s][t]
                    t1, t2 = (q_target, m_target) if q_target.mtarget.min_frame_id <= m_target.frame_id else (m_target, q_target)
                    mtarget = t1.mtarget

                    t1.update_state(TargetState.Match)
                    mtarget.match(t2, self.frame_id, distmat[index])

                    # cv2.imwrite(f'test/#{mtarget.mTarget_id}_{s}-{t}.jpg', self.target_pool[s][t].img)
                    # cv2.imwrite(f'test/#{mtarget.mTarget_id}-{distmat[index]:.4f}_{source_name}-{track_id}.jpg', self.target_pool[source_name][track_id].img)

        '''刪除遺失的target'''
        for source_name, track_id in self.target_lost_deque[-1]:
            self.remove_match_list(source_name, track_id)
            del self.target_pool[source_name][track_id]

        '''整理輸出'''
        return self.get_result(data)
    
    def get_lost_feature(self, source_name, track_id):
        features = []
        feature_info = [] # 用來存放比較的target
        if self.target_pool[source_name][track_id].state == TargetState.New:
            for s, t in self.source_match_list[source_name]:
                features.append(self.target_pool[s][t].feature)
                feature_info.append((s, t))
        return features, feature_info
    
    def get_area_feature(self, source_name, track_id):
        features = []
        feature_info = [] # 用來存放比較的target

        target = self.target_pool[source_name][track_id]
        for s in self.area_match[source_name]:
            if s not in target.area_match: # 限制在同一個相機只能匹配一次
                for t_id, t in self.target_pool[s].items():
                    if t.state == TargetState.New:
                        features.append(t.feature)
                        feature_info.append((s, t_id))
        return features, feature_info
    
    def remove_match_list(self, source_name, track_id):
        for s in self.target_pool[source_name][track_id].find_match_list:
            self.source_match_list[s].remove((source_name, track_id))
        self.target_pool[source_name][track_id].find_match_list.clear()

    def add_match_list(self, source_name, track_id, add_source_name):
        self.source_match_list[add_source_name].add((source_name, track_id))
        self.target_pool[source_name][track_id].find_match_list.add(add_source_name)

    #TODO:緩存已處理好的target
    def get_result(self, data):
        res = defaultdict(list)
        for source_name, (img, pred, targets) in data.items():
            for *xyxy, conf, cls, track_id in targets:
                target = self.target_pool[source_name][track_id]
                mtarget = target.mtarget
                res_id = mtarget.mTarget_id
                res[source_name].append((*xyxy, conf, cls, track_id, res_id))
        return res


