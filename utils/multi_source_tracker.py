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
        self.fast_frame_id = frame_id
        self.frame_id = frame_id
        self.xyxy = xyxy

        self.img = deque([], maxlen=5)
        self.conf = deque([], maxlen=5)
        self.feature = deque([], maxlen=5)
        self.mean_feature = None
        if feature != None:
            self.update_feature_img(img, conf, feature)

        self.state = TargetState.New
        self.find_match_list = set()

        self.mtarget = MTargetNode(mtarget_id, self)
        self.match_conf = None

    def update_state(self, state):
        self.state = state
        if state == TargetState.Match:
            self.find_match_list.clear()
        elif state == TargetState.Lost:
            self.mtarget.match_dict[self.origin] = -1

    def update_feature_img(self, img, conf, feature, replace=None):
        if replace == None:
            self.feature.append(feature)
            self.conf.append(conf)
            self.img.append(img)
        else:
            self.feature[replace] = feature
            self.conf[replace] = conf
            self.img[replace] = img
        
        self.mean_feature = torch.mean(torch.stack(list(self.feature)), dim=0)


    def update(self, frame_id, xyxy):
        self.frame_id = frame_id
        self.xyxy = xyxy
        self.mtarget.match_dict[self.origin] = self.track_id
        if self.mtarget.is_match:
            self.update_state(TargetState.Match)
        else:
            self.update_state(TargetState.New)

class MTargetNode:
    @staticmethod
    def _default_match_dict():
        return lambda: -1

    def __init__(self, mTarget_id, target):
        self.mTarget_id = mTarget_id
        self.match_dict = defaultdict(self._default_match_dict())

        self.match_dict[target.origin] = target.track_id
        self.min_frame_id = target.frame_id
        self.is_match = False

    def match(self, target_node, frame_id, match_conf=None):
        self.is_match = True
        self.match_dict[target_node.origin] = target_node.track_id
        target_node.mtarget = self
        target_node.match_conf = match_conf if match_conf != None else target_node.match_conf
        target_node.update_state(TargetState.Match)

class MultiSourceTracker:
    def __init__(self, config, source_names):
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
        self.areas = self.config['areas'] # 重疊區域

        self.target_pool = {x:{} for x in self.source_names} # 用來存放target
        self.target_lost_deque = deque([[] for _ in range(self.max_target_lost)], maxlen=self.max_target_lost) # 用來存放丟失target的source, track_id
        self.source_match_list = {x:set() for x in self.source_names} # 用來存放每個source的匹配list

        self.area_match = defaultdict(set)
        for area in self.areas:
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

                    if len(self.target_pool[source_name][track_id].conf) < 5:
                        imgs.append(img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :])
                        img_info.append((track_id, conf, None))
                    # conf比較高, 更新future
                    elif min(self.target_pool[source_name][track_id].conf) < conf:
                        i = np.argmin(self.target_pool[source_name][track_id].conf)
                        imgs.append(img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :])
                        img_info.append((track_id, conf, i))
                        pass
                # 新target
                else:
                    imgs.append(img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :])
                    img_info.append((track_id, conf, None))
                    self.target_pool[source_name][track_id] = TargetNode(source_name, cls, track_id, self.frame_id, self.count, xyxy)
                    self.count += 1
                    self.target_lost_deque[0].append((source_name, track_id))
                
                reslut_target[source_name].append(self.target_pool[source_name][track_id])

            if len(imgs):
                features = F.normalize(self.predictor(imgs))

                for i, (img_id, conf, replace) in enumerate(img_info):
                    self.target_pool[source_name][img_id].update_feature_img(imgs[i], conf, features[i], replace)

        '''更新Lost target state'''
        for source_name, track_id in self.target_lost_deque[1]:
            self.target_pool[source_name][track_id].update_state(TargetState.Lost)
            # if self.target_pool[source_name][track_id].state == TargetState.New:
            #     self.target_pool[source_name][track_id].update_state(TargetState.Lost)

        '''更新匹配list'''
        for source_name, track_id in self.target_lost_deque[self.min_match_lost]:
            if self.target_pool[source_name][track_id].state == TargetState.Lost:
                # TODO 加入距離匹配
                for s in self.source_names:
                    if self.target_pool[source_name][track_id].mtarget.match_dict[s] == -1:
                        self.add_match_list(source_name, track_id, s)

        '''匹配target'''
        lost_info_list = []
        area_info_list = []
        query_list = []
        distmats = []
        max_len = 0
        for source_name, track_id in self.target_lost_deque[0]:
            if not len(self.source_match_list[source_name]) and not len(self.area_match[source_name]):
                continue
            query_feature = self.target_pool[source_name][track_id].mean_feature
            
            lost_features, lost_info = self.get_lost_feature(source_name, track_id) # lost匹配
            area_features, area_info = self.get_area_feature(source_name, track_id) # 重疊區域匹配
            # area_features, area_info = [], []

            if not len(lost_info) and not len(area_info):
                continue

            lost_info_list.append(lost_info)
            area_info_list.append(area_info)
            query_list.append((source_name, track_id))

            features = torch.stack([query_feature, *lost_features, *area_features])
            distmat = 1 - torch.mm(features[:1], features[1:].t())
            distmat = distmat.numpy()[0]
            distmats.append(distmat)
            max_len = max(max_len, distmat.shape[0])
        
        if len(distmats):
            distmat_matrix = np.full((len(distmats), max_len), self.match_thres + 1)
            for i, distmat in enumerate(distmats):
                distmat_matrix[i, :distmat.shape[0]] = distmat

            temp_match = defaultdict(set)
            cost = 0
            while len(distmats) > cost:
                index = np.unravel_index((np.argmin(distmat_matrix)), distmat_matrix.shape)
                if distmat_matrix[index] > self.match_thres:
                    break

                r, i = index
                lost_info = lost_info_list[r]
                area_info = area_info_list[r]
                q_source_name, q_track_id = query_list[r]
                # lost匹配
                if len(lost_info) > i:
                    s, t = lost_info[i]
                    if t in temp_match[s]:
                        distmat_matrix[index] = self.match_thres + 1
                        continue

                    self.remove_match_list(s, t)
                    mtarget = self.target_pool[s][t].mtarget
                    self.target_pool[s][t].update_state(TargetState.Match)
                    mtarget.match(self.target_pool[q_source_name][q_track_id], self.frame_id, distmat_matrix[index])

                    # cv2.imwrite(f'test/#{mtarget.mTarget_id}_{s}-{t}-lost.jpg', self.target_pool[s][t].img[-1])
                    # cv2.imwrite(f'test/#{mtarget.mTarget_id}-{distmat_matrix[index]:.4f}_{q_source_name}-{q_track_id}-lost.jpg', self.target_pool[source_name][track_id].img[-1])
                # 重疊區域匹配
                else:
                    s, t = area_info[i-len(lost_info)]
                    if t in temp_match[s]:
                        distmat_matrix[index] = self.match_thres + 1
                        continue

                    q_target = self.target_pool[q_source_name][q_track_id]
                    m_target = self.target_pool[s][t]
                    t1, t2 = (q_target, m_target) if q_target.mtarget.min_frame_id <= m_target.frame_id else (m_target, q_target)
                    mtarget = t1.mtarget

                    t1.update_state(TargetState.Match)
                    mtarget.match(t2, self.frame_id, distmat_matrix[index])

                    # cv2.imwrite(f'test/#{mtarget.mTarget_id}_{s}-{t}-area.jpg', self.target_pool[s][t].img[-1])
                    # cv2.imwrite(f'test/#{mtarget.mTarget_id}-{distmat_matrix[index]:.4f}_{q_source_name}-{q_track_id}-area.jpg', self.target_pool[source_name][track_id].img[-1])

                temp_match[s].add(t)
                distmat_matrix[r, :] = self.match_thres + 1
                cost += 1

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
                if self.target_pool[s][t].frame_id <= self.target_pool[source_name][track_id].fast_frame_id:
                    features.append(self.target_pool[s][t].mean_feature)
                    feature_info.append((s, t))
        return features, feature_info
    
    def get_area_feature(self, source_name, track_id):
        features = []
        feature_info = [] # 用來存放比較的target

        target = self.target_pool[source_name][track_id]
        for s in self.area_match[source_name]:
            if target.mtarget.match_dict[s] == -1: # 限制在同一個相機只能匹配一次
                for t_id, t in self.target_pool[s].items():
                    if t.state != TargetState.Lost:
                        # features.append(t.feature)
                        features.append(t.mean_feature)
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
    
    def stop(self):
        self.predictor.stop()


