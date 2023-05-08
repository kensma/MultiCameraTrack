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
    def __init__(self, origin, cls, track_id, frame_id, target_id, xyxy, img=None, conf=None, feature=None):
        self.origin = origin
        self.cls = cls
        self.track_id = track_id
        self.frame_id = frame_id
        self.target_id = target_id
        self.xyxy = xyxy
        self.img = img
        self.conf = conf
        self.feature = feature

        self.state = TargetState.New
        self.find_match_list = set()

        self.mtarget = None
        self.match_conf = None

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
    def __init__(self, mTarget_id):
        self.mTarget_id = mTarget_id
        self.match_dict = defaultdict(dict)

    def match(self, target_node, frame_id, match_conf=None):
        self.match_dict[target_node.origin][target_node.track_id] = frame_id
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

        self.target_pool = {x:{} for x in self.source_names} # 用來存放target
        self.target_lost_deque = deque([[] for _ in range(self.max_target_lost)], maxlen=self.max_target_lost) # 用來存放丟失target的source, track_id
        self.source_match_list = {x:set() for x in self.source_names} # 用來存放每個source的匹配list

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
                #TODO: 增加距離匹配邏輯
                for s in self.source_names:
                    self.add_match_list(source_name, track_id, s)

        '''匹配target'''
        for source_name, track_id in self.target_lost_deque[0]:
            if not len(self.source_match_list[source_name]):
                continue
            if self.target_pool[source_name][track_id].state == TargetState.New:
                features = [self.target_pool[source_name][track_id].feature]
                feature_info = [] # 用來存放比較的target
                for s, t in self.source_match_list[source_name]:
                    features.append(self.target_pool[s][t].feature)
                    feature_info.append((s, t))
                features = torch.stack(features)

                distmat = 1 - torch.mm(features[:1], features[1:].t())
                distmat = distmat.numpy()[0]
                max_index = np.argmax(distmat)
                if distmat[max_index] < self.match_thres:
                    s, t = feature_info[max_index]
                    self.remove_match_list(s, t)
                    mtarget = MTargetNode(self.target_pool[s][t].target_id)
                    mtarget.match(self.target_pool[s][t], self.frame_id)
                    mtarget.match(self.target_pool[source_name][track_id], self.frame_id, distmat[max_index])

                    # cv2.imwrite(f'test/#{mtarget.mTarget_id}_{s}-{t}.jpg', self.target_pool[s][t].img)
                    # cv2.imwrite(f'test/#{mtarget.mTarget_id}-{distmat[max_index]:.4f}_{source_name}-{track_id}.jpg', self.target_pool[source_name][track_id].img)

        '''刪除遺失的target'''
        for source_name, track_id in self.target_lost_deque[-1]:
            self.remove_match_list(source_name, track_id)
            del self.target_pool[source_name][track_id]

        '''整理輸出'''
        return self.get_result(data)
    
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
                res_id = mtarget.mTarget_id if mtarget != None else target.target_id
                res[source_name].append((*xyxy, conf, cls, track_id, res_id))
        return res


