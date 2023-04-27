import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

from utils.predictor import AsyncPredictor

from fastreid.config import get_cfg

class TargetState(object):
    New = 0
    Match = 1
    Lost = 2

class TargetNode:
    def __init__(self, origin, cls, track_id, frame_id, xyxy, img=None, conf=None, feature=None):
        self.origin = origin
        self.cls = cls
        self.track_id = track_id
        self.frame_id = frame_id
        self.xyxy = xyxy
        self.img = img
        self.conf = conf
        self.feature = feature

        self.state = TargetState.New
        self.find_match_list = []

        self.next = None
        self.prev = None
        self.match_conf = None

    def match(self, target_node, match_conf):
        self.prev = target_node
        target_node.next = self
        self.match_conf = match_conf
        self.update_state(TargetState.Match)
        target_node.update_state(TargetState.Match)

    def update_state(self, state):
        self.state = state
        if state == TargetState.Match:
            self.find_match_list = []

    def update_feature_img(self, img, conf, feature):
        self.feature = feature
        self.conf = conf

    def update(self, frame_id, xyxy):
        self.frame_id = frame_id
        self.xyxy = xyxy
        if self.prev == None:
            self.update_state(TargetState.New)
        else:
            self.update_state(TargetState.Match)

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
        self.max_target_lost = self.config['max_target_lost'] # 丟失多少幀後刪除
        self.match_thres = self.config['match_thres'] # 匹配閾值

        self.target_pool = {x:{} for x in self.source_names} # 用來存放target
        self.target_lost_deque = deque([[] for _ in range(self.max_target_lost)], maxlen=self.max_target_lost) # 用來存放丟失target的source, track_id
        self.source_match_list = {x:[] for x in self.source_names} # 用來存放每個source的匹配list


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
                    self.target_pool[source_name][track_id] = TargetNode(source_name, cls, track_id, self.frame_id, xyxy)
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
                    self.source_match_list[s].append((source_name, track_id))
                    self.target_pool[source_name][track_id].find_match_list.append(s)

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
                    self.target_pool[source_name][track_id].match(self.target_pool[s][t], distmat[max_index])
                    old_deque_index = self.frame_id - self.target_pool[s][t].frame_id
                    self.target_lost_deque[old_deque_index].remove((s, t))
                    del self.target_pool[s][t]

        '''刪除遺失的target'''
        for source_name, track_id in self.target_lost_deque[-1]:
            self.remove_match_list(source_name, track_id)
            del self.target_pool[source_name][track_id]

        '''整理輸出'''
        reslut = {}
        for name in self.source_names:
            reslut[name] = list(map(self.target2result, reslut_target[name]))
        return reslut
    
    def remove_match_list(self, source_name, track_id):
        for s in self.target_pool[source_name][track_id].find_match_list:
            self.source_match_list[s].remove((source_name, track_id))
        self.target_pool[source_name][track_id].find_match_list = []

    def target2result(self, target):
        if target.prev:
            return (target.prev.origin, target.match_conf)
        return (None, None)

