import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque, defaultdict
import cv2
import os

from utils.reid import AsyncExtractor

class TargetState(object):
    New = 0
    Match = 1
    Lost = 2

class TargetNode:
    def __init__(self, origin, cls, track_id, frame_id, mtarget_id, xyxy, img=None, conf=None, feature=None, max_feature=5):
        self.origin = origin
        self.cls = cls
        self.track_id = track_id
        self.fast_frame_id = frame_id
        self.frame_id = frame_id
        self.xyxy = xyxy

        self.max_feature = max_feature
        self.img = deque([], maxlen=self.max_feature)
        self.conf = deque([], maxlen=self.max_feature)
        self.feature = deque([], maxlen=self.max_feature)
        self.mean_feature = None
        if feature != None:
            self.update_feature_img(img, conf, feature)

        self.state = TargetState.New
        self.find_match_list = set()

        self.mtarget = MTargetNode(mtarget_id, self)
        self.match_conf = None

        # 之前匹配過的target
        self.matched = defaultdict(set)
    
    def matched_clear(self):
        self.matched = defaultdict(set)

    def matched_add(self, tracks):
        for track in tracks:
            self.matched[track[0]].add(track[1])

    def is_matched(self, cam, track_id):
        return track_id in self.matched[cam]

    def update_state(self, state):
        self.state = state
        if state == TargetState.Match:
            pass
        elif state == TargetState.Lost:
            pass

    def update_feature_img(self, img, conf, feature, replace=None):
        self.matched_clear()
        if replace == None:
            self.feature.append(feature)
            self.conf.append(conf)
            self.img.append(img)
        else:
            self.feature[replace] = feature
            self.conf[replace] = conf
            self.img[replace] = img

        if len(self.feature) == self.max_feature:
            self.mean_feature = torch.mean(torch.stack(list(self.feature)), dim=0)
        


    def update(self, frame_id, xyxy):
        self.frame_id = frame_id
        self.xyxy = xyxy
        if self.mtarget.is_match:
            self.update_state(TargetState.Match)
        else:
            self.update_state(TargetState.New)

class MTargetNode:
    @staticmethod
    def _default_match_dict():
        return lambda: None

    def __init__(self, mTarget_id, target):
        self.mTarget_id = mTarget_id
        self.match_dict = defaultdict(self._default_match_dict())
        self.min_frame_id = target.frame_id

        self.match_dict[target.origin] = target
        self.match_count = 1
        self.is_match = False

        self.last_lost_target = None

    def match(self, target_node, match_conf=None):
        if target_node.state == TargetState.Match:
            target_node.mtarget.unmatch(target_node, roll_back=True)
        self.match_dict[target_node.origin] = target_node
        self.match_count += 1
        if (self.last_lost_target != None and self.match_count >= 1) or (self.last_lost_target == None and self.match_count > 1):
            self.is_match = True
        target_node.mtarget = self
        target_node.match_conf = match_conf if match_conf != None else target_node.match_conf
        target_node.update_state(TargetState.Match)

    def unmatch(self, target_node, roll_back=False):
        self.match_dict[target_node.origin] = None
        self.match_count -= 1
        if (self.last_lost_target != None and self.match_count < 1) or (self.last_lost_target == None and self.match_count <= 1):
            self.is_match = False

        if self.match_count == 0 and not roll_back:
            self.last_lost_target = target_node

class MultiSourceTracker:
    @torch.no_grad()
    def __init__(self, cfg, reid_cfg, source_names):
        self.source_names = source_names
        self.cfg = cfg

        # 初始化 reid
        self.extractor = AsyncExtractor(reid_cfg)
        self.img_transform = self.extractor.img_transform

        self.frame_id  = 0
        self.count = 0
        self.max_target_lost = self.cfg.max_target_lost # 丟失多少幀後刪除
        self.match_thres = self.cfg.match_thres # 匹配閾值
        self.min_match_lost = self.cfg.min_match_lost # 最少lost多少幀後才匹配
        self.areas = self.cfg.areas # 重疊區域
        self.min_frame_feature = self.cfg.min_frame_feature # 最少出現幀數
        self.max_feature = self.cfg.max_feature # 最多特徵數

        self.target_pool = {x:{} for x in self.source_names} # 用來存放target
        self.target_lost_deque = deque([[] for _ in range(self.max_target_lost)], maxlen=self.max_target_lost) # 用來存放丟失target的source, track_id，index越大越久前丟失
        self.source_match_list = {x:set() for x in self.source_names} # 用來存放每個source的匹配list

        self.area_match = defaultdict(set)
        for area in self.areas:
            area_sort = sorted(area)
            for i, n in enumerate(area_sort):
                for n2 in area_sort[i+1:]:
                    self.area_match[n].add(n2)

    @torch.no_grad()
    def update(self, data):
        self.frame_id += 1
        reslut_target = {}
        self.target_lost_deque.appendleft([])

        '''更新feature and frame_id'''
        # 遍歷source
        imgs, img_info = [], []
        for source_name, (img, targets) in data.items():
            reslut_target[source_name] = []
            # 遍歷targets
            for *xyxy, conf, cls, track_id in targets:
                # 更新現有target
                if track_id in self.target_pool[source_name].keys():
                    old_deque_index = self.frame_id - self.target_pool[source_name][track_id].frame_id
                    self.target_lost_deque[old_deque_index].remove((source_name, track_id))
                    self.target_lost_deque[0].append((source_name, track_id))

                    self.target_pool[source_name][track_id].update(self.frame_id, xyxy)
                    self.remove_match_list(source_name, track_id)

                    in_border = np.min(xyxy) < 0
                    in_border = img.shape[0] < xyxy[3] or img.shape[1] < xyxy[2] or in_border
                    # 前幾幀不做feature、當xyxy有負值(在邊界)不做feature
                    if self.frame_id - self.target_pool[source_name][track_id].fast_frame_id > self.min_frame_feature and not in_border:
                    #人物完整性實驗
                    # if True:
                    #     xyxy = list(map(lambda x: 0 if x < 0 else x, xyxy ))
                        if len(self.target_pool[source_name][track_id].conf) < self.max_feature:
                            im = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
                            im = self.img_transform(im)
                            imgs.append(im)
                            img_info.append((source_name, track_id, conf, None))
                        # conf比較高, 更新future
                        elif min(self.target_pool[source_name][track_id].conf) < conf:
                            i = np.argmin(self.target_pool[source_name][track_id].conf)
                            im = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
                            im = self.img_transform(im)
                            imgs.append(im)
                            img_info.append((source_name, track_id, conf, i))
                # 新target
                else:
                    self.target_pool[source_name][track_id] = TargetNode(source_name, cls, track_id, self.frame_id, self.count, xyxy, max_feature=self.max_feature)
                    self.count += 1
                    self.target_lost_deque[0].append((source_name, track_id))
                
                reslut_target[source_name].append(self.target_pool[source_name][track_id])

        if len(imgs):
            imgs_tensor = torch.stack(imgs)
            features, _ = self.extractor(imgs_tensor)
            features = F.normalize(features)
            features = features.cpu()

            for i, (source_name, img_id, conf, replace) in enumerate(img_info):
                self.target_pool[source_name][img_id].update_feature_img(imgs[i], conf, features[i], replace)

        '''更新Lost target state'''
        for source_name, track_id in self.target_lost_deque[1]:
            self.target_pool[source_name][track_id].update_state(TargetState.Lost)

        '''更新匹配list'''
        for source_name, track_id in self.target_lost_deque[self.min_match_lost]:
            target = self.target_pool[source_name][track_id]
            if target.state == TargetState.Lost:
                # 在重疊區域內被匹配後消失並且還有同一個target在不同相機內,不應該再被匹配
                if target.mtarget.match_count > 1:
                    del self.target_pool[source_name][track_id]
                    self.target_lost_deque[self.min_match_lost].remove((source_name, track_id))
                else:
                    # TODO 加入距離匹配
                    for s in self.source_names:
                        self.add_match_list(source_name, track_id, s)
                target.mtarget.unmatch(target)

        '''匹配target'''
        lost_info_list = []
        area_info_list = []
        query_list = []
        distmats = []
        max_len = 0
        for source_name, track_id in self.target_lost_deque[0]:
            if not len(self.source_match_list[source_name]) and not len(self.area_match[source_name]):
                continue

            # 需要有足夠的特徵
            if len(self.target_pool[source_name][track_id].feature) < self.max_feature:
                continue

            query_feature = self.target_pool[source_name][track_id].mean_feature
            
            lost_features, lost_info = self.get_lost_feature(source_name, track_id) # lost匹配
            area_features, area_info = self.get_area_feature(source_name, track_id) # 重疊區域匹配
            # area_features, area_info = [], []

            self.target_pool[source_name][track_id].matched_add(lost_info)

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

            # '''餘弦距離結果合併實驗 start'''
            # sort_index = np.argsort(distmat)
            # lost_match = False
            # for i, index in enumerate(sort_index):
            #     if distmat[index] > self.match_thres:
            #         break
            #     # lost匹配
            #     if len(lost_info) > index:
            #         if lost_match: # 限制重複匹配
            #             continue
            #         s, t = lost_info[index]
            #         mtarget = self.target_pool[s][t].mtarget
            #         self.target_pool[s][t].update_state(TargetState.Match)
            #         mtarget.match(self.target_pool[source_name][track_id] distmat[index])
            #     # 重疊區域匹配
            #     else:
            #         s, t = area_info[index-len(lost_info)]
            #         q_target = self.target_pool[source_name][track_id]
            #         m_target = self.target_pool[s][t]
            #         reverse = True
            #         if q_target.match_conf and m_target.match_conf and q_target.match_conf < m_target.match_conf:
            #             reverse = False
            #         elif q_target.fast_frame_id <= m_target.fast_frame_id:
            #             reverse = False
            #         t1, t2 = (m_target, q_target) if reverse else (q_target, m_target)
            #         mtarget = t1.mtarget

            #         t1.update_state(TargetState.Match)
            #         mtarget.match(t2, distmat[index])
            # '''餘弦距離結果合併實驗 end'''
        
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

                    mtarget = self.target_pool[s][t].mtarget
                    self.target_pool[s][t].update_state(TargetState.Match)
                    mtarget.match(self.target_pool[q_source_name][q_track_id], distmat_matrix[index])

                # 重疊區域匹配
                else:
                    s, t = area_info[i-len(lost_info)]

                    if t in temp_match[s]:
                        distmat_matrix[index] = self.match_thres + 1
                        continue

                    q_target = self.target_pool[q_source_name][q_track_id]
                    m_target = self.target_pool[s][t]

                    reverse = True
                    #特徵距離優先
                    if q_target.match_conf and m_target.match_conf and q_target.match_conf < m_target.match_conf:
                        reverse = False
                    elif q_target.fast_frame_id <= m_target.fast_frame_id:
                        reverse = False

                    t1, t2 = (m_target, q_target) if reverse else (q_target, m_target)
                    mtarget = t1.mtarget

                    t1.update_state(TargetState.Match)
                    mtarget.match(t2, distmat_matrix[index])

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
        target = self.target_pool[source_name][track_id]
        if target.state == TargetState.New:
            for s, t in self.source_match_list[source_name]:
                # 需要是Lost狀態
                if self.target_pool[s][t].state != TargetState.Lost:
                    continue
                # 需要lost時間比query出現的時間早
                if self.target_pool[s][t].frame_id > target.fast_frame_id:
                    continue
                # 需要有足夠的特徵
                if len(self.target_pool[s][t].feature) < self.max_feature:
                    continue
                # 之前沒有匹配過
                if target.is_matched(s, t):
                    continue
                features.append(self.target_pool[s][t].mean_feature)
                feature_info.append((s, t))
        return features, feature_info
    
    def get_area_feature(self, source_name, track_id):
        features = []
        feature_info = [] # 用來存放比較的target

        target = self.target_pool[source_name][track_id]
        for s in self.area_match[source_name]:
            if target.mtarget.match_dict[s] == None: # 限制在同一個相機只能匹配一次
                for t_id, t in self.target_pool[s].items():
                    if t.state != TargetState.Lost:
                        if self.target_pool[s][t_id].mtarget.match_dict[source_name] != None: # 這個target不應該在這個相機內匹配
                            continue
                        # 需要有足夠的特徵
                        if len(self.target_pool[s][t_id].feature) < self.max_feature:
                            continue
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
        for source_name, (img, targets) in data.items():
            for *xyxy, conf, cls, track_id in targets:
                target = self.target_pool[source_name][track_id]
                mtarget = target.mtarget
                mTarget_id = None
                match_conf = None
                if len(target.feature) >= self.max_feature:
                    mTarget_id = mtarget.mTarget_id
                    match_conf = target.match_conf
                res[source_name].append((*xyxy, conf, cls, track_id, mTarget_id, match_conf))
        return res
    
    def stop(self):
        # self.extractor.stop()
        pass