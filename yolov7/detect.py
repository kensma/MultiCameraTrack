from collections.abc import Callable, Iterable, Mapping
from typing import Any
import torch
import sys
import cv2
import numpy as np
import random
import torch.multiprocessing as mp
# import multiprocessing as mp
# import threading
# import queue
import time

sys.path.append('./yolov7')

from yolov7.utils.torch_utils import select_device
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import scale_coords, non_max_suppression, check_img_size
from yolov7.utils.plots import plot_one_box
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import TracedModel

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

class Detect:
    @torch.no_grad()
    def __init__(self, cfg):
        self.device = cfg.device
        self._half = cfg.half
        self._batch_size = cfg.batch_size
        self._conf_thres = cfg.conf_thres
        self._iou_thres = cfg.iou_thres
        self._classes = cfg.classes
        self._agnostic_nms = cfg.agnostic_nms

        self.device = torch.device(self.device)
        self._half = self._half and self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(cfg.weights, map_location=self.device)  # load FP32 model
        self._stride = int(self.model.stride.max())  # model stride
        self._imgsz = check_img_size(cfg.imgsz, s=self._stride)  # check img_size
        
        if cfg.trace:
            self.model = TracedModel(self.model, self.device, self._imgsz)

        if self._half:
            self.model.half()

        self._names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        self.model(torch.zeros(self._batch_size, 3, self._imgsz, self._imgsz).to(self.device).type_as(next(self.model.parameters())))

    @torch.no_grad()
    def __call__(self, im0s, shapes):
        imgs_tensor = im0s.to(self.device)

        imgs_tensor = imgs_tensor.half() if self._half else imgs_tensor.float()
        imgs_tensor /= 255.0

        pred = self.model(imgs_tensor)[0]

        pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, classes=self._classes, agnostic=self._agnostic_nms)

        res = []
        for i, det in enumerate(pred):
            det[:, :4] = scale_coords((self._imgsz, self._imgsz), det[:, :4], shapes[i]).round()
            res.append(det.cpu().numpy())

        return res

    def get_names(self):
        return self._names
    

class AsyncDetect:

    class _DetectWorker(mp.Process):
        def __init__(self, in_queue, out_queue, cfg):
            mp.Process.__init__(self)
            self.cfg = cfg

            self.in_queue = in_queue
            self.out_queue = out_queue

        def run(self):
            detect = Detect(self.cfg)

            while True:
                idx, imgs, shapes  = self.in_queue.get()
                res = detect(imgs, shapes)
                self.out_queue.put((idx, res))
                del imgs
                del shapes
                del idx

    def __init__(self, cfg):
        self.num_detect = cfg.num_detect
        self.batch_size = cfg.batch_size
        self.imgsz = cfg.imgsz
        self.in_queue = mp.Queue(self.num_detect)
        self.out_queue = mp.Queue(self.num_detect)
        self.detects = []
        self.device = cfg.device
        for i in range(self.num_detect):
            cfg.device = self.device[i]
            self.detects.append(self._DetectWorker(self.in_queue, self.out_queue, cfg))
            self.detects[-1].start()

        #  初始化
        for i in range(self.num_detect):
            self.__call__(
                np.zeros((self.batch_size*self.num_detect, 3, self.imgsz, self.imgsz)),
                [(self.imgsz, self.imgsz, 3) for _ in range(self.batch_size*self.num_detect)]
            )

    def __call__(self, im0s, shapes):
        # print("====================================")
        # t0 = time.time()
        # imgs = []
        # shapes = []
        # for img in im0s:
        #     shapes.append(img.shape)
        #     img, _, _ = letterbox(img, (self.imgsz, self.imgsz), auto=False, scaleup=True)
        #     img = img.transpose(2, 0, 1)
        #     imgs.append(img)

        # imgs = np.array(imgs)
        imgs = np.array(im0s)
        imgs_tensor = torch.from_numpy(imgs)
        # print('preprocess time: ', time.time()-t0)
        # t0 = time.time()
        for i in range(self.num_detect):
            self.in_queue.put((i, imgs_tensor[self.batch_size*i:self.batch_size*(i+1)], shapes[self.batch_size*i:self.batch_size*(i+1)]))
        # print('put time: ', time.time()-t0)
        # t0 = time.time()

        res = []
        for _ in range(self.num_detect):
            idx, data = self.out_queue.get()
            res[self.batch_size*idx:self.batch_size*idx] = data
        # print('get time: ', time.time()-t0)
        # print("====================================")
        return res