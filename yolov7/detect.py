from collections.abc import Callable, Iterable, Mapping
from typing import Any
import torch
import sys
import cv2
import numpy as np
import random
import torch.multiprocessing as mp
from multiprocessing import shared_memory
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
    def __call__(self, im0s):
        imgs = []
        shape = im0s.shape[1:]
        for img in im0s:
            img, _, _ = letterbox(img, (self._imgsz, self._imgsz), auto=False, scaleup=True)
            img = img.transpose(2, 0, 1)
            imgs.append(img)

        imgs = np.array(imgs)

        imgs_tensor = torch.from_numpy(imgs).to(self.device)

        imgs_tensor = imgs_tensor.half() if self._half else imgs_tensor.float()
        imgs_tensor /= 255.0

        pred = self.model(imgs_tensor)[0]

        pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, classes=self._classes, agnostic=self._agnostic_nms)

        res = []
        for i, det in enumerate(pred):
            det[:, :4] = scale_coords((self._imgsz, self._imgsz), det[:, :4], shape).round()
            res.append(det.cpu().numpy())

        return res

    def get_names(self):
        return self._names
    

class AsyncDetect:

    class _StopToken:
        pass

    class _DetectWorker(mp.Process):
        def __init__(self, in_queue, out_queue, cfg):
            mp.Process.__init__(self)
            self.cfg = cfg

            self.in_queue = in_queue
            self.out_queue = out_queue

        def run(self):
            detect = Detect(self.cfg)
            self.out_queue.put("OK")
            while True:
                idx, shm_name, batch_shape, start, end = self.in_queue.get()
                if isinstance(idx, AsyncDetect._StopToken):
                    break
                shm = shared_memory.SharedMemory(name=shm_name)
                im0s = np.ndarray(batch_shape, dtype=np.uint8, buffer=shm.buf)
                res = detect(im0s[start:end])
                self.out_queue.put((idx, res))
                shm.close()

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

        # #  初始化
        for _ in range(self.num_detect):
            _ = self.out_queue.get()

    def __call__(self, shm_name, batch_shape):
        for i in range(self.num_detect):
            start = self.batch_size*i
            end = self.batch_size*(i+1)
            self.in_queue.put((i, shm_name, batch_shape, start, end))

        res = []
        for _ in range(self.num_detect):
            idx, data = self.out_queue.get()
            res[self.batch_size*idx:self.batch_size*idx] = data
        return res
    
    def stop(self):
        for _ in self.detects:
            self.in_queue.put((self._StopToken, None, None, None, None))
        
        for detect in self.detects:
            while detect.is_alive():
                self.out_queue.get()