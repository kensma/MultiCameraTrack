import torch
import sys
import numpy as np
import torch.multiprocessing as mp
from multiprocessing import shared_memory
import time

sys.path.append('./yolov7')

from yolov7.utils.datasets import letterbox
from yolov7.utils.general import scale_coords, non_max_suppression, check_img_size
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import TracedModel
from utils.utils import StopToken, close_sharedMemory

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

class Detector:
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
    

class AsyncDetector:

    class _DetectWorker(mp.Process):
        def __init__(self, in_queue, out_queue, cfg):
            mp.Process.__init__(self)
            self.cfg = cfg

            self.in_queue = in_queue
            self.out_queue = out_queue

        def run(self):
            detector = Detector(self.cfg)
            self.out_queue.put("OK")
            while True:
                idx, shm_name, batch_shape = self.in_queue.get()
                if isinstance(idx, StopToken):
                    # shm.unlink()
                    break
                shm = shared_memory.SharedMemory(name=shm_name)
                im0s = np.ndarray(batch_shape, dtype=np.uint8, buffer=shm.buf)
                res = detector(im0s)
                self.out_queue.put((idx, res))
                shm.close()

    def __init__(self, cfg):
        self.num_detector = len(cfg.device)
        self.batch_size = cfg.batch_size
        self.imgsz = cfg.imgsz
        self.in_queue = mp.Queue(self.num_detector)
        self.out_queue = mp.Queue(self.num_detector)
        self.detectors = []
        self.device = cfg.device
        for i in range(self.num_detector):
            cfg.device = self.device[i]
            self.detectors.append(self._DetectWorker(self.in_queue, self.out_queue, cfg))
            self.detectors[-1].start()

        # #  初始化
        for _ in range(self.num_detector):
            _ = self.out_queue.get()

    @staticmethod
    def predict(in_queue, out_queue, idx, shm_name, batch_shape):
        in_queue.put((idx, shm_name, batch_shape))

        while True:
            out_idx, res = out_queue.get()
            if out_idx == idx:
                return res
            out_queue.put((out_idx, res))

    def stop(self):
        for _ in self.detectors:
            if self.in_queue.full():
                self.in_queue.get()
            self.in_queue.put((StopToken(), None, None))
        
        while not self.out_queue.empty():
            self.out_queue.get()