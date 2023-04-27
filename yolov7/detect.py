import torch
import sys
import cv2
import numpy as np
import random

sys.path.insert(1, './yolov7')
# sys.path.append('./yolov7')
# sys.path.append('.')

from yolov7.utils.torch_utils import select_device
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import scale_coords, non_max_suppression, check_img_size
from yolov7.utils.plots import plot_one_box
from yolov7.models.experimental import attempt_load
from yolov7.utils.torch_utils import TracedModel

class Detect:
    @torch.no_grad()
    def __init__(self, weights, device, imgsz=512, half=False, batch_size=1, trace=True, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=True):
        self.device = device
        self._half = half
        self._batch_size = batch_size
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._classes = classes
        self._agnostic_nms = agnostic_nms

        self.device = select_device(self.device)
        self._half = self._half and self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self._stride = int(self.model.stride.max())  # model stride
        self._imgsz = check_img_size(imgsz, s=self._stride)  # check img_size
        
        if trace:
            self.model = TracedModel(self.model, self.device, self._imgsz)

        if half:
            self.model.half()

        self._names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        self.model(torch.zeros(self._batch_size, 3, self._imgsz, self._imgsz).to(self.device).type_as(next(self.model.parameters())))

    @torch.no_grad()
    def __call__(self, im0s):
        try:
            imgs = []
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
                det[:, :4] = scale_coords(imgs[i].shape[1:], det[:, :4], im0s[i].shape).round()
                res.append(det.cpu().detach().numpy())

            return res
        except Exception as e:
            raise e

    def get_names(self):
        return self._names