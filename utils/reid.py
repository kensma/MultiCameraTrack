import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as T
import cv2
import random
# import time

from solider.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224, swin_tiny_patch4_window7_224

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

class letterbox:
    def __init__(self, new_shape=(384, 128), color=(114, 114, 114), scaleup=True):
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        self.new_shape = new_shape
        self.color = color
        self.scaleup = scaleup

    def __call__(self, img):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # add border
        return img

class ImgTransform:
    def __init__(self, cfg):
        self.img_size = cfg.img_size
        self.mean = cfg.pixel_mean
        self.std = cfg.pixel_std

        self.transform = T.Compose([
            letterbox(self.img_size),
            T.ToTensor(),
            # T.Resize(self.img_size), # 維持長寬比實驗
            T.Normalize(mean=self.mean, std=self.std)
        ])
    def __call__(self, img):
        return self.transform(img)

class build_model(nn.Module):
    def __init__(self, cfg):
        super(build_model, self).__init__()
        if cfg.model_type == 'base':
            model = swin_base_patch4_window7_224
        elif cfg.model_type == 'small':
            model = swin_small_patch4_window7_224
        elif cfg.model_type == 'tiny':
            model = swin_tiny_patch4_window7_224

        self.base = model(
            img_size = cfg.img_size,
            drop_path_rate=0.1,
            drop_rate= 0.0,
            attn_drop_rate=0.0,
            pretrained=None,
            convert_weights=True,
            semantic_weight=cfg.semantic_weight
        )
        self.base.to(cfg.device)


    def forward(self, x):
        return self.base(x)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))

class AsyncPredictor:

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, in_queue, out_queue):
            mp.Process.__init__(self)
            self.cfg = cfg
            self.in_queue = in_queue
            self.out_queue = out_queue

        @torch.no_grad()
        def run(self):
            predictor = build_model(self.cfg)
            predictor.load_param(self.cfg.param)
            predictor.to(self.cfg.device)
            predictor.eval()
            self.out_queue.put("OK")

            while True:
                task = self.in_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task

                imgs_tensor = data.to(self.cfg.device).float()
                # t0 = time.time()
                result = predictor(imgs_tensor)
                # print('time: ', time.time() - t0)
                self.out_queue.put((idx, result))

    def __init__(self, cfg):
        self.in_queue = mp.Queue(maxsize=10)
        self.out_queue = mp.Queue(maxsize=10)

        self.procs = AsyncPredictor._PredictWorker(cfg, self.in_queue, self.out_queue)
        self.procs.start()

        self.img_transform = ImgTransform(cfg)

        _ = self.out_queue.get()
        self.conut = 0

    def put(self, key, data):
        self.in_queue.put((key, data))

    def get(self, key):
        while True:
            out_key, res = self.out_queue.get()
            if out_key == key:
                return res
            self.in_queue.put((out_key, res))
    
    def __call__(self, images):
        self.conut += 1
        key = self.conut
        self.put(key, images)
        return self.get(key)
    
    # @staticmethod
    # def predict(in_queue, out_queue, idx, data):
    #     in_queue.put((idx, data))

    #     while True:
    #         out_idx, res = out_queue.get()
    #         if out_idx == idx:
    #             return res
    #         in_queue.put((out_idx, res))