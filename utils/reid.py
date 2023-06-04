import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.transforms as T
# import time

from solider.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224, swin_tiny_patch4_window7_224

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

class ImgTransform:
    def __init__(self, cfg):
        self.img_size = cfg.img_size
        self.mean = cfg.pixel_mean
        self.std = cfg.pixel_std

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(self.img_size),
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

    class _InitToken:
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

            while True:
                task = self.in_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task

                if isinstance(data, AsyncPredictor._InitToken):
                    self.out_queue.put((idx, None))
                    continue

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

        self.conut = 0
        _ = self.__call__(AsyncPredictor._InitToken())

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