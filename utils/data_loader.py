import os
import cv2
import numpy as np
import threading
import queue
import time
import re

class BaseLoader:
    def get_shape(self):
        raise NotImplementedError

    def get_fps(self):
        raise NotImplementedError

    def state(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class LoadWebcam(BaseLoader, threading.Thread):
    def __init__(self, source: str, name: str = None, skip: int = 0, buffer_size: int = 30):
        print(f'LoadWebcam: {source}')
        threading.Thread.__init__(self)
        self.skip = skip
        self.name = name
        self.burrer_size = buffer_size
        self.img0s = queue.Queue(maxsize=self.burrer_size)

        self.source = source
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

        self.is_run = True
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) / (self.skip + 1)
        self.update_time = time.time()
        self.start()

    def run(self):
        while self.is_run:
            for _ in range(self.skip+1):
                self.cap.grab()
            ret, img0 = self.cap.retrieve()
            if not ret:
                self.reconnect()
                continue
            if self.img0s.full():
                self.img0s.get()
            self.img0s.put(img0)
            self.update_time = time.time()

    def get_shape(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def get_fps(self):
        return self.fps

    def reconnect(self):
        self.cap.release()
        self.cap.open(self.source, cv2.CAP_FFMPEG)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) / (self.skip + 1)

    def stop(self):
        self.is_run = False

    def state(self):
        return not self.is_disconnect()

    # 斷線偵測
    def is_disconnect(self):
        while True:
            if self.img0s.qsize():
                break
            else:
                disconnect = (time.time() - self.update_time) > (1 / (self.fps*0.1))
                if self.img0s.empty() and disconnect:
                    return True
                elif self.img0s.empty() and not disconnect:
                    continue
        return False

    def __len__(self):
        return 0

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if not self.is_run:
            return None

        start_time = time.time()
        disconnect = self.is_disconnect()
        if disconnect:
            while True:
                if (1 / ((time.time() - start_time)+1e-5)) < self.fps:
                    break
            return np.zeros((*self.get_shape()[::-1], 3), dtype=np.uint8)
        else:
            img0 = self.img0s.get()
            if len(img0):
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                return img0
            else:
                return np.zeros((*self.get_shape()[::-1], 3), dtype=np.uint8)


class LoadVideo(BaseLoader):
    def __init__(self, source: str, name: str = None):
        print(f'LoadVideo: {source}')
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.ret = True
        self.is_run = True

    def get_shape(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def get_fps(self):
        return self.fps

    def state(self):
        return self.is_run or self.ret

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.is_run:        
            self.ret, img0 = self.cap.read()

            if self.ret:
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                return img0
            else:
                return np.zeros((*self.get_shape()[::-1], 3), dtype=np.uint8)
        else:
            return None
        
    def stop(self):
        self.is_run = False

class LoadImage(BaseLoader):
    def __init__(self, imgs_path: str, name: str = None, fps: int = 30):
        print(f'LoadImage: {imgs_path}')
        self.imgs_path = imgs_path
        self.name = name
        self.fps = fps
        self.is_run = True

        self.files = os.listdir(imgs_path)
        p = re.compile(r'.jpg|.png|.jpeg', re.VERBOSE)
        self.files = [f for f in self.files if p.search(f)]
        self.files.sort()
        self.index = 0
        self.max_index = len(self.files)

        img0 = cv2.imread(os.path.join(imgs_path, self.files[0]))
        self.shape = img0.shape[:2]
        self.shape = self.shape[::-1]

    def get_shape(self):
        return self.shape
    
    def get_fps(self):
        return self.fps

    def state(self):
        return self.is_run

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.is_run:
            self.index += 1
            if self.index < self.max_index:
                img0 = cv2.imread(os.path.join(self.imgs_path, self.files[self.index]))
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                return img0
            else:
                return np.zeros((*self.get_shape()[::-1], 3), dtype=np.uint8)
        else:
            return None
        
    def stop(self):
        self.is_run = False

