import os
import cv2
import numpy as np
import threading
import queue
import time

class LoadWebcam(threading.Thread):
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
                return img0
            else:
                return np.zeros((*self.get_shape()[::-1], 3), dtype=np.uint8)


class LoadVideo:
    def __init__(self, source: str, name: str = None):
        print(f'LoadVideo: {source}')
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.ret = True

    def get_shape(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def get_fps(self):
        return self.fps

    def state(self):
        return self.ret

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        self.ret, img0 = self.cap.read()

        if self.ret:
            return img0
        else:
            return np.zeros((*self.get_shape()[::-1], 3), dtype=np.uint8)
        
    def stop(self):
        pass

    # def __len__(self):
    #     return self.length

