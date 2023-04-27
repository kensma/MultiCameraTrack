import os
import cv2
import numpy as np
import threading
import queue

#TODO: 增加無訊號的處理
class LoadWebcam(threading.Thread):
    def __init__(self, source: str, name: str = None, skip: int = 0, buffer_size: int = 30):
        print(f'LoadWebcam: {source}')
        threading.Thread.__init__(self)
        self.skip = skip
        self.name = name
        self.burrer_size = buffer_size
        self.img0s = queue.Queue(maxsize=self.burrer_size)

        self._source = source
        self._cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.start()

    def run(self):
        while True:
            for _ in range(self.skip+1):
                self._cap.grab()
            ret, img0 = self._cap.retrieve()
            assert ret, f'Camera Error {self._source}'
            if self.img0s.full():
                self.img0s.get()
            self.img0s.put(img0)

    def get_shape(self):
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        img0 = self.img0s.get()
        return img0

    def __len__(self):
        return 0


class LoadVideo:
    def __init__(self, source: str, name: str = None):
        print(f'LoadVideo: {source}')
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.count = 0

    def get_shape(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if self.count >= self.length:
            self.count = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.count += 1
        ret, img0 = self.cap.read()

        assert ret, f'Load Error {self.source}'
        return img0

    def __len__(self):
        return self.length

