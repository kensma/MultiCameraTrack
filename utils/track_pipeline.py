import threading
import queue
import time
from yolov7.detect import Detect, AsyncDetect
import numpy as np
from attrdict import AttrDict
from tracker.basetrack.byte_tracker import BYTETracker
import torch
from yolov7.utils.datasets import letterbox
from multiprocessing import shared_memory, Process, Queue
from multiprocessing.managers import SharedMemoryManager
from functools import reduce

from utils.data_loader import LoadVideo, LoadWebcam, LoadImage

class TrackPipelineProcess(Process):

    class PutQueueThread(threading.Thread):
        def __init__(self, smm, batch_queue, batch_size, source, type):
            threading.Thread.__init__(self)

            self.is_run = True

            if type == "video":
                self.load_data = LoadVideo(**dict(source))
            elif type == "webcam":
                self.load_data = LoadWebcam(**dict(source))
            elif type == "image":
                self.load_data = LoadImage(**dict(source))
            _ = next(self.load_data)

            self.shape = self.load_data.get_shape()
            self.shape = (self.shape[1], self.shape[0], 3)

            self.batch_queue = batch_queue
            self.batch_size = batch_size
            self.batch_shape = (batch_size, *self.shape)

            self.smm = smm
            self.buf_size = reduce(lambda x, y: x * y, self.batch_shape)

            self.start()


        def run(self):
            conut = 0
            shm = self.smm.SharedMemory(size=self.buf_size)
            shm_array = np.ndarray(self.batch_shape, dtype=np.uint8, buffer=shm.buf)
            while self.is_run:
                img0 = next(self.load_data)
                if img0 is None:
                    break

                shm_array[conut, :] = img0
                if conut == self.batch_size - 1:
                    self.batch_queue.put((shm.name, self.batch_shape))
                    shm.close()
                    conut = 0
                    shm = self.smm.SharedMemory(size=self.buf_size)
                    shm_array = np.ndarray(self.batch_shape, dtype=np.uint8, buffer=shm.buf)
                else:
                    conut += 1

        def stop(self):
            self.is_run = False
            self.load_data.stop()

    def __init__(self, cfg, smm_address, detect, source):
        Process.__init__(self)
        self.name = "TrackPipelineProcess[{}]".format(self.name)
        self.cfg = cfg
        self.batch_size = self.cfg.detector.batch_size
        self.source_name = source.name
        self.result = Queue(self.cfg.detector.detect_queue_size*3) # 自少要有三倍的空間，不然會卡住
        self.source = source
        self.is_run = True
        self.smm_address = smm_address

        self.detect_in, self.detect_out = detect

        # self.fps = 0
        # self.img_shape = None

    def run(self):
        smm = SharedMemoryManager(address=self.smm_address)

        batch_queue = queue.Queue(self.cfg.detector.detect_queue_size)
        put_queue_thread = self.PutQueueThread(smm, batch_queue, self.batch_size, self.source, self.cfg.sourceType)
        # self.fps = put_queue_thread.load_data.get_fps()
        # self.img_shape = put_queue_thread.shape

        tracker = BYTETracker(self.cfg.tracker, frame_rate=int(put_queue_thread.load_data.get_fps()))

        while self.is_run:
            shm_name, batch_shape = batch_queue.get(timeout=5000)
            shm = shared_memory.SharedMemory(name=shm_name)
            im0s = np.ndarray(batch_shape, dtype=np.uint8, buffer=shm.buf)

            # pred = self.detect(shm_name, batch_shape)
            pred = AsyncDetect.predict(self.detect_in, self.detect_out, self.source_name, shm_name, batch_shape)

            targets = []
            for i in range(self.batch_size):
                target_output = tracker.update(pred[i], im0s[i].shape, im0s[i].shape)
                target = [[*t.tlbr, t.score, t.cls, t.track_id] for t in target_output]
                targets.append(target)

            self.result.put(((shm_name, batch_shape), pred, targets))
            
            shm.close()

        put_queue_thread.stop()
        put_queue_thread.join()
        smm.unlink()

    def get_result(self):
        return self.result.get()

    def stop(self):
        self.is_run = False
        print(f"{self.source_name} TrackPipelineProcess stop")
