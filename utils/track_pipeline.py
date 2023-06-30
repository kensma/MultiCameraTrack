import threading
import queue
import time
from yolov7.detect import AsyncDetector
import numpy as np
from tracker.basetrack.byte_tracker import BYTETracker
from multiprocessing import shared_memory, Process, Queue, Value
from multiprocessing.managers import SharedMemoryManager
from functools import reduce
import sys

from utils.data_loader import LoadVideo, LoadWebcam, LoadImage
from utils.utils import StopToken


class TrackPipelineProcess(Process):

    class PutQueueThread(threading.Thread):
        def __init__(self, smm_address, batch_queue, is_run, batch_size, source, type):
            threading.Thread.__init__(self)
            self.name = "PutQueueThread[{}]".format(self.name)

            self.is_run = is_run

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

            self.smm = SharedMemoryManager(address=smm_address)
            self.buf_size = reduce(lambda x, y: x * y, self.batch_shape)

            self.start()


        def run(self):
            conut = 0
            shm = self.smm.SharedMemory(size=self.buf_size)
            shm_array = np.ndarray(self.batch_shape, dtype=np.uint8, buffer=shm.buf)
            while self.is_run.value:
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

            shm.close()
            shm.unlink()
            self.load_data.stop()
            

    #平行化比較實驗
    # def __init__(self, cfg, smm_address, detect, source, locks):
    def __init__(self, cfg, smm_address, detect, source):
        Process.__init__(self)
        self.name = "TrackPipelineProcess[{}]".format(self.name)
        self.cfg = cfg
        self.batch_size = self.cfg.detector.batch_size
        self.source_name = source.name
        self.result = Queue(self.cfg.detector.detector_queue_size*3) # 自少要有三倍的空間，不然會卡住
        self.source = source
        # self.is_run = True
        self.smm_address = smm_address

        self.detect_in, self.detect_out = detect

        self.is_run = Value('b', True)

        # self.locks = locks #平行化比較實驗

    def run(self):
        batch_queue = queue.Queue(self.cfg.detector.detector_queue_size)
        put_queue_thread = self.PutQueueThread(self.smm_address, batch_queue, self.is_run, self.batch_size, self.source, self.cfg.sourceType)

        tracker = BYTETracker(self.cfg.tracker, frame_rate=int(put_queue_thread.load_data.get_fps()))
        conut = 0
        while self.is_run.value:
            # self.locks[0].acquire() #平行化比較實驗
            shm_name, batch_shape = batch_queue.get()
            shm = shared_memory.SharedMemory(name=shm_name)
            im0s = np.ndarray(batch_shape, dtype=np.uint8, buffer=shm.buf)
            # self.locks[0].release() #平行化比較實驗

            # self.locks[1].acquire() #平行化比較實驗
            pred = AsyncDetector.predict(self.detect_in, self.detect_out, self.source_name, shm_name, batch_shape)
            # self.locks[1].release() #平行化比較實驗

            # self.locks[2].acquire() #平行化比較實驗
            targets = []
            for i in range(self.batch_size):
                conut += 1
                target_output = tracker.update(pred[i], im0s[i].shape, im0s[i].shape)
                target = [[*t.tlbr, t.score, t.cls, t.track_id] for t in target_output]
                targets.append(target)
            # self.locks[2].release() #平行化比較實驗

            self.result.put(((shm_name, batch_shape), pred, targets))
            
            shm.close()

        batch_queue.get()
        while not self.result.empty():
            self.result.get()
        self.result.close()
        self.result.join_thread()
        print(f"{self.source_name} TrackPipelineProcess stop")

    def get_result(self):
        return self.result.get()

    def stop(self):
        self.is_run.value = False
        self.result.get()
