import threading
import queue
import time
from yolov7.detect import Detect, AsyncDetect
import numpy as np
from attrdict import AttrDict
from tracker.basetrack.byte_tracker import BYTETracker
import torch
from yolov7.utils.datasets import letterbox
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from functools import reduce

class TrackPipelineThread(threading.Thread):

    class PutQueueThread(threading.Thread):
        def __init__(self, detect_queue, batch_size, smm, load_data, name):
            threading.Thread.__init__(self)
            self.detect_queue = detect_queue
            self.load_data = load_data
            _ = next(self.load_data)
            self.name = name
            self.shape = self.load_data.get_shape()
            self.shape = (self.shape[1], self.shape[0], 3)
            self.batch_size = batch_size
            self.batch_shape = (batch_size, *self.shape)
            self.smm = smm

            self.start()

        def run(self):
            cost = 0
            shm = self.smm.SharedMemory(size=reduce(lambda x, y: x * y, self.batch_shape))
            shm_array = np.ndarray(self.batch_shape, dtype=np.uint8, buffer=shm.buf)
            while True:
                if pause_queue_name == self.name:
                    continue
                img0 = next(self.load_data)
                if img0 is None:
                    break

                shm_array[cost, :] = img0
                if cost == self.batch_size-1:
                    self.detect_queue.put((self.name, shm.name, self.batch_shape))

                    shm.close()
                    cost = 0
                    shm = self.smm.SharedMemory(size=reduce(lambda x, y: x * y, self.batch_shape))
                    shm_array = np.ndarray(self.batch_shape, dtype=np.uint8, buffer=shm.buf)
                else:
                    cost += 1

    def __init__(self, config):
        threading.Thread.__init__(self)
        self.name = "TrackPipelineThread[{}]".format(self.name)
        self.config = config
        self.batch_size = self.config['detector']['batch_size'] * self.config['detector']['num_detect']
        self.detect_queue = queue.Queue(self.config['detector']['detect_queue_size'])
        self.result = {}
        self.put_queue_threads = []
        self.source_names = []
        self.detect = AsyncDetect(AttrDict(self.config['detector']))
        self.trackers = {}

        self.is_run = True

        global pause_queue_name
        pause_queue_name = None

        self.smm = SharedMemoryManager()
        self.smm.start()

    def run(self):
        global pause_queue_name
        while self.is_run:
            name, shm_name, batch_shape = self.detect_queue.get()
            shm = shared_memory.SharedMemory(name=shm_name)
            im0s = np.ndarray(batch_shape, dtype=np.uint8, buffer=shm.buf)

            # t0 = time.time()
            pred = self.detect(shm_name, batch_shape)
            # pred = np.zeros((self.batch_size, 1, 6))
            # print('detect time: ', time.time() - t0)

            for i in range(len(im0s)):
                target_output = self.trackers[name].update(pred[i], im0s[i].shape, im0s[i].shape)
                target = [[*t.tlbr, t.score, t.cls, t.track_id] for t in target_output]

                self.result[name].put((im0s[i].copy(), pred[i], target))

                # 防止某個queue太多，導致其他queue都要等待
                qsizes = [self.result
                [name].qsize() for name in self.source_names]
                max_qsize_index = np.argmax(qsizes)
                min_qsize_index = np.argmin(qsizes)
                if qsizes[max_qsize_index] - qsizes[min_qsize_index] > self.batch_size:
                    pause_queue_name = self.source_names[max_qsize_index]
                else:
                    pause_queue_name = None
                # print("=====================================")
                # for name in self.source_names:
                #     print(f"{name}: ", self.result[name].qsize())
                # print("pause_queue_name: ", pause_queue_name)
                # print("=====================================")
            
            shm.close()
            shm.unlink()

    def add_put_queue_thread(self, load_data, name):
        self.put_queue_threads.append(
            TrackPipelineThread.PutQueueThread(self.detect_queue, self.batch_size, self.smm, load_data, name))
        self.result[name] = queue.Queue(self.batch_size*3) # 自少要有三倍的空間，不然會卡住
        self.trackers[name] = BYTETracker(AttrDict(self.config['tracker']), frame_rate=int(load_data.get_fps()))
        self.source_names.append(name)

    def get_result(self, name):
        return self.result[name].get()

    def stop(self):
        self.is_run = False
        self.detect.stop()
        print("TrackPipelineThread stop")
