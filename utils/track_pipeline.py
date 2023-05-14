import threading
import queue
import time
from yolov7.detect import Detect, AsyncDetect
import numpy as np
from attrdict import AttrDict
from tracker.basetrack.byte_tracker import BYTETracker
import torch
from yolov7.utils.datasets import letterbox

class TrackPipelineThread(threading.Thread):

    class PutQueueThread(threading.Thread):
        def __init__(self, detect_queue, load_data, name):
            threading.Thread.__init__(self)
            self.detect_queue = detect_queue
            self.load_data = load_data
            _ = next(self.load_data)
            self.name = name
            self.start()

        def run(self):
            while True:
                if pause_queue_name == self.name:
                    continue
                img0 = next(self.load_data)
                if img0 is None:
                    break
                img, _, _ = letterbox(img0, (512, 512), auto=False, scaleup=True)
                img = img.transpose(2, 0, 1)
                self.detect_queue.put((self.name, img0, img))

    def __init__(self, config):
        threading.Thread.__init__(self)
        self.name = "TrackPipelineThread[{}]".format(self.name)
        self.config = config
        self.queue_size = self.config['detector']['batch_size'] * self.config['detector']['num_detect']
        self.detect_queue = queue.Queue(self.queue_size)
        self.result = {}
        self.put_queue_threads = []
        self.source_names = []
        self.detect = AsyncDetect(AttrDict(self.config['detector']))
        self.trackers = {}

        self.is_run = True

        global pause_queue_name
        pause_queue_name = None

    def run(self):
        global pause_queue_name
        while self.is_run:
            if self.detect_queue.full():
                names = []
                im0s = []
                imgs = []
                shapes = []
                for _ in range(self.queue_size):
                    name, img0, img = self.detect_queue.get()
                    names.append(name)
                    im0s.append(img0)
                    imgs.append(img)
                    shapes.append(img0.shape)


                # t0 = time.time()
                pred = self.detect(imgs, shapes)
                # print('detect time: ', time.time() - t0)

                for i in range(len(names)):
                    target_output = self.trackers[names[i]].update(pred[i], im0s[i].shape, im0s[i].shape)
                    target = [[*t.tlbr, t.score, t.cls, t.track_id] for t in target_output]

                    self.result[names[i]].put((im0s[i], pred[i], target))

                    # 防止某個queue太多，導致其他queue都要等待
                    qsizes = [self.result
                    [name].qsize() for name in self.source_names]
                    max_qsize_index = np.argmax(qsizes)
                    min_qsize_index = np.argmin(qsizes)
                    if qsizes[max_qsize_index] - qsizes[min_qsize_index] > self.queue_size:
                        pause_queue_name = self.source_names[max_qsize_index]
                    else:
                        pause_queue_name = None
                    # print("=====================================")
                    # print("cam1: ", self.result['cam1'].qsize())
                    # print("cam2: ", self.result['cam2'].qsize())
                    # print("pause_queue_name: ", pause_queue_name)
                    # print("=====================================")

    def add_put_queue_thread(self, load_data, name):
        self.put_queue_threads.append(
            TrackPipelineThread.PutQueueThread(self.detect_queue, load_data, name))
        self.result[name] = queue.Queue(self.queue_size*3) # 自少要有三倍的空間，不然會卡住
        self.trackers[name] = BYTETracker(AttrDict(self.config['tracker']), frame_rate=int(load_data.get_fps()))
        self.source_names.append(name)

    def get_result(self, name):
        return self.result[name].get()

    def stop(self):
        self.is_run = False
        self.detect.stop()
        print("TrackPipelineThread stop")
