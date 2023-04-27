import threading
import queue
import time
from yolov7.detect import Detect
import numpy as np
from attrdict import AttrDict
from tracker.basetrack.byte_tracker import BYTETracker

class TrackPipelineThread(threading.Thread):

    class PutQueueThread(threading.Thread):
        def __init__(self, detect_queue, load_data, name):
            threading.Thread.__init__(self)
            self.detect_queue = detect_queue
            self.load_data = load_data
            _ = next(self.load_data)
            self.name = name
            self.is_stop = False
            self.start()

        def run(self):
            while not self.is_stop:
                img0 = next(self.load_data)
                self.detect_queue.put((self.name, img0))

        def stop(self):
            self.is_stop = True

    def __init__(self, config):
        threading.Thread.__init__(self)
        self.config = config
        self.queue_size = self.config['detector']['batch_size']
        self.queue = queue.Queue(self.queue_size)
        self.is_stop = False
        self.result = {}
        self.put_queue_threads = []
        self.detect = Detect(**self.config['detector'])
        self.trackers = {}

    def run(self):
        while not self.is_stop:
            if self.queue.full():
                names = []
                im0s = []
                for _ in range(self.queue_size):
                    name, img0 = self.queue.get()
                    names.append(name)
                    im0s.append(img0)

                pred = self.detect(im0s)

                for i in range(len(names)):
                    target_output = self.trackers[names[i]].update(pred[i], im0s[i].shape, im0s[i].shape)
                    target = [[*t.tlbr, t.score, t.cls, t.track_id] for t in target_output]

                    self.result[names[i]].put((im0s[i], pred[i], target))

    def add_put_queue_thread(self, load_data, name):
        self.put_queue_threads.append(
            TrackPipelineThread.PutQueueThread(self.queue, load_data, name))
        self.result[name] = queue.Queue(self.queue_size)
        self.trackers[name] = BYTETracker(AttrDict(self.config['tracker']))

    def get_result(self, name):
        return self.result[name].get()
    
    def get_names(self):
        return self.detect.get_names()

    def stop(self):
        self.is_stop = True
