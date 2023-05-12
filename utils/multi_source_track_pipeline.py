import threading
from collections import deque
from vidgear.gears import WriteGear
from datetime import datetime
import queue
import numpy as np
import json
import atexit
import csv
import os

from utils.data_loader import LoadVideo, LoadWebcam
from utils.track_pipeline import TrackPipelineThread
from utils.multi_source_tracker import MultiSourceTracker


class BaseMultiSourceTrackPipeline(threading.Thread):
    def __init__(self, config):
        threading.Thread.__init__(self)
        self.name = "MultiSourceTrackPipeline[{}]".format(self.name)

        self.track_pipeline = TrackPipelineThread(config)
        self.load_data_threads = {}

        for source in config['sources']:
            name = source['name']
            if config['sourceType'] == 'webcam':
                self.load_data_threads[name] = LoadWebcam(**source)
            elif config['sourceType'] == 'video':
                self.load_data_threads[name] = LoadVideo(**source)

            self.track_pipeline.add_put_queue_thread(self.load_data_threads[name], name)
            
        self.source_names = self.load_data_threads.keys()

        self.track_pipeline.start()

        self.is_run = True

        self.multi_source_tracker = MultiSourceTracker(config['MultiSourceTracker'], self.source_names)
    def run(self):
        while self.is_run:
            res = {}
            single_result = {}
            for name in self.source_names:
                single_result[name] = self.track_pipeline.get_result(name)

            multi_result = self.multi_source_tracker.update(single_result)
            for name in self.source_names:
                img0, pred, _ = single_result[name]
                res[name] = (img0, pred, multi_result[name]) # img0, pred, target
            '''
            {
            'source1': (img0, pred, target),
            'source2': (img0, pred, target),
            }
            '''
            self.process_result(res)

    def stop(self):
        self.track_pipeline.stop()
        for load_data in self.load_data_threads.values():
            load_data.stop()
        self.multi_source_tracker.stop()
        self.is_run = False
        print("MultiSourceTrackPipeline stop")

    def process_result(self, res):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError

class MultiSourceTrackPipeline(BaseMultiSourceTrackPipeline):

    class FileWriter(threading.Thread):
        def __init__(self, name, load_data):
            threading.Thread.__init__(self)
            self.name = "FileWriter[{}]".format(name)
            self.is_run = True
            self.video_writer = None
            self.frame_id = 0

            self.queue = queue.Queue(128)
            self.save_path = None
            self.name = name

            self.video_params = {
                    "-input_framerate": load_data.get_fps(),
                    "-output_dimensions": load_data.get_shape(),
                    "-b:v": "500k",
                    "-maxrate": "1000k",
                    "-bufsize": "1000k"
            }

            atexit.register(self.create_file)
            # self.start()

        def run(self):
            while self.is_run:
                img, preds, targets = self.queue.get()

                for pred in preds:
                    line = [self.frame_id, *pred]
                    self.pred_csv_writer.writerow(line)

                for target in targets:
                    line = [self.frame_id, *target]
                    self.target_csv_writer.writerow(line)

                self.video_writer.write(img)
            self.video_writer.close()

        def update(self, data, save_path):
            self.frame_id += 1
            if self.save_path != save_path:
                self.save_path = save_path
                self.create_file()
            self.queue.put(data)

        def create_file(self):
            self.close_file()

            video_file_name = f'{self.name}.mp4'
            video_path = os.path.join(self.save_path, video_file_name)
            self.video_writer = WriteGear(video_path, compression_mode = True, **self.video_params)

            pred_file_name = f'{self.name}_pred.csv'
            pred_path = os.path.join(self.save_path, pred_file_name)
            self.pred_writer = open(pred_path, 'a')
            self.pred_csv_writer = csv.writer(self.pred_writer)

            target_file_name = f'{self.name}_target.csv'
            target_path = os.path.join(self.save_path, target_file_name)
            self.target_writer = open(target_path, 'a')
            self.target_csv_writer = csv.writer(self.target_writer)

        def close_file(self):
            if self.video_writer != None:
                self.frame_id = 0
                self.video_writer.close()
                self.pred_writer.close()
                self.target_writer.close()

    def __init__(self, config):
        BaseMultiSourceTrackPipeline.__init__(self, config)

        self.return_queues = {x: queue.Queue(64) for x in self.load_data_threads.keys()}

        self.save_result = config['MultiSourceTracker']['save_result']
        if self.save_result:
            self.save_root_path = config['MultiSourceTracker']['save_root_path']
            self.writers = {}
            for key, value in self.load_data_threads.items():
                self.writers[key] = self.FileWriter(key, value)
                self.writers[key].start()

    def get_datetime(self):
        # return datetime.now().strftime("%Y-%m-%d_%H-%M")
        return datetime.now().strftime("%Y-%m-%d")

    def process_result(self, res):
        if self.save_result:
            datetime = self.get_datetime()
            self.save_path = os.path.join(self.save_root_path, datetime)
            if not os.path.isdir(self.save_path):
                os.mkdir(self.save_path)

        for key, value in res.items():
            if self.return_queues[key].full():
                self.return_queues[key].get()
            self.return_queues[key].put(value)
            if self.save_result:
                self.writers[key].update(value, self.save_path)

    def get_result(self, name):
        return self.return_queues[name].get()
