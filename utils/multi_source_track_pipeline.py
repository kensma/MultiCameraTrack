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
import random
import cv2
from attrdict import AttrDict

from utils.data_loader import LoadVideo, LoadWebcam, LoadImage
from utils.track_pipeline import TrackPipelineThread
from utils.multi_source_tracker import MultiSourceTracker
from utils.file_utls import CSVFile
from yolov7.utils.plots import plot_one_box
from utils.reid import AsyncPredictor


class BaseMultiSourceTrackPipeline(threading.Thread):
    def __init__(self, cfg):
        threading.Thread.__init__(self)
        self.name = "MultiSourceTrackPipeline[{}]".format(self.name)

        self.track_pipeline = TrackPipelineThread(cfg)
        self.load_data_threads = {}

        for source in cfg.sources:
            name = source['name']
            if cfg.sourceType == 'webcam':
                self.load_data_threads[name] = LoadWebcam(**source)
            elif cfg.sourceType == 'video':
                self.load_data_threads[name] = LoadVideo(**source)
            elif cfg.sourceType == 'image':
                self.load_data_threads[name] = LoadImage(**source)

            self.track_pipeline.add_put_queue_thread(self.load_data_threads[name], name)
            
        self.source_names = self.load_data_threads.keys()
        self.frame_id = 0
        self.max_frame_id = cfg.MultiSourceTracker.max_frame

        self.track_pipeline.start()

        self.is_run = True

        self.predictor = AsyncPredictor(cfg.reid)

        self.multi_source_tracker = MultiSourceTracker(cfg.MultiSourceTracker, self.source_names, self.predictor)
    def run(self):
        while self.is_run:
            self.frame_id += 1
            if self.max_frame_id != -1 and self.frame_id > self.max_frame_id:
                self.stop()
                break
            
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
        for load_data in self.load_data_threads.values():
            load_data.stop()
        self.track_pipeline.stop()
        self.multi_source_tracker.stop()
        self.is_run = False
        print("MultiSourceTrackPipeline stop")

    def process_result(self, res):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError

class MultiSourceTrackPipeline(BaseMultiSourceTrackPipeline):

    class FileWriter(threading.Thread):
        def __init__(self, name, load_data, plot_result=False, line_thickness=1):
            threading.Thread.__init__(self)
            self.name = "FileWriter[{}]".format(name)
            self.is_run = True
            self.plot_result = plot_result
            self.video_writer = None
            self.frame_id = 0
            self.line_thickness = line_thickness

            self.queue = queue.Queue(128)
            self.save_path = None
            self.name = name

            self.video_params = {
                    "-input_framerate": load_data.get_fps(),
                    "-output_dimensions": load_data.get_shape(),
                    # "-b:v": "1000k",
                    # "-maxrate": "2000k",
                    # "-bufsize": "2000k"
            }

            atexit.register(self.create_file)
            # self.start()

        def run(self):
            while self.is_run:
                self.frame_id += 1
                img, preds, targets = self.queue.get()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                for pred in preds:
                    line = [self.frame_id, *pred]
                    self.pred_file.write(line)

                for target in targets:
                    line = [self.frame_id, *target]
                    self.target_file.write(line)

                if self.plot_result:
                    for *xyxy, conf, cls, track_id, match_id, match_conf in targets:
                        label = f'{names[int(cls)]}  {int(track_id)}  {conf:.2f} #{match_id}'
                        if isinstance(match_conf, float):
                            label += f' {match_conf:.4f}'
                        plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=self.line_thickness)
        
                self.video_writer.write(img)
            self.close_file()

        def update(self, data, save_path):
            # self.frame_id += 1
            if self.save_path != save_path:
                self.save_path = save_path
                self.create_file()
            self.queue.put(data)

        def create_file(self):
            self.close_file()

            video_file_name = f'{self.name}.mp4'
            video_path = os.path.join(self.save_path, video_file_name)
            self.video_writer = WriteGear(video_path, compression_mode = True, **self.video_params)

            self.pred_file = CSVFile(self.save_path, f'{self.name}_pred.csv')

            self.target_file = CSVFile(self.save_path, f'{self.name}_target.csv')

        def close_file(self):
            if self.video_writer != None:
                self.frame_id = 0
                self.video_writer.close()
                self.pred_file.close()
                self.target_file.close()
        
        def stop(self):
            self.is_run = False

    def __init__(self, cfg):
        super().__init__(cfg)

        self.return_queues = {x: queue.Queue(64) for x in self.load_data_threads.keys()}

        self.save_result = cfg.MultiSourceTracker.save_result
        if self.save_result:
            self.save_root_path = cfg.MultiSourceTracker.save_root_path
            self.plot_result =  cfg.MultiSourceTracker.plot_result
            self.plot_line_thickness = cfg.MultiSourceTracker.plot_line_thickness


            random.seed(10)
            global names, colors
            with open(cfg.detector.names, newline='') as f:
                names = f.read().split('\n')
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            self.writers = {}
            for key, value in self.load_data_threads.items():
                self.writers[key] = self.FileWriter(key, value, self.plot_result, self.plot_line_thickness)
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
        
    def stop(self):
        if self.save_result:
            for writer in self.writers.values():
                writer.stop()
        super().stop()
