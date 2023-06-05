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
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager

from utils.data_loader import LoadVideo, LoadWebcam, LoadImage
from utils.track_pipeline import TrackPipelineProcess
from utils.multi_source_tracker import MultiSourceTracker
from utils.file_utls import CSVFile
from yolov7.utils.plots import plot_one_box
from yolov7.detect import Detect, AsyncDetect
from utils.reid import AsyncPredictor


class BaseMultiSourceTrackPipeline(threading.Thread):
    def __init__(self, cfg):
        threading.Thread.__init__(self)
        self.name = "MultiSourceTrackPipeline[{}]".format(self.name)

        # 初始化 detect
        self.detect = AsyncDetect(cfg.detector)
        self.detect_in = self.detect.in_queue
        self.detect_out = self.detect.out_queue
        self.batch_size = cfg.detector.batch_size

        # # 初始化共享記憶體
        self.smm_address=('', 50000)
        self.smm = SharedMemoryManager(address=self.smm_address)
        self.smm.start()

        # 初始化 reid
        # self.predictor = AsyncPredictor(cfg.reid)

        # 初始化 track_pipeline
        self.track_pipeline_processes = {}
        for source in cfg.sources:
            self.track_pipeline_processes[source['name']] = TrackPipelineProcess(cfg, self.smm_address, (self.detect_in, self.detect_out), source)
            self.track_pipeline_processes[source['name']].start()

        self.source_names = self.track_pipeline_processes.keys()
        self.frame_id = 0
        self.max_frame_id = cfg.MultiSourceTracker.max_frame

        self.is_run = True

        # 初始化 multi_source_tracker
        self.multi_source_tracker = MultiSourceTracker(cfg.MultiSourceTracker, cfg.reid, self.source_names)
    def run(self):
        while self.is_run:
            preds = {}
            tracks = {}
            shms = {}
            im0s = {}
            for name, track_pipeline in self.track_pipeline_processes.items():
                track_result = track_pipeline.get_result()
                (shm_name, batch_shape), preds[name], tracks[name]  = track_result
                shms[name] = shared_memory.SharedMemory(name=shm_name)
                im0s[name] = np.ndarray(batch_shape, dtype=np.uint8, buffer=shms[name].buf)

            for i in range(self.batch_size):
                self.frame_id += 1
                if self.max_frame_id != -1 and self.frame_id > self.max_frame_id:
                    self.stop()
                    break

                res = {}
                temp = {}
                for name in self.source_names:
                    temp[name] = (im0s[name][i], tracks[name][i])

                multi_result = self.multi_source_tracker.update(temp)

                for name in self.source_names:
                    res[name] = (im0s[name][i].copy(), preds[name][i], multi_result[name])
                '''
                {
                'source1': (img0, pred, target),
                'source2': (img0, pred, target),
                }
                '''
                self.process_result(res)

            for shm in shms.values():
                shm.close()
                shm.unlink()

    def stop(self):
        for track_pipeline in self.track_pipeline_processes.values():
            track_pipeline.stop()
        for track_pipeline in self.track_pipeline_processes.values():
            track_pipeline.join()

        self.multi_source_tracker.stop()
        self.is_run = False

        self.smm.shutdown()
        
        print("MultiSourceTrackPipeline stop")

    def process_result(self, res):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError

class MultiSourceTrackPipeline(BaseMultiSourceTrackPipeline):

    class FileWriter(threading.Thread):
        def __init__(self, name, track_pipeline, save_target, save_pred, save_video, plot_result=False, line_thickness=1):
            threading.Thread.__init__(self)
            self.name = "FileWriter[{}]".format(name)
            self.is_run = True
            self.frame_id = 0
            self.updata_queue = queue.Queue(128)
            self.save_path = None
            self.name = name
            self.is_open_file = False
            self.save_target = save_target
            self.save_pred = save_pred
            self.save_video = save_video

            self.plot_result = plot_result
            self.line_thickness = line_thickness


            # self.video_params = {
            #         "-input_framerate": track_pipeline.fps,
            #         "-output_dimensions": track_pipeline.img_shape,
            #         # "-b:v": "1000k",
            #         # "-maxrate": "2000k",
            #         # "-bufsize": "2000k"
            # }


        def run(self):
            while self.is_run:
                self.frame_id += 1
                img, preds, targets = self.updata_queue.get()

                if self.save_pred:
                    for pred in preds:
                        line = [self.frame_id, *pred]
                        self.pred_file.write(line)

                if self.save_target:
                    for target in targets:
                        line = [self.frame_id, *target]
                        self.target_file.write(line)

                # if self.save_video:
                #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                #     if self.plot_result:
                #         for *xyxy, conf, cls, track_id, match_id, match_conf in targets:
                #             label = f'{names[int(cls)]}  {int(track_id)}  {conf:.2f} #{match_id}'
                #             if isinstance(match_conf, float):
                #                 label += f' {match_conf:.4f}'
                #             plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=self.line_thickness)
            
                #     self.video_writer.write(img)
            self.close_file()

        def update(self, data, save_path):
            if self.save_path != save_path:
                self.save_path = save_path
                self.create_file()
            self.updata_queue.put(data)

        def create_file(self):
            self.close_file()
            self.is_open_file = True

            # if self.save_video:
            #     video_file_name = f'{self.name}.mp4'
            #     video_path = os.path.join(self.save_path, video_file_name)
            #     self.video_writer = WriteGear(video_path, compression_mode = True, **self.video_params)

            if self.save_pred:
                self.pred_file = CSVFile(self.save_path, f'{self.name}_pred.csv')

            if self.save_target:
                self.target_file = CSVFile(self.save_path, f'{self.name}_target.csv')

        def close_file(self):
            if self.is_open_file:
                self.frame_id = 0
                # if self.save_video:
                #     self.video_writer.close()
                if self.save_target:
                    self.pred_file.close()
                if self.save_pred:
                    self.target_file.close()
                
                self.is_open_file = False
        
        def stop(self):
            self.is_run = False

    def __init__(self, cfg):
        super().__init__(cfg)

        self.return_queues = {x: queue.Queue(64) for x in self.source_names}

        self.save_target = cfg.MultiSourceTracker.save_target
        self.save_pred = cfg.MultiSourceTracker.save_pred
        self.save_video = cfg.MultiSourceTracker.save_video

        self.plot_result =  cfg.MultiSourceTracker.plot_result
        self.plot_line_thickness = cfg.MultiSourceTracker.plot_line_thickness

        self.save_result = self.save_target or self.save_pred or self.save_video
        if self.save_result:
            self.save_root_path = cfg.MultiSourceTracker.save_root_path

            random.seed(10)
            global names, colors
            with open(cfg.detector.names, newline='') as f:
                names = f.read().split('\n')
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            self.writers = {}
            for key, value in self.track_pipeline_processes.items():
                self.writers[key] = self.FileWriter(key, value, self.save_target, self.save_pred, self.save_video, self.plot_result, self.plot_line_thickness)
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
