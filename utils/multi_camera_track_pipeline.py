import threading
from datetime import datetime
import queue
import numpy as np
import os
import random
import time
import sys
import cv2
from vidgear.gears import WriteGear
from collections import defaultdict
from multiprocessing import shared_memory, Lock
from multiprocessing.managers import SharedMemoryManager

from utils.track_pipeline import TrackPipelineProcess
from utils.multi_camera_tracker import MultiCameraTracker
from utils.utils import CSVFile, StopToken, plot_one_box
from yolov7.detect import AsyncDetector


class BaseMultiCameraTrackPipeline(threading.Thread):
    def __init__(self, cfg):
        threading.Thread.__init__(self)
        self.name = "MultiCameraTrackPipeline[{}]".format(self.name)

        # 初始化 detect
        self.detect = AsyncDetector(cfg.detector)
        self.detect_in = self.detect.in_queue
        self.detect_out = self.detect.out_queue
        self.batch_size = cfg.detector.batch_size

        # # 初始化共享記憶體
        if sys.platform == 'win32':
            self.smm_address=('localhost', 50000)
        else:
            self.smm_address=('', 50000)
        self.smm = SharedMemoryManager(address=self.smm_address)
        self.smm.start()

        # 初始化 reid
        # self.extractor = AsyncExtractor(cfg.reid)

        # 平行化比較實驗
        # locks = [Lock() for _ in range(3)]

        # 初始化 track_pipeline
        self.track_pipeline_processes = {}
        for source in cfg.sources:
            self.track_pipeline_processes[source['name']] = TrackPipelineProcess(cfg, self.smm_address, (self.detect_in, self.detect_out), source)
            # self.track_pipeline_processes[source['name']] = TrackPipelineProcess(cfg, self.smm_address, (self.detect_in, self.detect_out), source, locks) #平行化比較實驗
            self.track_pipeline_processes[source['name']].start()

        self.source_param = {}
        for name, track_pipeline in self.track_pipeline_processes.items():
            fps, img_shape = track_pipeline.get_result()
            self.source_param[name] = {"fps": fps, "img_shape": img_shape}

        self.source_names = self.track_pipeline_processes.keys()
        self.frame_id = 0
        self.tolal_time = 0
        self.max_frame_id = cfg.MultiCameraTracker.max_frame

        self.is_run = True

        # 初始化 multi_camera_tracker
        self.multi_camera_tracker = MultiCameraTracker(cfg.MultiCameraTracker, cfg.reid, self.source_names)
    def run(self):
        t0 = time.time()
        while self.is_run:
            preds = {}
            tracks = {}
            shms = {}
            im0s = {}
            for name, track_pipeline in self.track_pipeline_processes.items():
                track_result = track_pipeline.get_result()
                if isinstance(track_result, StopToken):
                    break
                (shm_name, batch_shape), preds[name], tracks[name]  = track_result
                shms[name] = shared_memory.SharedMemory(name=shm_name)
                im0s[name] = np.ndarray(batch_shape, dtype=np.uint8, buffer=shms[name].buf)
            for i in range(self.batch_size):
                if isinstance(track_result, StopToken):
                    break
                self.frame_id += 1
                if self.max_frame_id != -1 and self.frame_id > self.max_frame_id:
                    self.tolal_time = time.time() - t0
                    self.stop()
                    break

                res = {}
                temp = {}
                for name in self.source_names:
                    temp[name] = (im0s[name][i], tracks[name][i])

                multi_result = self.multi_camera_tracker.update(temp)

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

        
        for track_pipeline in self.track_pipeline_processes.values():
            track_pipeline.join()
        self.detect.stop()
        self.smm.shutdown()
        print("MultiCameraTrackPipeline stop")

    def stop(self):
        self.is_run = False
        for track_pipeline in self.track_pipeline_processes.values():
            if track_pipeline.result.empty():
                track_pipeline.result.put(StopToken())
        for track_pipeline in self.track_pipeline_processes.values():
            track_pipeline.stop()

        self.multi_camera_tracker.stop()
        # self.is_run = False

    def process_result(self, res):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError

class MultiCameraTrackPipeline(BaseMultiCameraTrackPipeline):

    class FileWriter(threading.Thread):
        def __init__(self, name, source_param, save_target, save_pred, save_video, plot_result=False, line_thickness=1):
            threading.Thread.__init__(self)
            self.name = "FileWriter[{}]".format(name)
            self.updata_queue = queue.Queue(128)
            self.save_path = None
            self.source_name = name
            self.is_open_file = False
            self.save_target = save_target
            self.save_pred = save_pred
            self.save_video = save_video

            self.plot_result = plot_result
            self.line_thickness = line_thickness


            self.video_params = {
                    "-input_framerate": source_param["fps"],
                    "-output_dimensions": source_param["img_shape"],
                    # "-b:v": "1000k",
                    # "-maxrate": "2000k",
                    # "-bufsize": "2000k"
            }


        def run(self):
            while True:
                frame_id, img, preds, targets = self.updata_queue.get()

                if isinstance(frame_id, StopToken):
                    break

                if self.save_pred:
                    for pred in preds:
                        line = [frame_id, *pred]
                        self.pred_file.write(line)

                if self.save_target:
                    for target in targets:
                        line = [frame_id, *target]
                        self.target_file.write(line)

                if self.save_video:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    if self.plot_result:
                        cv2.putText(img, str(frame_id), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 112, 0), 3, cv2.LINE_AA)
                        for *xyxy, conf, cls, track_id, match_id, match_conf in targets:
                            label = f'{names[int(cls)]}  {int(track_id)}  {conf:.2f}'
                            color = [226,228,229]
                            if isinstance(match_id, int):
                                label += f' #{match_id}'
                                color = colors[int(match_id)]
                                if isinstance(match_conf, float):
                                    label += f' {match_conf:.4f}'
                            plot_one_box(xyxy, img, label=label, color=color, line_thickness=3)
            
                    self.video_writer.write(img)
            self.close_file()
            print(f'{self.name} FileWriter stop')

        def update(self, frame_id, data, save_path):
            if self.save_path != save_path:
                self.save_path = save_path
                self.create_file()
            self.updata_queue.put((frame_id, *data))

        def create_file(self):
            self.close_file()
            self.is_open_file = True

            if self.save_video:
                video_file_name = f'{self.source_name}.mp4'
                video_path = os.path.join(self.save_path, video_file_name)
                self.video_writer = WriteGear(video_path, compression_mode = True, **self.video_params)

            if self.save_pred:
                self.pred_file = CSVFile(self.save_path, f'{self.source_name}_pred.csv')

            if self.save_target:
                self.target_file = CSVFile(self.save_path, f'{self.source_name}_target.csv')

        def close_file(self):
            if self.is_open_file:
                self.frame_id = 0
                if self.save_video:
                    self.video_writer.close()
                if self.save_pred:
                    self.pred_file.close()
                if self.save_target:
                    self.target_file.close()
                
                self.is_open_file = False
        
        def stop(self):
            self.updata_queue.put((StopToken(), None, None, None))

    def __init__(self, cfg):
        super().__init__(cfg)

        self.return_queues = {x: queue.Queue(64) for x in self.source_names}

        self.save_target = cfg.MultiCameraTracker.save_target
        self.save_pred = cfg.MultiCameraTracker.save_pred
        self.save_video = cfg.MultiCameraTracker.save_video

        self.plot_result =  cfg.MultiCameraTracker.plot_result
        self.plot_line_thickness = cfg.MultiCameraTracker.plot_line_thickness

        self.save_result = self.save_target or self.save_pred or self.save_video
        if self.save_result:
            self.save_root_path = cfg.MultiCameraTracker.save_root_path

            random.seed(10)
            global names, colors
            with open(cfg.detector.names, newline='') as f:
                names = f.read().split('\n')
            colors = defaultdict(lambda :[random.randint(0, 255) for _ in range(3)])

            self.writers = {}
            for key, value in self.track_pipeline_processes.items():
                self.writers[key] = self.FileWriter(key, self.source_param[key], self.save_target, self.save_pred, self.save_video, self.plot_result, self.plot_line_thickness)
                self.writers[key].start()

    def process_result(self, res):
        if self.save_result:
            self.save_path = self.save_root_path
            if not os.path.isdir(self.save_path):
                os.mkdir(self.save_path)

        for key, value in res.items():
            if self.return_queues[key].full():
                self.return_queues[key].get()
            self.return_queues[key].put(value)
            if self.save_result:
                self.writers[key].update(self.frame_id, value, self.save_path)

    def get_result(self, name):
        return self.return_queues[name].get()
        
    def stop(self):
        with open(os.path.join(self.save_root_path, 'info.txt'), 'w') as f:
            f.write(f'max_frame: {self.max_frame_id}\n')
            f.write(f'total time: {self.tolal_time:.2f}s')
        if self.save_result:
            for writer in self.writers.values():
                writer.stop()
            for writer in self.writers.values():
                writer.join()
        super().stop()
