# import yaml
# from attrdict import AttrDict
# from utils.reid import AsyncPredictor, ImgTransform
# import torch
# import numpy as np
# import time
# import random
# from multiprocessing.managers import SharedMemoryManager
# from yolov7.detect import Detect, AsyncDetect
# from utils.track_pipeline import TrackPipelineProcess

# if __name__ == '__main__':
#     config = yaml.load(open('config5.yaml', 'r'), Loader=yaml.FullLoader)
#     cfg = AttrDict(config)

#     # 初始化 detect
#     detect = AsyncDetect(cfg.detector)
#     detect_in = detect.in_queue
#     detect_out = detect.out_queue
#     # detect = None

#     # 初始化共享記憶體
#     smm_address=('', 50000)
#     smm = SharedMemoryManager(address=smm_address)
#     smm.start()

#     track_pipeline_processes = {}
#     for source in cfg.sources[:1]:
#         track_pipeline_processes[source['name']] = TrackPipelineProcess(cfg, smm_address, (detect_in, detect_out), source)
#         track_pipeline_processes[source['name']].start()

#     for track_pipeline in track_pipeline_processes.values():
#         track_pipeline.join()

a = 0
b = 1
for _ in range(10):
    a += 1
    b += 1
    print(a, b)
