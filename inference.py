import yaml
import threading
from tqdm import tqdm
from attrdict import AttrDict
import os

from utils.multi_source_track_pipeline import MultiSourceTrackPipeline

def get_frame(multi_sources_track, max_frame):
    progress = tqdm(total=max_frame)
    frame_id = 0

    while True:
        _ = multi_sources_track.get_result("cam1")
        frame_id += 1
        progress.update(1)
        if frame_id >= max_frame:
            break

if __name__ == '__main__':
    config_files = ['nd_v2_p1.yaml', 'nd_v2_p2.yaml', 'nd_v2_p4.yaml', 'nd_v2_p5.yaml', 'nd_v2_p6.yaml', 'nd_v2_p7.yaml']
    for config_file in config_files:
        config = yaml.load(open(f'cfg/{config_file}', 'r'), Loader=yaml.FullLoader)
        config['MultiSourceTracker']['save_root_path'] += '-aspectRatio'
        multi_sources_track = MultiSourceTrackPipeline(AttrDict(config))
        multi_sources_track.start()

        max_frame = config['MultiSourceTracker']['max_frame']

        threading.Thread(target=get_frame, args=(multi_sources_track, max_frame, ), name=f"TestThread-get_frame").start()

        multi_sources_track.join()

        os.popen(f'rm -rf /dev/shm/psm_*').read()