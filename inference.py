import yaml
import threading
from tqdm import tqdm
from attrdict import AttrDict
import os
import sys

from utils.multi_camera_track_pipeline import MultiCameraTrackPipeline

def get_frame(multi_cameras_track, max_frame):
    progress = tqdm(total=max_frame)
    frame_id = 0

    while True:
        _ = multi_cameras_track.get_result(list(multi_cameras_track.source_names)[0])
        frame_id += 1
        progress.update(1)
        if frame_id >= max_frame:
            break

if __name__ == '__main__':
    inference_cfg_file = sys.argv[1] if len(sys.argv) > 1 else 'inference_cfg.yaml'
    inference_config = yaml.load(open(inference_cfg_file, 'r'), Loader=yaml.FullLoader)

    config_files = inference_config['config_files']
    save_dir_name_end = inference_config['save_dir_name_end']

    for config_file in config_files:
        config = yaml.load(open(f'cfg/{config_file}', 'r'), Loader=yaml.FullLoader)
        config['MultiCameraTracker']['save_root_path'] += save_dir_name_end
        multi_cameras_track = MultiCameraTrackPipeline(AttrDict(config))
        multi_cameras_track.start()

        max_frame = config['MultiCameraTracker']['max_frame']

        threading.Thread(target=get_frame, args=(multi_cameras_track, max_frame, ), name=f"TestThread-get_frame").start()

        multi_cameras_track.join()

        # os.popen(f'rm -rf /dev/shm/psm_*').read()