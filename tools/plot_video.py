from collections import defaultdict
from vidgear.gears import WriteGear, CamGear
from yolov7.utils.plots import plot_one_box
import argparse
import random
import re
import os
import csv
import cv2
from tqdm import tqdm
from multiprocessing import Process

def plot_video(video_file, target_file, save_path):
    frame_targets = defaultdict(list)
    with open(target_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            frame_id, *xyxy, conf, cls, track_id, match_id, match_conf = row
            frame_id = int(frame_id)
            xyxy = tuple([float(x) for x in xyxy])
            conf = float(conf)
            cls = int(float(cls))
            track_id = int(track_id)
            match_id = int(match_id)
            if match_conf == '':
                match_conf = None
            else:
                match_conf = float(match_conf)

            frame_targets[frame_id].append((xyxy, conf, cls, track_id, match_id, match_conf))

    cap1 = cv2.VideoCapture(video_file)
    cap = CamGear(video_file).start()
    length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=length, desc=video_file.split('/')[-1])

    shape = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_params = {
        "-input_framerate": cap1.get(cv2.CAP_PROP_FPS),
        "-output_dimensions": shape,
    }
    cap1.release()
    video_writer = WriteGear(save_path, compression_mode = True, **video_params)

    frame_id = 1
    while True:
        img = cap.read()
        if img is None:
            break
        for xyxy, conf, cls, track_id, match_id, match_conf in frame_targets[frame_id]:
            label = f'{names[int(cls)]}  {int(track_id)}  {conf:.2f} #{match_id}'
            if isinstance(match_conf, float):
                label += f' {match_conf:.4f}'
            plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=line_thickness)

        video_writer.write(img)
        pbar.update(1)
        frame_id += 1


    video_writer.close()
    cap.stop()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--video-dir', type=str, help='影片目錄')
    parser.add_argument('-c', '--classes', default='classes.txt', type=str, help='classes 檔案')
    parser.add_argument('-l', '--line-thickness', type=int, default=3, help='框線粗細')
    opt = parser.parse_args()

    p = re.compile(r'.mp4', re.VERBOSE)
    files = os.listdir(opt.video_dir)
    files = [f for f in files if p.search(f)]

    random.seed(10)
    global names, colors
    with open("classes.txt", newline='') as f:
        names = f.read().split('\n')
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    global line_thickness
    line_thickness = opt.line_thickness
    path = opt.video_dir


    threads = []
    for file_name in files:
        target_name = file_name.split('.')[0] + '_target.csv'
        save_name = file_name.split('.')[0] + '_plot.mp4'
        threads.append(Process(target=plot_video, args=(
            os.path.join(path, file_name),
            os.path.join(path, target_name),
            os.path.join(path, save_name),
        )))
        threads[-1].start()

    for t in threads:
        t.join()

