import csv
import os
import random
import cv2
from multiprocessing import shared_memory

class StopToken:
        pass

class BaseFile:
    def write(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
    
class CSVFile(BaseFile):
    def __init__(self, path, name):
        pred_path = os.path.join(path, name)
        self.file_writer = open(pred_path, 'a')
        self.csv_writer = csv.writer(self.file_writer)
    
    def write(self, line):
        self.csv_writer.writerow(line)

    def close(self):
        self.file_writer.close()

def close_sharedMemory(shm_name):
    shm = shared_memory.SharedMemory(name=shm_name)
    shm.close()
    shm.unlink()

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

