from yolov7.detect import  AsyncDetect
import yaml
from attrdict import AttrDict
import numpy as np
import time

if __name__ == '__main__':
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    detect = AsyncDetect(AttrDict(config['detector']))

    while True:
        imgs = [np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8) for _ in range(64)]
        # t0 = time.time()
        pred = detect(imgs)
        # time.sleep(0.1)
        # print('detect time: ', time.time() - t0)
        # print("=====================================")