import numpy as np
import multiprocessing as mp
import time
import torch
import queue

imgs = [torch.zeros((1920, 1080, 3)) for _ in range(64)]
# # # imgs = torch.from_numpy(np.array(imgs))
# q = mp.Queue(2)
# q1 = queue.Queue(2)

# t0 = time.time()
# q.put(imgs)
# q.get()
# print(time.time()-t0)

# t0 = time.time()
# q1.put(imgs)
# q1.get()
# print(time.time()-t0)

class Test(mp.Process):
    def __init__(self, q):
        mp.Process.__init__(self)
        self.q = q

    def run(self):
        while True:
            t0, data = self.q.get()
            print("Process:", time.time()-t0)
            break

if __name__ == '__main__':
    q = mp.Queue()
    # q = queue.Queue()
    p = Test(q)
    p.start()
    
    t0 = time.time()
    q.put((t0, imgs))