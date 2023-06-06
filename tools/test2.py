import time
from typing import Any
import yaml
import threading
import statistics
from collections import deque
from multiprocessing import Process, Queue, Lock
import random

class process1(Process):
    def __init__(self, in_queue, out_queue):
        Process.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        while True:
            d = self.in_queue.get()
            self.out_queue.put(d)

class process2(Process):
    def __init__(self, n, process, lock):
        Process.__init__(self)
        self.n = n
        self.process = process
        self.lock = lock
    
    def run(self):
        while True:
            # self.lock.acquire()
            res = self.process(self.n)
            # self.lock.release()
            meg = f'{self.n}: {res}'
            if res != self.n:
                meg += ' error'
            print(meg)
            time.sleep(0.1)

class Test:
    def __init__(self):
        self.conut = 0

        self.in_queue = Queue(maxsize=10)
        self.out_queue = Queue(maxsize=10)

        self.p1 = process1(self.in_queue, self.out_queue)
        self.p1.start()

        self.result_data = {}

    def put(self, key, data):
        self.in_queue.put((key, data))
        
    def get(self, key):
        while True:
            out_key, res = self.out_queue.get()
            if out_key == key:
                return res
            self.in_queue.put((out_key, res))

    def __call__(self, n):
        self.conut += 1
        key = self.conut
        # key = name
        # print(f'put {key} {n}')
        self.put(key, n)
        # print(f'get {key} {n}')
        return self.get(key)




if __name__ == '__main__':
    test = Test()
    p2 = []
    lock = Lock()
    c = 12
    for i in range(c):
        p2.append(process2(i, test, lock))
        p2[i].start()

    for i in range(c):
        p2[i].join()


