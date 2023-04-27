import threading
from collections import deque
import queue

from utils.data_loader import LoadVideo, LoadWebcam
from utils.track_pipeline import TrackPipelineThread
from utils.multi_source_tracker import MultiSourceTracker


class MultiSourceTrackPipeline(threading.Thread):
    def __init__(self, config):
        threading.Thread.__init__(self)
        self.name = 'MultiSourceTrackPipeline'

        self.track_pipeline = TrackPipelineThread(config)
        self.load_data_threads = []

        self.source_names = []

        for source in config['sources']:
            if config['sourceType'] == 'webcam':
                self.load_data_threads.append(LoadWebcam(**source))
            elif config['sourceType'] == 'video':
                self.load_data_threads.append(LoadVideo(**source))

            self.source_names.append(source['name'])
            self.track_pipeline.add_put_queue_thread(
                self.load_data_threads[-1], source['name'])

        self.track_pipeline.start()

        self.shutdown = False

        self.multi_source_tracker = MultiSourceTracker(config['MultiSourceTracker'], self.source_names)
        self.results = {x:queue.Queue(64) for x in self.source_names}

    def run(self):
        while not self.shutdown:
            #TODO: 增加無訊號的處理
            results = {}
            for i, load_data in enumerate(self.load_data_threads):
                results[self.source_names[i]] = self.track_pipeline.get_result(self.source_names[i])

            res = self.multi_source_tracker.update(results)
            for name in self.source_names:
                # r = list(map(lambda x: (x.prev.origin if x.prev else None, x.match_conf), res[name]))
                self.results[name].put((*results[name], res[name]))

    def stop(self):
        self.shutdown = True

    def get_result(self, name):
        return self.results[name].get()

    def get_names(self):
        return self.track_pipeline.get_names()
