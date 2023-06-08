import csv
import os

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
