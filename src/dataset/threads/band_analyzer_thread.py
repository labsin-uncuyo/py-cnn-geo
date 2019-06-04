import multiprocessing
import numpy as np
import sys
from dataset.threads.sample_selector_thread_status import SampleSelectorThreadStatus


class BandAnalyzerThread(multiprocessing.Process):

    # def __init__(self, i, raster_array, raster_filepath, target=None, name=None, daemon=None):
    def __init__(self, i, db, bigdata, iter_bigdata_idx_0, iter_bigdata_idx_1, partitions, band):
        super().__init__(target=self.run)
        self.i = i
        self.bigdata = bigdata
        self.iter_bigdata_idx_0 = iter_bigdata_idx_0
        self.iter_bigdata_idx_1 = iter_bigdata_idx_1
        self.partitions = partitions
        self.band = band

        self.total = None
        self.processed = None

        self.db = db

        self.status = SampleSelectorThreadStatus.STATUS_LOADING_FILES
        self.db.set("status", self.status.encode('utf-8'))

    def run(self):
        item_counter_0 = np.zeros(shape=(self.partitions + 1), dtype=np.uint32)
        item_counter_1 = np.zeros(shape=(self.partitions + 1), dtype=np.uint32)

        for item in self.iter_bigdata_idx_0:
            bucket = self.bigdata[item[0]][self.band][item[1]][item[2]]
            item_counter_0[bucket] += 1

        self.db.set("item_counter_0", np.array2string(item_counter_0, separator=';', max_line_width=sys.maxsize).strip('[]').replace(' ', '').encode('utf-8'))

        for item in self.iter_bigdata_idx_1:
            bucket = self.bigdata[item[0]][self.band][item[1]][item[2]]
            item_counter_1[bucket] += 1

        self.db.set("item_counter_1", np.array2string(item_counter_1, separator=';', max_line_width=sys.maxsize).strip('[]').replace(' ', '').encode('utf-8'))

        self.status = SampleSelectorThreadStatus.STATUS_DONE
        self.db.set("status", self.status.encode('utf-8'))