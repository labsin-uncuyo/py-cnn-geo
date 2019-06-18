import multiprocessing
import numpy as np
from os.path import join
from dataset.threads.sample_selector_thread_status import SampleSelectorThreadStatus


class BandAnalyzerThread(multiprocessing.Process):

    # def __init__(self, i, raster_array, raster_filepath, target=None, name=None, daemon=None):
    def __init__(self, i, db, bigdata, iter_bigdata_idx_0, iter_bigdata_idx_1, band, storage_folder, edges_0=None, values_0=None, percentages_0=None, edges_1=None, values_1=None, percentages_1=None):
        super().__init__(target=self.run)
        self.i = i
        self.bigdata = bigdata
        self.iter_bigdata_idx_0 = iter_bigdata_idx_0
        self.iter_bigdata_idx_1 = iter_bigdata_idx_1
        self.band = band
        self.storage_folder = storage_folder
        self.edges_0 = edges_0
        self.values_0 = values_0
        self.percentages_0 = percentages_0
        self.edges_1 = edges_1
        self.values_1 = values_1
        self.percentages_1 = percentages_1

        self.total = None
        self.processed = None

        self.db = db

        self.status = SampleSelectorThreadStatus.STATUS_LOADING_FILES
        self.db.set("status", self.status.encode('utf-8'))

    def run(self):
        items = np.zeros(shape=(self.iter_bigdata_idx_0.shape[0]), dtype=np.float32)

        for i, item in enumerate(self.iter_bigdata_idx_0):
            items[i] = self.bigdata[item[0]][self.band][item[1]][item[2]]

        h_values, h_edges = np.histogram(items, bins=('fd' if self.edges_0 is None else self.edges_0))

        h_percentages = np.multiply(np.divide(h_values[h_values != 0] if self.values_0 is None else h_values[self.values_0 != 0], items.shape[0]), 100.0)

        analysis_band_path = join(self.storage_folder, "band_{:02d}_cls_{:02d}_histogram_info.npz".format(self.band, 0))
        np.savez_compressed(analysis_band_path, h_values=h_values, h_edges=h_edges, h_percentages=h_percentages)

        if self.percentages_0 is not None:
            rel_err = np.abs(self.percentages_0 - h_percentages) / self.percentages_0

            err_mean = np.mean(rel_err*100)
            err_median = np.median(rel_err*100)

            analysis_band_path = join(self.storage_folder,
                                      "band_{:02d}_cls_{:02d}_histogram_err.npz".format(self.band, 0))
            np.savez_compressed(analysis_band_path, rel_err=rel_err, err_mean=err_mean, err_median=err_median)

        items = np.zeros(shape=(self.iter_bigdata_idx_1.shape[0]), dtype=np.float32)

        for i, item in enumerate(self.iter_bigdata_idx_1):
            items[i] = self.bigdata[item[0]][self.band][item[1]][item[2]]

        h_values, h_edges = np.histogram(items, bins=('fd' if self.edges_1 is None else self.edges_1))

        h_percentages = np.multiply(
            np.divide(h_values[h_values != 0] if self.values_1 is None else h_values[self.values_1 != 0],
                      items.shape[0]), 100.0)

        analysis_band_path = join(self.storage_folder, "band_{:02d}_cls_{:02d}_histogram_info.npz".format(self.band, 1))
        np.savez_compressed(analysis_band_path, h_values=h_values, h_edges=h_edges, h_percentages=h_percentages)

        if self.percentages_1 is not None:
            rel_err = np.abs(self.percentages_1 - h_percentages) / self.percentages_1

            err_mean = np.mean(rel_err * 100)
            err_median = np.median(rel_err * 100)

            analysis_band_path = join(self.storage_folder,
                                      "band_{:02d}_cls_{:02d}_histogram_err.npz".format(self.band, 1))
            np.savez_compressed(analysis_band_path, rel_err=rel_err, err_mean=err_mean, err_median=err_median)

        self.status = SampleSelectorThreadStatus.STATUS_DONE
        self.db.set("status", self.status.encode('utf-8'))
