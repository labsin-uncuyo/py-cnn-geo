import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from dataset.threads.sample_selector_thread_status import SampleSelectorThreadStatus


class BandAnalyzerThread(multiprocessing.Process):

    # def __init__(self, i, raster_array, raster_filepath, target=None, name=None, daemon=None):
    def __init__(self, i, db, bigdata, iter_bigdata_idx_0, iter_bigdata_idx_1, band, storage_folder, edges_0=None,
                 values_0=None, percentages_0=None, edges_1=None, values_1=None, percentages_1=None):
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

        sel_bins = 'fd' if self.edges_0 is None else self.edges_0[self.band]

        h_values, h_edges, h_lower, h_upper, h_lower_outliers, h_upper_outliers = \
            self.outlier_aware_hist(items, sel_bins, *self.calculate_bounds(items, z_thresh=4.5), data_min=-1,
                                    data_max=1)

        # h_values, h_edges = np.histogram(items, bins=sel_bins)

        h_percentages = np.multiply(
            np.divide(h_values[h_values != 0] if self.values_0 is None else h_values[self.values_0[self.band] != 0],
                      items.shape[0]), 100.0)

        analysis_band_path = join(self.storage_folder, "band_{:02d}_cls_{:02d}_histogram_info.npz".format(self.band, 0))
        np.savez_compressed(analysis_band_path, h_values=h_values, h_edges=h_edges,h_lower=h_lower, h_upper=h_upper,
                            h_lower_outliers=h_lower_outliers,h_upper_outliers=h_upper_outliers,
                            h_percentages=h_percentages)

        if self.percentages_0 is not None:
            rel_err = np.abs(self.percentages_0[self.band] - h_percentages) / self.percentages_0[self.band]

            err_mean = np.mean(rel_err * 100)
            err_median = np.median(rel_err * 100)

            analysis_band_path = join(self.storage_folder,
                                      "band_{:02d}_cls_{:02d}_histogram_err.npz".format(self.band, 0))
            np.savez_compressed(analysis_band_path, rel_err=rel_err, err_mean=err_mean, err_median=err_median)

        items = np.zeros(shape=(self.iter_bigdata_idx_1.shape[0]), dtype=np.float32)

        for i, item in enumerate(self.iter_bigdata_idx_1):
            items[i] = self.bigdata[item[0]][self.band][item[1]][item[2]]

        sel_bins = 'fd' if self.edges_1 is None else self.edges_1[self.band]

        h_values, h_edges, h_lower, h_upper, h_lower_outliers, h_upper_outliers = \
            self.outlier_aware_hist(items, sel_bins, *self.calculate_bounds(items, z_thresh=4.5), data_min=-1,
                                    data_max=1)
        #h_values, h_edges = np.histogram(items, bins=sel_bins)

        h_percentages = np.multiply(
            np.divide(h_values[h_values != 0] if self.values_1 is None else h_values[self.values_1[self.band] != 0],
                      items.shape[0]), 100.0)

        analysis_band_path = join(self.storage_folder, "band_{:02d}_cls_{:02d}_histogram_info.npz".format(self.band, 1))
        np.savez_compressed(analysis_band_path, h_values=h_values, h_edges=h_edges, h_lower=h_lower, h_upper=h_upper,
                            h_lower_outliers=h_lower_outliers, h_upper_outliers=h_upper_outliers,
                            h_percentages=h_percentages)

        if self.percentages_1 is not None:
            rel_err = np.abs(self.percentages_1[self.band] - h_percentages) / self.percentages_1[self.band]

            err_mean = np.mean(rel_err * 100)
            err_median = np.median(rel_err * 100)

            analysis_band_path = join(self.storage_folder,
                                      "band_{:02d}_cls_{:02d}_histogram_err.npz".format(self.band, 1))
            np.savez_compressed(analysis_band_path, rel_err=rel_err, err_mean=err_mean, err_median=err_median)

        self.status = SampleSelectorThreadStatus.STATUS_DONE
        self.db.set("status", self.status.encode('utf-8'))

    def outlier_aware_hist(self, data, sel_bins, lower=None, upper=None, data_min=None, data_max=None):
        if not lower or lower < data.min():
            lower = data.min() if data_min is None else data_min
            lower_outliers = False
        else:
            lower_outliers = True

        if not upper or upper > data.max():
            upper = data.max() if data_max is None else data_max
            upper_outliers = False
        else:
            upper_outliers = True

        n, bins = plt.histogram(data, range=(lower, upper), bins=sel_bins)

        if lower_outliers:
            n_lower_outliers = (data < lower).sum()
            '''patches[0].set_height(patches[0].get_height() + n_lower_outliers)
            patches[0].set_facecolor('c')
            patches[0].set_label(
                'Lower outliers: ({:.2f}, {:.2f})'.format(data.min() if data_min is None else data_min, lower))'''

        if upper_outliers:
            n_upper_outliers = (data > upper).sum()
            '''patches[-1].set_height(patches[-1].get_height() + n_upper_outliers)
            patches[-1].set_facecolor('m')
            patches[-1].set_label(
                'Upper outliers: ({:.2f}, {:.2f})'.format(upper, data.max() if data_max is None else data_max))'''

        if lower_outliers or upper_outliers:
            plt.legend()

        return n, bins, lower, upper, n_lower_outliers if lower_outliers else None, n_upper_outliers if upper_outliers else None

    def mad(self, data):
        median = np.median(data)
        diff = np.abs(data - median)
        mad = np.median(diff)
        return mad

    def calculate_bounds(self, data, z_thresh=3.5):
        MAD = self.mad(data)
        median = np.median(data)
        const = z_thresh * MAD / 0.6745
        return (median - const, median + const)
