import multiprocessing
import numpy as np
from config import RasterParams, NormParameters
from raster_rw import GTiffHandler
from os import listdir, makedirs
from os.path import isfile, join, exists
from natsort import natsorted
from dataset.threads.sample_selector_thread_status import SampleSelectorThreadStatus


class SampleExtractorThread(multiprocessing.Process):

    # def __init__(self, i, raster_array, raster_filepath, target=None, name=None, daemon=None):
    def __init__(self, i, path_to_pck, db, processed_path):
        super().__init__(target=self.run)
        self.i = i
        self.path_to_pck = path_to_pck
        self.path_to_processed = processed_path + '/' + str(self.i + 1)

        self.counts = None
        self.fdominance = None
        self.mat_idxs = None
        self.mat_opposite_idxs = None

        self.total = None
        self.processed = None

        self.db = db

        self.status = SampleSelectorThreadStatus.STATUS_LOADING_FILES
        self.db.set("status", self.status.encode('utf-8'))

    def run(self):
        bigdata = np.zeros(shape=(12, RasterParams.LST_MAX_X, RasterParams.LST_MAX_Y), dtype=np.float32)
        bigdata_gt = np.zeros(shape=(RasterParams.FNF_MAX_X, RasterParams.FNF_MAX_Y), dtype=np.uint8)

        pckmetadata = np.empty(shape=(12,), dtype=object)
        pckmetadata_gt = np.empty(shape=(7,), dtype=object)
        pck_idx = int(self.i) - 1

        raster_files = [f for f in listdir(self.path_to_pck) if isfile(join(self.path_to_pck, f))]

        srtm_file_path = ''
        lst_file_paths = []
        fnf_file_path = ''
        for fle in raster_files:
            if 'SRTM' in fle[0:4]:
                srtm_file_path = fle
            elif 'LC' in fle[0:4]:
                lst_file_paths.append(fle)
            elif 'FNF' in fle[0:4]:
                fnf_file_path = fle

        lst_file_paths = natsorted(lst_file_paths, key=lambda y: y.lower())

        pckmetadata[0] = join(self.path_to_pck, srtm_file_path)

        srtm_handler = GTiffHandler()
        srtm_handler.readFile(pckmetadata[0])
        srtm_np_array = np.array(srtm_handler.src_Z)

        srtm_np_array[srtm_np_array < NormParameters.ELEVATION_MIN] = NormParameters.ELEVATION_MIN
        srtm_np_array[srtm_np_array > NormParameters.ELEVATION_MAX] = NormParameters.ELEVATION_MAX
        srtm_np_array = self.scale(srtm_np_array, -1, 1, NormParameters.ELEVATION_MIN, NormParameters.ELEVATION_MAX)
        bigdata[0] = srtm_np_array

        srtm_handler.closeFile()


        for idx, lst_file in enumerate(lst_file_paths):
            pckmetadata[idx + 1] = join(self.path_to_pck, lst_file)

            lst_handler = GTiffHandler()
            lst_handler.readFile(pckmetadata[idx + 1])

            lst_np_array = np.array(lst_handler.src_Z)

            if (idx + 1) < 10:
                lst_np_array[lst_np_array < NormParameters.REFLECTION_MIN] = NormParameters.REFLECTION_MIN
                lst_np_array[lst_np_array > NormParameters.REFLECTION_MAX] = NormParameters.REFLECTION_MAX
                lst_np_array = self.scale(lst_np_array, -1, 1, NormParameters.REFLECTION_MIN,
                                          NormParameters.REFLECTION_MAX)
            else:
                lst_np_array[lst_np_array < NormParameters.RADIANCE_MIN] = NormParameters.RADIANCE_MIN
                lst_np_array[lst_np_array > NormParameters.RADIANCE_MAX] = NormParameters.RADIANCE_MAX
                lst_np_array = self.scale(lst_np_array, -1, 1, NormParameters.RADIANCE_MIN,
                                          NormParameters.RADIANCE_MAX)
            bigdata[idx + 1] = lst_np_array

            lst_handler.closeFile()

        pckmetadata_gt[0] = join(self.path_to_pck, fnf_file_path)
        pckmetadata_gt[1] = None

        if not exists(self.path_to_processed):
            makedirs(self.path_to_processed)

        fnf_handler = GTiffHandler()
        fnf_handler.readFile(pckmetadata_gt[0])

        fnf_np_array = np.array(fnf_handler.src_Z)

        bigdata_gt = fnf_np_array

        fnf_handler.closeFile()

        self.status = SampleSelectorThreadStatus.STATUS_TAKING_SAMPLE_COORDS
        self.db.set("status", self.status.encode('utf-8'))

        unique, self.counts = np.unique(bigdata_gt, return_counts=True)

        # Checking if Forest points are more frequent than No Forest points
        self.fdominance = False
        if self.counts[0] < self.counts[1]:
            self.fdominance = True

        x_0, y_0 = np.where(bigdata_gt == 0)
        x_1, y_1 = np.where(bigdata_gt == 1)

        self.mat_idxs = np.array(list(zip(x_0, y_0)), dtype=np.uint16)
        self.mat_opposite_idxs = np.array(list(zip(x_1, y_1)), dtype=np.uint16)

        bigdata_idx_0 = [tuple(list(tup)) for tup in self.mat_idxs]
        bigdata_idx_1 = [tuple(list(tup)) for tup in self.mat_opposite_idxs]

        pckmetadata_gt[2] = self.counts[0]
        pckmetadata_gt[3] = self.counts[1]
        pckmetadata_gt[4] = self.fdominance

        self.status = SampleSelectorThreadStatus.STATUS_STORING_DATA
        self.db.set("status", self.status.encode('utf-8'))

        dataset_path = join(self.path_to_processed, 'dataset.npz')
        np.savez_compressed(dataset_path, bigdata=bigdata, bigdata_gt=bigdata_gt, pckmetadata=pckmetadata,
                            pckmetadata_gt=pckmetadata_gt)

        idxs_path = join(self.path_to_processed, 'idxs.npz')
        np.savez_compressed(idxs_path, bigdata_idx_0=bigdata_idx_0, bigdata_idx_1=bigdata_idx_1)

        self.status = SampleSelectorThreadStatus.STATUS_DONE
        self.db.set("status", self.status.encode('utf-8'))

    def scale(self, X, x_min, x_max, x_data_min, x_data_max):
        nom = (X - x_data_min) * (x_max - x_min)
        denom = x_data_max - x_data_min
        denom = denom + (denom is 0)
        return x_min + nom / denom

    '''def scale(self, X, x_min, x_max, x_data_min, x_data_max):
        nom = (X - X.min()) * (x_max - x_min)
        denom = X.max() - X.min()
        denom = denom + (denom is 0)
        return x_min + nom / denom'''
