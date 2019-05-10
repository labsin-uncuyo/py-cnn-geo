import multiprocessing
import numpy as np
import time
from os import listdir, makedirs
from os.path import isfile, join, exists
from natsort import natsorted
from config import DatasetConfig, RasterParams, NormParameters
from raster_rw import GTiffHandler
from dataset.threads.sample_selector_thread_status import SampleSelectorThreadStatus


class SampleSelectorThread(multiprocessing.Process):

    # def __init__(self, i, raster_array, raster_filepath, target=None, name=None, daemon=None):
    def __init__(self, i, path_to_pck, db, process_class_total, processed_path):
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
        self.process_class_total = process_class_total

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

        srtm_np_array[srtm_np_array < 0] = 0
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
        pckmetadata_gt[1] = join(self.path_to_processed, 'DATA_SELECTION_MASK.tif')

        if not exists(self.path_to_processed):
            makedirs(self.path_to_processed)

        fnf_handler = GTiffHandler()
        fnf_handler.readFile(pckmetadata_gt[0])

        fnf_np_array = np.array(fnf_handler.src_Z)

        bigdata_gt = fnf_np_array

        fnf_zero_mask = np.zeros(shape=fnf_np_array.shape)

        fnf_handler.src_Z = fnf_zero_mask
        fnf_handler.writeNewFile(pckmetadata_gt[1])

        fnf_handler.closeFile()

        self.status = SampleSelectorThreadStatus.STATUS_TAKING_SAMPLE_COORDS
        self.db.set("status", self.status.encode('utf-8'))

        unique, self.counts = np.unique(bigdata_gt, return_counts=True)

        # Checking if Forest points are more frequent than No Forest points
        self.fdominance = False
        if self.counts[0] < self.counts[1]:
            self.fdominance = True

        x = y = None
        if self.fdominance:
            x, y = np.where(bigdata_gt == 0)
        else:
            x, y = np.where(bigdata_gt == 1)

        self.mat_idxs = np.array(list(zip(x, y)), dtype=np.uint16)

        points_left = np.zeros(shape=bigdata_gt.shape, dtype=np.uint8)

        power = 4096
        mat_idxs_dic = {}
        for el in self.mat_idxs:
            points_left[el[0]][el[1]] = 1
            key = (el[0] * power) + el[1]
            mat_idxs_dic[key] = True

        mat_op_idxs_dic = {}

        if self.process_class_total != -1:
            self.mat_opposite_idxs = np.zeros(shape=(self.process_class_total,2), dtype=np.uint16)
            self.total = self.process_class_total
            self.db.set('total', int(self.total))
        else:
            self.mat_opposite_idxs = np.zeros(shape=self.mat_idxs.shape, dtype=np.uint16)
            self.total = self.mat_idxs.shape[0]
            self.db.set('total', self.total)

        for elem_idx, elem in enumerate(self.mat_idxs):
            self.processed = elem_idx + 1
            if elem_idx % 100 == 0:
                self.db.set('processed', self.processed)
                #time.sleep(0.3)

            if self.process_class_total != -1 and elem_idx == self.process_class_total:
                break

            x_min, x_max, y_min, y_max = self.determine_XY_offset_limits(elem[0], elem[1])

            opposite_retrieved = False
            for intent in range(DatasetConfig.DATASET_POSITION_OFFSET_TRIES):
                xy_off = tuple(self.get_XY_offset(x_min, x_max, y_min, y_max))
                #if not self.isInArray(self.mat_opposite_idxs, xy_off) and not self.isInArray(self.mat_idxs, xy_off):
                if mat_op_idxs_dic.get((xy_off[0] * power) + xy_off[1]) is None and mat_idxs_dic.get((xy_off[0] * power) + xy_off[1]) is None:
                    mat_op_idxs_dic[(xy_off[0] * power) + xy_off[1]] = True
                    points_left[el[0]][el[1]] = 1
                    self.mat_opposite_idxs[elem_idx] = xy_off
                    opposite_retrieved = True
                    break

            if not opposite_retrieved:
                while not opposite_retrieved:
                    xy_off = tuple(self.get_random_offset())
                    #if not self.isInArray(self.mat_opposite_idxs, xy_off) and not self.isInArray(self.mat_idxs, xy_off):
                    if mat_op_idxs_dic.get((xy_off[0] * power) + xy_off[1]) is None and mat_idxs_dic.get(
                            (xy_off[0] * power) + xy_off[1]) is None:
                        mat_op_idxs_dic[(xy_off[0] * power) + xy_off[1]] = True
                        points_left[el[0]][el[1]] = 1
                        self.mat_opposite_idxs[elem_idx] = xy_off
                        opposite_retrieved = True

        left_x, left_y = np.where(points_left == 0)
        points_left = np.array(list(zip(left_x, left_y)), dtype=np.uint16)

        np.random.shuffle(points_left)
        points_left = points_left[:self.process_class_total - self.processed]

        for elem_idx, elem  in enumerate(points_left):
            self.processed = self.processed + 1
            if self.processed % 100 == 0:
                self.db.set('processed', self.processed)

            self.mat_opposite_idxs[self.processed-1] = elem

        '''if self.process_class_total != -1 and self.process_class_total > len(self.mat_idxs):

            for i in range(self.processed, self.process_class_total):

                self.processed = i + 1
                if i % 100 == 0:
                    self.db.set('processed', self.processed)
                    #time.sleep(0.3)

                opposite_retrieved = False
                while not opposite_retrieved:
                    xy_off = tuple(self.get_random_offset())
                    #if not self.isInArray(self.mat_opposite_idxs, xy_off) and not self.isInArray(self.mat_idxs, xy_off):
                    if mat_op_idxs_dic.get((xy_off[0] * power) + xy_off[1]) is None and mat_idxs_dic.get(
                            (xy_off[0] * power) + xy_off[1]) is None:
                        mat_op_idxs_dic[(xy_off[0] * power) + xy_off[1]] = True
                        self.mat_opposite_idxs[i] = xy_off
                        opposite_retrieved = True'''

        self.processed = self.total
        self.db.set('processed', int(self.total))

        data_handler = GTiffHandler()
        data_handler.readFile(pckmetadata_gt[1])

        data_np = np.array(data_handler.src_Z)
        if self.fdominance:
            for elem in self.mat_idxs:
                data_np[elem[0]][elem[1]] = 1
            for elem in self.mat_opposite_idxs:
                data_np[elem[0]][elem[1]] = 2
        else:
            for elem in self.mat_idxs:
                data_np[elem[0]][elem[1]] = 2
            for elem in self.mat_opposite_idxs:
                data_np[elem[0]][elem[1]] = 1

        data_handler.src_Z = data_np
        data_handler.writeSrcFile()
        data_handler.closeFile()

        bigdata_idx_0 = None
        bigdata_idx_1 = None
        if not self.fdominance:
            # bigdata_idx_1 = [tuple([self.i] + list(tup)) for tup in self.mat_idxs]
            bigdata_idx_1 = [tuple(list(tup)) for tup in self.mat_idxs]
            bigdata_idx_0 = [tuple(list(tup)) for tup in self.mat_opposite_idxs]
        else:
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

    def determine_XY_offset_limits(self, x, y):
        xmin = None
        xmax = None
        if x < DatasetConfig.DATASET_POSITION_MAX_OFFSET_X:
            xmin = 0
            xmax = x + DatasetConfig.DATASET_POSITION_MAX_OFFSET_X
        elif x + DatasetConfig.DATASET_POSITION_MAX_OFFSET_X > RasterParams.FNF_MAX_X:
            xmin = x - DatasetConfig.DATASET_POSITION_MAX_OFFSET_X
            xmax = RasterParams.FNF_MAX_X
        else:
            xmin = x - DatasetConfig.DATASET_POSITION_MAX_OFFSET_X
            xmax = x + DatasetConfig.DATASET_POSITION_MAX_OFFSET_X

        ymin = None
        ymax = None
        if y < DatasetConfig.DATASET_POSITION_MAX_OFFSET_Y:
            ymin = 0
            ymax = y + DatasetConfig.DATASET_POSITION_MAX_OFFSET_Y
        elif y + DatasetConfig.DATASET_POSITION_MAX_OFFSET_Y > RasterParams.FNF_MAX_Y:
            ymin = y - DatasetConfig.DATASET_POSITION_MAX_OFFSET_Y
            ymax = RasterParams.FNF_MAX_Y
        else:
            ymin = y - DatasetConfig.DATASET_POSITION_MAX_OFFSET_Y
            ymax = y + DatasetConfig.DATASET_POSITION_MAX_OFFSET_Y

        return xmin, xmax, ymin, ymax

    def get_XY_offset(self, xmin, xmax, ymin, ymax):

        x_offset = np.random.randint(xmin, xmax)
        y_offset = np.random.randint(ymin, ymax)

        return x_offset, y_offset

    def get_random_offset(self):
        x_offset = np.random.randint(0, RasterParams.FNF_MAX_X)
        y_offset = np.random.randint(0, RasterParams.FNF_MAX_Y)

        return x_offset, y_offset

    def scale(self, X, x_min, x_max, x_data_min, x_data_max):
        nom = (X - x_data_min) * (x_max - x_min)
        denom = x_data_max - x_data_min
        denom = denom + (denom is 0)
        return x_min + nom / denom

    def isInArray(self, array, elem):
        return True if len(np.where(array[np.where(array[:, 0] == elem[0])[0], 1] == elem[1])[0]) > 0 else False
