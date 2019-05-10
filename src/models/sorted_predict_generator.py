import numpy as np
from config import SamplesConfig, RasterParams
from keras.utils import Sequence


class SortedPredictGenerator(Sequence):

    def __init__(self, batch_size, dataset):
        self.dataset = dataset

        self.batch_size = batch_size
        self.total_samples = RasterParams.FNF_MAX_X*RasterParams.FNF_MAX_Y

    def __len__(self):
        return int(np.ceil(self.total_samples / float(self.batch_size)))

    def __getitem__(self, idx):

        patches_batch = []
        left_lim = idx * self.batch_size
        right_lim = (idx + 1) * self.batch_size if (idx + 1) * self.batch_size <= self.total_samples else self.total_samples


        y_start = int(left_lim / RasterParams.FNF_MAX_Y)
        y_end = int(right_lim / RasterParams.FNF_MAX_Y)

        for y in range(y_start, y_end+1):
            if y_start == y_end:
                x_start = left_lim % RasterParams.FNF_MAX_Y
                x_end = right_lim % RasterParams.FNF_MAX_Y
            elif y == y_start:
                x_start = left_lim % RasterParams.FNF_MAX_Y
                x_end = RasterParams.FNF_MAX_Y
            elif y == y_end:
                x_start = 0
                x_end = right_lim % RasterParams.FNF_MAX_Y

            for x in range(x_start, x_end):

                patch = []
                for i in range(0, self.dataset.shape[0]):
                    patch.append(self.dataset[i, y:y + SamplesConfig.PATCH_SIZE, x:x + SamplesConfig.PATCH_SIZE])

                patches_batch.append(patch)

        patches_batch = np.array(patches_batch).reshape(len(patches_batch), SamplesConfig.PATCH_SIZE, SamplesConfig.PATCH_SIZE, self.dataset.shape[0])
        patches_batch = patches_batch.astype('float32')
        return patches_batch