import numpy as np
from config import SamplesConfig
from keras.utils import Sequence, to_categorical


class IndexBasedGenerator(Sequence):

    def __init__(self, batch_size=None, dataset=None, dataset_gt=None, indexes=None, offset=None, patch_size=None):
        self.dataset = dataset
        self.dataset_gt = dataset_gt
        self.indexes = indexes
        self.offset = offset
        self.patch_size = patch_size

        self.batch_size = batch_size
        self.total_samples = len(indexes)

    def __len__(self):
        return int(np.ceil(self.total_samples / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_idxs = None
        left_lim = idx * self.batch_size
        right_lim = (idx + 1) * self.batch_size

        if right_lim > self.total_samples:
            batch_idxs = self.indexes[left_lim:self.total_samples]
        else:
            batch_idxs = self.indexes[left_lim:right_lim]

        input_channels = self.dataset.shape[1]

        current_patch_size = self.patch_size if self.patch_size is not None else SamplesConfig.PATCH_SIZE

        batch_x = []
        batch_y = []

        start_offset = 0
        end_offset = current_patch_size
        if self.offset is not None:
            start_offset += self.offset
            end_offset += self.offset

        for i, idx in enumerate(batch_idxs):
            batch_x.append(self.dataset[idx[0], :, idx[1] + start_offset: idx[1] + end_offset,
                           idx[2] + start_offset: idx[2] + end_offset])
            batch_y.append(self.dataset_gt[idx[0], idx[1], idx[2]])

        batch_x = np.array(batch_x).reshape(len(batch_x), current_patch_size, current_patch_size, input_channels)
        batch_x = batch_x.astype('float32')
        batch_y = np.array(batch_y)
        batch_y = to_categorical(batch_y, 2)
        return batch_x, batch_y
