import numpy as np
from config import SamplesConfig
from keras.utils import Sequence, to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras as K
import skimage.transform as tf
import cv2


class IndexBasedGenerator(Sequence):

    def __init__(self, batch_size=None, dataset=None, dataset_gt=None, indexes=None, offset=None, patch_size=None,
                 augment=False, aug_granularity=1, aug_patch_size=None, rotation_range=45, shear_range=25):
        self.dataset = dataset
        self.dataset_gt = dataset_gt
        self.indexes = indexes
        self.offset = offset
        self.patch_size = patch_size
        self.augment = augment
        self.aug_granularity = aug_granularity
        self.aug_patch_size = aug_patch_size
        self.rotation_range = rotation_range
        self.rotation_range_rad = np.deg2rad(self.rotation_range)
        self.shear_range = shear_range
        self.shear_range_rad = np.deg2rad(self.shear_range)

        center_shift = int(self.aug_patch_size / 2)
        self.tf_center = tf.SimilarityTransform(translation=-center_shift)
        self.tf_uncenter = tf.SimilarityTransform(translation=center_shift)

        self.start_size = self.aug_patch_size // 2 - (self.patch_size // 2)

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

        final_patch_size = self.patch_size if self.patch_size is not None else SamplesConfig.PATCH_SIZE
        if not self.augment:
            current_patch_size = final_patch_size
        else:
            current_patch_size = self.aug_patch_size

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

        if self.augment:
            final_batch_x = []

            if self.aug_granularity > 1:
                i = 0
                temp_x = []
                for x in batch_x:
                    temp_x.append(x)
                    i+=1
                    if i == self.aug_granularity:
                        temp_x = np.dstack((temp_x[:]))
                        temp_x = self.cv_random_flip(temp_x)
                        temp_x = self.cv_random_rotation_and_shear(temp_x)
                        temp_x = np.split(temp_x, self.aug_granularity, 2)
                        final_batch_x.extend(np.array(temp_x)[:,self.start_size:self.start_size+final_patch_size,self.start_size:self.start_size+final_patch_size,:])
                        temp_x = []
                        i = 0
                if i != 0:
                    temp_x = np.dstack((temp_x[:]))
                    temp_x = self.cv_random_flip(temp_x)
                    temp_x = self.cv_random_rotation_and_shear(temp_x)
                    temp_x = np.split(temp_x, i, 2)
                    final_batch_x.extend(np.array(temp_x)[:,self.start_size:self.start_size+final_patch_size,self.start_size:self.start_size+final_patch_size,:])
            else:
                for x in batch_x:
                    x = self.cv_random_flip(x)
                    x = self.cv_random_rotation_and_shear(x)
                    final_batch_x.append(x[self.start_size:self.start_size+final_patch_size, self.start_size:self.start_size+final_patch_size, :])
            final_batch_x = np.array(final_batch_x)
            final_batch_x = final_batch_x.astype('float32')
        else:
            final_batch_x = batch_x

        batch_y = np.array(batch_y)
        batch_y = to_categorical(batch_y, 2)

        return final_batch_x, batch_y

    def random_fliplr(self, x):
        if np.random.random() < 0.5:
            x = np.fliplr(x)
        return x

    def random_flipud(self, x):
        if np.random.random() < 0.5:
            x = np.flipud(x)
        return x

    def random_rotation_and_shear(self, x):
        tf_total = self.random_affine()
        warped_x = tf.warp(x, tf_total, order=1, preserve_range=True, mode='constant', cval=1)

        return warped_x

    def fixed_rotation_and_shear(self, x, tf_matrix):
        warped_x = tf.warp(x, tf_matrix, order=1, preserve_range=True, mode='constant', cval=1)

        return warped_x

    def random_affine(self):
        affine = tf.AffineTransform(rotation=self.get_random_rot(self.rotation_range_rad), shear=self.get_random_rot(self.shear_range_rad))
        tf_total = self.tf_center + affine + self.tf_uncenter
        return tf_total

    def get_random_rot(self, rad):
        return np.random.uniform(-rad, rad)

    def cv_random_flip(self, x):
        f = np.random.uniform()
        if f < 0.25:
            return cv2.flip(x, flipCode=0)
        elif f < 0.5:
            return cv2.flip(x, flipCode=1)
        elif f < 0.75:
            return cv2.flip(x, flipCode=-1)
        return x

    def cv_random_rotation_and_shear(self, x):
        T_opencv = np.float32(self.random_affine().params.flatten()[:6].reshape(2, 3))
        return cv2.warpAffine(x, M=T_opencv, dsize=(self.aug_patch_size, self.aug_patch_size))
