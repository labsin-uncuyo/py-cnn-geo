import gdal
import getopt
import sys
sys.path.append("..")
import os
import gc
import numpy as np
from operator import itemgetter
from os import listdir
from os.path import isfile, join
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, RMSprop, Adamax
from sklearn.model_selection import GridSearchCV
from search.keras_batch_classifier import KerasBatchClassifier
from config import DatasetConfig, RasterParams
import tensorflow as tf
from keras import backend as K


num_cores = 8

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for training on gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Preparing for model parameter search')
    gdal.UseExceptions()
    dataset_folder = None

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config=config)

    try:
        opts, args = getopt.getopt(argv, "hs:", ["dataset_folder="])
    except getopt.GetoptError:
        print('model_search.py -s <dataset_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('model_search.py -s <dataset_folder>')
            sys.exit()
        elif opt in ["-s", "--dataset_folder"]:
            dataset_folder = arg

    print('Working with dataset folder %s' % dataset_folder)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    parameter_searcher(dataset_folder)

    sys.exit()


def parameter_searcher(dataset_folder):
    dataset, dataset_gt, dataset_idxs = prepare_generator_dataset(dataset_folder, DatasetConfig.MAX_PADDING)
    #dataset = dataset_gt =  dataset_idxs = None

    #train_idxs = validation_idxs = None

    # create model
    model = KerasBatchClassifier(build_fn=create_model, verbose=1)
    # define the grid search parameters
    #cls = [4, 5, 6]
    cls = [6]
    fms = [64, 128]
    #fms = [128]
    clks = [5, 3]
    #clks = [3]
    fcls = [2, 3]
    #fcls = [1]
    fcns = [1000, 1500, 2000]
    #fcns = [500]
    #lr = [0.01, 0.001, 0.0001]
    #opt = [Adam(), RMSprop(), Adamax()]

    name_args = ['cls', 'fms', 'clks', 'fcls', 'fcns']
    param_grid = dict(cls=cls, fms=fms, clks=clks, fcls=fcls, fcns=fcns)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, error_score='raise', verbose=1, cv=3)

    grid_result = grid.fit(dataset_idxs, dataset_idxs, dataset=dataset, dataset_gt=dataset_gt, name_args=name_args)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print(model.summary())


def create_model(cls=4, fms=64, clks=5, fcls=1, fcns=1000, lr=0.001):
    print('---> Create model parameters: ', cls, fms, clks, fcls, fcns)
    K.clear_session()
    # this may get harder to calculate if the
    patch_size = (clks-1) + ((cls-1) * 2) + 1

    input_shape = (patch_size, patch_size, DatasetConfig.DATASET_LST_BANDS_USED)

    model = Sequential()
    for i in range(cls):
        if i == 0:
            model.add(Conv2D(fms, kernel_size=(clks, clks), strides=(1, 1), activation='relu', input_shape=input_shape))
        else:
            model.add(Conv2D(fms, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    for i in range(fcls):
        model.add(Dense(fcns, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=binary_crossentropy, optimizer=Adam(lr=lr), metrics=['accuracy'])
    return model


def prepare_generator_dataset(dataset_folder, padding):
    rasters_folders = [f for f in listdir(dataset_folder) if not isfile(join(dataset_folder, f))]

    rasters_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print(rasters_folders)

    pad_factor = int(padding / 2) * 2
    bigdata = np.zeros(shape=(
        len(rasters_folders), DatasetConfig.DATASET_LST_BANDS_USED, RasterParams.SRTM_MAX_X + pad_factor,
        RasterParams.SRTM_MAX_Y + pad_factor), dtype=np.float32)
    bigdata_gt = np.zeros(shape=(len(rasters_folders), RasterParams.FNF_MAX_X, RasterParams.FNF_MAX_Y),
                          dtype=np.uint8)

    for i, pck in enumerate(rasters_folders):
        path_to_pck = join(dataset_folder, pck, 'dataset.npz')

        print('Loading dataset folder ', pck)

        pck_bigdata = None
        item_getter = itemgetter('bigdata')
        with np.load(path_to_pck) as df:
            pck_bigdata = item_getter(df)

        half_padding = int(padding / 2)
        pck_bigdata = np.pad(pck_bigdata, [(0, 0), (half_padding, half_padding), (half_padding, half_padding)],
                             mode='constant')

        bigdata[i] = pck_bigdata

        pck_bigdata_gt = None
        item_getter = itemgetter('bigdata_gt')
        with np.load(path_to_pck) as df:
            pck_bigdata_gt = item_getter(df)

        bigdata_gt[i] = pck_bigdata_gt

        del pck_bigdata
        del pck_bigdata_gt

        gc.collect()

    bigdata_idx_0 = None
    bigdata_idx_1 = None

    path_to_idxs = join(dataset_folder, 'samples_shuffled_factor_idx.npz')
    item_getter = itemgetter('bigdata_idx_0', 'bigdata_idx_1')
    with np.load(path_to_idxs) as df:
        bigdata_idx_0, bigdata_idx_1 = item_getter(df)

    bigdata_idx_mix = np.empty((bigdata_idx_0.shape[0] + bigdata_idx_1.shape[0], 3), dtype=bigdata_idx_0.dtype)
    bigdata_idx_mix[0::2] = bigdata_idx_0
    del bigdata_idx_0
    gc.collect()
    bigdata_idx_mix[1::2] = bigdata_idx_1
    del bigdata_idx_1
    gc.collect()

    return bigdata, bigdata_gt, bigdata_idx_mix



main(sys.argv[1:])
