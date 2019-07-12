import gdal
import getopt
import sys
sys.path.append("..")
import os
import gc
import datetime
import numpy as np
import tensorflow as tf
from operator import itemgetter
from os import listdir
from os.path import isfile, join
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from search.keras_batch_classifier import KerasBatchClassifier
from config import DatasetConfig, RasterParams, NetworkParameters
from entities.AccuracyHistory import AccuracyHistory
from models.index_based_generator import IndexBasedGenerator


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
    cls = 4
    fms = 64
    clks = 3
    fcls = 3
    fcns = 1000

    name_args = ['cls', 'fms', 'clks', 'fcls', 'fcns']
    param_grid = dict(cls=cls, fms=fms, clks=clks, fcls=fcls, fcns=fcns)

    model = create_model(cls=cls, fms=fms, clks=clks, fcls=fcls, fcns=fcns)

    patch_size = get_padding(model.layers)
    offset = int(DatasetConfig.MAX_PADDING / 2) - int(patch_size / 2)

    traingen = IndexBasedGenerator(batch_size=NetworkParameters.BATCH_SIZE, dataset=dataset,
                                   dataset_gt=dataset_gt, indexes=dataset_idxs, patch_size=patch_size,
                                   offset=offset)
    '''valgen = IndexBasedGenerator(batch_size=NetworkParameters.BATCH_SIZE, dataset=self.dataset,
                                 dataset_gt=self.dataset_gt, indexes=val_idxs, patch_size=patch_size, offset=offset)'''

    name = '{date:%Y%m%d_%H%M%S}'.format(date=datetime.datetime.now()) + '.' + build_name(name_args,
                                                                                                    param_grid)

    model_json = model.to_json()
    with open("../storage/search/" + name + ".json", "w") as json_file:
        json_file.write(model_json)

    accuracy_history = AccuracyHistory()
    early_stopping = EarlyStopping(patience=5, verbose=5, mode="auto", monitor='acc')
    model_checkpoint = ModelCheckpoint(
        "../storage/search/" + name + ".weights.{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5", monitor='acc',
        verbose=5, save_best_only=False, mode="auto")

    callbacks = [accuracy_history, early_stopping, model_checkpoint]

    model.fit_generator(
        traingen,
        steps_per_epoch=int(dataset_idxs.shape[0] // NetworkParameters.BATCH_SIZE) + 1,
        callbacks=callbacks,
        epochs=10,
        verbose=1
    )


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


def get_padding(layers, ws=None):
    ws = ws if ws is not None else 1

    for layer_idx in range(len(layers) - 1, -1, -1):
        layer = layers[layer_idx]
        if type(layer) == Conv2D:
            padding = 0 if layer.padding == 'valid' else (
                int(layer.kernel_size[0] / 2) if layer.padding == 'same' else 0)
            ws = ((ws - 1) * layer.strides[0]) - (2 * padding) + layer.kernel_size[0]
        if isinstance(layer, Model):
            ws = get_padding(layer.layers, ws)

    return ws

def build_name(naming_args, sk_params):

    arr = []
    for naming_arg in naming_args:
        arr.append((naming_arg, sk_params[naming_arg]))

    name = ''
    for i, item in enumerate(arr):
        if i != 0:
            name += '-'

        name += item[0] + '_' + str(item[1])

    return name

main(sys.argv[1:])
