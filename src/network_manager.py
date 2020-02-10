import sys, getopt
import keras
import gc
import multiprocessing
import os
import time
import matplotlib.pylab as plt
import numpy as np

from functools import partial

from dask.array import store
from natsort import natsorted
from operator import itemgetter
from os import listdir
from os.path import isfile, join, exists
from keras.models import Sequential, model_from_json, Model
from keras.layers import Conv2D, Flatten, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, RMSprop, Adamax, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils.multiclass import unique_labels
from raster_rw import GTiffHandler
from entities.AccuracyHistory import AccuracyHistory
from config import SamplesConfig, NetworkParameters, RasterParams, DatasetConfig, TestParameters
from models.index_based_generator import IndexBasedGenerator
from models.sorted_predict_generator import SortedPredictGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # for training on gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

OPERATION_CREATE_SAMPLES = 10
OPERATION_DIVIDE_SAMPLES = 20
OPERATION_CREATE_NETWORK = 30
OPERATION_CREATE_NETWORK_WITH_GENERATOR = 31
OPERATION_CREATE_NETWORK_WITH_GENERATOR_AND_KFOLD = 32
OPERATION_TRAIN_NETWORK = 40
OPERATION_TEST_NETWORK = 50
OPERATION_TEST_NETWORK_WITH_GENERATOR_AND_KFOLD = 52
OPERATION_TEST_FULLSIZE_NETWORK = 60
OPERATION_TEST = 70
OPERATION_PLAY = 99


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Entering network manager')
    dataset_folder = ''
    samples_file = ''
    model_file = ''
    weights_file = ''
    network_name = ''
    dataset_file = ''
    tif_sample = ''
    tif_real = ''
    result_name = ''
    models_folder = ''
    augment = False
    aug_granularity = 1

    operation = None
    include_validation = False

    try:
        opts, args = getopt.getopt(argv, "hs:d:c:k:z:e:r:f:n:o:g:pa:b:m:",
                                   ["create_sample=", "divide_sample=", "create_network=", "create_network_gen=",
                                    "test=", "tif_sample=", "tif_real", "test_fullsize=", "network_name=",
                                    "result_name=", "augment=", "play", "create_network_gen_kfold=",
                                    "test_network_gen_kfold=", "models_folder="])
    except getopt.GetoptError:
        print('network_manager.py -h')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('network_manager.py -s <dataset_folder>\n'
                  'network_manager.py -d <sample_file>\n'
                  'network_manager.py -c <sample_file>\n'
                  'network_manager.py -tr <sample_file> -n <network_name>\n'
                  'network_manager.py -tef <model_file> <weights_file> <dataset_folder>')
            sys.exit()
        elif opt in ["-s", "--create_sample"]:
            dataset_folder = arg
            operation = OPERATION_CREATE_SAMPLES
        elif opt in ["-d", "--divide_sample"]:
            samples_file = arg
            operation = OPERATION_DIVIDE_SAMPLES
        elif opt in ["-c", "--create_network"]:
            samples_file = arg
            operation = OPERATION_CREATE_NETWORK
        elif opt in ["-k", "--create_network_gen"]:
            dataset_folder = arg
            operation = OPERATION_CREATE_NETWORK_WITH_GENERATOR
        elif opt in ["-z", "--test"]:
            dataset_file = arg
            operation = OPERATION_TEST
        elif opt in ["-a", "--create_network_gen_kfold"]:
            dataset_folder = arg
            operation = OPERATION_CREATE_NETWORK_WITH_GENERATOR_AND_KFOLD
        elif opt in ["-b", "--test_network_gen_kfold"]:
            dataset_folder = arg
            operation = OPERATION_TEST_NETWORK_WITH_GENERATOR_AND_KFOLD
        elif opt in ["-f", "--test_fullsize"]:
            opt_args = arg.split('&')
            model_file = opt_args[0]
            weights_file = opt_args[1]
            dataset_folder = opt_args[2]
            operation = OPERATION_TEST_FULLSIZE_NETWORK
        elif opt in ["-p", "--play"]:
            samples_file = "data/processed/samples.npz"
            operation = OPERATION_PLAY
        elif opt in ["-n", "--network_name"]:
            network_name = arg
        elif opt in ["-e", "--tif_sample"]:
            tif_sample = arg
        elif opt in ["-r", "--tif_real"]:
            tif_real = arg
        elif opt in ["-o", "--result_name"]:
            result_name = arg
        elif opt in ["-g", "--augment"]:
            augment = True
            aug_granularity = int(arg)
        elif opt in ["-m", "--models_folder"]:
            models_folder = arg

    print('Working with dataset file %s' % dataset_folder)

    if operation == OPERATION_CREATE_SAMPLES:
        prepare_dataset(dataset_folder)
    elif operation == OPERATION_DIVIDE_SAMPLES:
        divide_samples(samples_file)
    elif operation == OPERATION_CREATE_NETWORK:
        train_x = train_y = val_x = val_y = test_x = test_y = None
        if include_validation:
            train_x, train_y, val_x, val_y, test_x, test_y = divide_samples(samples_file)
            model = create_model()

            train_x = train_x.reshape(train_x.shape[0], SamplesConfig.PATCH_SIZE, SamplesConfig.PATCH_SIZE,
                                      train_x.shape[1])
            val_x = val_x.reshape(val_x.shape[0], SamplesConfig.PATCH_SIZE, SamplesConfig.PATCH_SIZE, val_x.shape[1])
            test_x = test_x.reshape(test_x.shape[0], SamplesConfig.PATCH_SIZE, SamplesConfig.PATCH_SIZE,
                                    test_x.shape[1])

            train_x = train_x.astype('float32')
            val_x = val_x.astype('float32')
            test_x = test_x.astype('float32')

            train_y = keras.utils.to_categorical(train_y, 2)
            val_y = keras.utils.to_categorical(val_y, 2)
            test_y = keras.utils.to_categorical(test_y, 2)

            model = train_network(model, train_x, train_y, NetworkParameters.BATCH_SIZE, NetworkParameters.EPOCHS,
                                  val_x, val_y, network_name)
        else:
            train_x, train_y, test_x, test_y = divide_samples(samples_file)
            model = create_model()

            # serialize model to JSON
            model_json = model.to_json()
            with open("storage/" + network_name + ".json", "w") as json_file:
                json_file.write(model_json)

            train_x = train_x.reshape(train_x.shape[0], SamplesConfig.PATCH_SIZE, SamplesConfig.PATCH_SIZE,
                                      train_x.shape[1])
            test_x = test_x.reshape(test_x.shape[0], SamplesConfig.PATCH_SIZE, SamplesConfig.PATCH_SIZE,
                                    test_x.shape[1])

            train_x = train_x.astype('float32')
            test_x = test_x.astype('float32')

            train_y = keras.utils.to_categorical(train_y, 2)
            test_y = keras.utils.to_categorical(test_y, 2)

            model, acc = train_network(model, train_x, train_y, NetworkParameters.BATCH_SIZE, NetworkParameters.EPOCHS,
                                       test_x, test_y, network_name)

        # serialize weights to HDF5
        model_weights_name = network_name + '_' + str(acc).format(".2f")
        model.save_weights("storage/" + model_weights_name + ".h5")
        print("Saved model to disk")
    elif operation == OPERATION_CREATE_NETWORK_WITH_GENERATOR:
        dataset, dataset_gt, dataset_idxs = prepare_generator_dataset(dataset_folder, DatasetConfig.MAX_PADDING)

        model = create_model()

        # serialize model to JSON
        model_json = model.to_json()
        with open("storage/" + network_name + ".json", "w") as json_file:
            json_file.write(model_json)

        accuracy_history = AccuracyHistory()

        filepath = network_name + "-weights-improvement-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5"

        checkpoint = ModelCheckpoint(join("storage/temp/", filepath), monitor='val_acc', verbose=1,
                                     save_best_only=False, mode='max')

        train_idxs, validation_idxs = divide_indexes(dataset_idxs)

        train_generator = IndexBasedGenerator(NetworkParameters.BATCH_SIZE, dataset, dataset_gt, train_idxs)

        val_x, val_y = prepare_validation_from_idxs(dataset, dataset_gt, validation_idxs)

        # val_x = val_x.astype('float32')
        val_y = keras.utils.to_categorical(val_y, 2)

        model.fit_generator(train_generator, steps_per_epoch=len(train_idxs) // NetworkParameters.BATCH_SIZE,
                            epochs=NetworkParameters.EPOCHS, verbose=1, callbacks=[accuracy_history, checkpoint],
                            validation_data=(val_x, val_y))

        plt.plot(range(1, NetworkParameters.EPOCHS + 1), accuracy_history.acc)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

        # serialize weights to HDF5
        model_weights_name = network_name + '_' + "{0:.4f}".format(accuracy_history.acc[-1])
        model.save_weights("storage/" + model_weights_name + ".h5")
        print("Saved model to disk")
    elif operation == OPERATION_CREATE_NETWORK_WITH_GENERATOR_AND_KFOLD:
        train_network_kfold(dataset_folder, network_name, 5, 10, augment, aug_granularity)
    elif operation == OPERATION_TEST_NETWORK_WITH_GENERATOR_AND_KFOLD:
        test_network_kfold(dataset_folder, network_name, models_folder, 5, augment, aug_granularity)
    elif operation == OPERATION_TEST_FULLSIZE_NETWORK:
        test_load_fullsize(model_file, weights_file, dataset_folder)
    elif operation == OPERATION_TEST:
        test(network_name, dataset_file, tif_sample, tif_real, result_name)
    elif operation == OPERATION_PLAY:
        play4()

    sys.exit()


def prepare_validation_from_idxs(dataset, dataset_gt, validation_idxs):
    val_x = np.zeros(
        shape=(validation_idxs.shape[0], SamplesConfig.PATCH_SIZE, SamplesConfig.PATCH_SIZE, dataset.shape[1]),
        dtype=dataset.dtype)
    val_y = np.zeros(shape=(validation_idxs.shape[0],), dtype=dataset_gt.dtype)
    for i, idx in enumerate(validation_idxs):
        sample_x = np.array(dataset[idx[0], :, idx[1]: idx[1] + SamplesConfig.PATCH_SIZE,
                            idx[2]: idx[2] + SamplesConfig.PATCH_SIZE])
        sample_y = np.array(dataset_gt[idx[0], idx[1], idx[2]])
        sample_x = sample_x.reshape(SamplesConfig.PATCH_SIZE, SamplesConfig.PATCH_SIZE, dataset.shape[1])

        val_x[i] = sample_x
        val_y[i] = sample_y

    return val_x, val_y


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


def prepare_dataset(dataset_folder):
    sample_rasters_folders = [f for f in listdir(dataset_folder) if not isfile(join(dataset_folder, f))]

    sample_rasters_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print(sample_rasters_folders)

    bigdata = np.zeros(shape=(
        len(sample_rasters_folders), DatasetConfig.DATASET_LST_BANDS_USED, RasterParams.SRTM_MAX_X,
        RasterParams.SRTM_MAX_Y), dtype=np.float32)
    bigdata_gt = np.zeros(shape=(len(sample_rasters_folders), RasterParams.FNF_MAX_X, RasterParams.FNF_MAX_Y),
                          dtype=np.uint8)
    bigdata_samples = []
    bigdata_gt_samples = []
    bigdata_idx = []
    # np.empty(shape=(bigdata_idx_0.shape[0] * 2, bigdata.shape[1], SamplesConfig.PATCH_SIZE, SamplesConfig.PATCH_SIZE), dtype=bigdata.dtype)

    # samples_cnt = 0
    for i, pck in enumerate(sample_rasters_folders):
        path_to_pck = join(dataset_folder, pck, 'dataset.npz')

        pck_bigdata = None
        item_getter = itemgetter('bigdata')
        with np.load(path_to_pck) as df:
            pck_bigdata = item_getter(df)

        bigdata[i] = pck_bigdata

        pad_amount = int(SamplesConfig.PATCH_SIZE / 2)
        pck_bigdata = np.pad(pck_bigdata, [(0, 0), (pad_amount, pad_amount), (pad_amount, pad_amount)], mode='constant')

        pck_bigdata_gt = pck_bigdata_idx_0 = pck_bigdata_idx_1 = None
        item_getter = itemgetter('bigdata_gt', 'bigdata_idx_0', 'bigdata_idx_1')
        with np.load(path_to_pck) as df:
            pck_bigdata_gt, pck_bigdata_idx_0, pck_bigdata_idx_1 = item_getter(df)

        bigdata_gt[i] = pck_bigdata_gt

        for elem_idx in range(len(pck_bigdata_idx_0)):
            sample_0 = []
            sample_1 = []
            elem_0 = pck_bigdata_idx_0[elem_idx]
            elem_1 = pck_bigdata_idx_1[elem_idx]
            for j in range(pck_bigdata.shape[0]):
                sample_0.append(pck_bigdata[j, elem_0[0]:elem_0[0] + SamplesConfig.PATCH_SIZE,
                                elem_0[1]:elem_0[1] + SamplesConfig.PATCH_SIZE])
                sample_1.append(pck_bigdata[j, elem_1[0]:elem_1[0] + SamplesConfig.PATCH_SIZE,
                                elem_1[1]:elem_1[1] + SamplesConfig.PATCH_SIZE])
            bigdata_samples.append(sample_0)
            bigdata_samples.append(sample_1)

            bigdata_gt_samples.append(pck_bigdata_gt[elem_0[0]][elem_0[1]])
            bigdata_gt_samples.append(pck_bigdata_gt[elem_1[0]][elem_1[1]])

            bigdata_idx.append([i] + list(elem_0))
            bigdata_idx.append([i] + list(elem_1))
            # samples_cnt += 2

        del pck_bigdata
        del pck_bigdata_gt
        del pck_bigdata_idx_0
        del pck_bigdata_idx_1

        gc.collect()

    np.savez_compressed('data/processed/full.npz', bigdata=bigdata, bigdata_gt=bigdata_gt, bigdata_idx=bigdata_idx)

    del bigdata
    del bigdata_gt

    bigdata_samples = np.array(bigdata_samples)
    bigdata_gt_samples = np.array(bigdata_gt_samples)
    bigdata_idx = np.array(bigdata_idx)

    bigdata_samples_s, bigdata_gt_samples_s, bigdata_idx_s = shuffle_in_unison(bigdata_samples, bigdata_gt_samples,
                                                                               bigdata_idx)
    del bigdata_samples
    del bigdata_gt_samples
    del bigdata_idx

    np.savez_compressed('data/processed/samples.npz', bigdata_samples=bigdata_samples_s,
                        bigdata_gt_samples=bigdata_gt_samples_s, bigdata_idx=bigdata_idx_s)

    print(bigdata_samples_s.shape)
    print(bigdata_gt_samples_s.shape)
    print(bigdata_idx_s.shape)


def shuffle_in_unison(a, b, c):
    assert len(a) == len(b) == len(c)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    shuffled_c = np.empty(c.shape, dtype=c.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
        shuffled_c[new_index] = c[old_index]
    return shuffled_a, shuffled_b, shuffled_c


def divide_indexes(indexes_file, val_percentage=SamplesConfig.TEST_PERCENTAGE):
    split_1 = int(val_percentage * len(indexes_file))

    train_idxs = indexes_file[split_1:]
    validation_idxs = indexes_file[:split_1]

    return train_idxs, validation_idxs


def divide_samples(samples_file, include_validation=False):
    bigdata_samples = bigdata_gt_samples = bigdata_idx = None

    item_getter = itemgetter('bigdata_samples', 'bigdata_gt_samples', 'bigdata_idx')
    with np.load(samples_file) as sf:
        bigdata_samples, bigdata_gt_samples, bigdata_idx = item_getter(sf)

    split_1 = int(SamplesConfig.TEST_PERCENTAGE * len(bigdata_samples))

    bigtrain_x = bigdata_samples[split_1:]
    test_x = bigdata_samples[:split_1]
    bigtrain_y = bigdata_gt_samples[split_1:]
    test_y = bigdata_gt_samples[:split_1]

    if include_validation:
        split_2 = int(SamplesConfig.VALIDATION_PERCENTAGE * len(bigtrain_x))

        train_x = bigtrain_x[split_2:]
        val_x = bigtrain_x[:split_2]
        train_y = bigtrain_y[split_2:]
        val_y = bigtrain_y[:split_2]

        return train_x, train_y, val_x, val_y, test_x, test_y
    else:
        return bigtrain_x, bigtrain_y, test_x, test_y


def create_model(cls=4, fms=64, clks=3, fcls=1, fcns=1000, optimizer=Adam()):
    print('---> Create model parameters: ', cls, fms, clks, fcls, fcns)
    # K.clear_session()
    # this may get harder to calculate if the
    patch_size = (clks - 1) + ((cls - 1) * 2) + 1

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

    model.compile(loss=binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    return model


'''def create_network(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model'''


def train_network(model, x_train, y_train, batch_size, epochs, x_val, y_val, network_name):
    accuracy_history = AccuracyHistory()

    filepath = network_name + "-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

    checkpoint = ModelCheckpoint(join("storage/temp/", filepath), monitor='val_acc', verbose=1, save_best_only=False,
                                 mode='max')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_val, y_val),
              callbacks=[accuracy_history, checkpoint])

    plt.plot(range(1, NetworkParameters.EPOCHS + 1), accuracy_history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    return model, accuracy_history.acc[-1]


def train_network_kfold(dataset_folder, network_name, splits, epochs=NetworkParameters.EPOCHS, augment=False,
                        aug_granularity=1):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    if not augment:
        dataset_padding = DatasetConfig.MAX_PADDING
    else:
        dataset_padding = (get_padding_from_params(cls=4, clks=3) * 2) + 1

    dataset, dataset_gt, dataset_idxs = prepare_generator_dataset(dataset_folder, dataset_padding)

    kfold = KFold(n_splits=splits, shuffle=True, random_state=seed)

    kfold_splits = kfold.split(X=dataset_idxs)

    cmap = plt.cm.get_cmap('hsv', splits + 1)

    store_dir = 'storage/kfold/'

    for i, (train, test) in enumerate(kfold_splits):
        print("=========================================")
        print("===== K Fold Validation step => %d/5 =====" % (i + 1))
        print("=========================================")

        # model = create_model(cls=5, fms=128, clks=5, fcls=3, fcns=2000, optimizer=SGD())
        model = create_model(cls=4, fms=64, clks=3, fcls=2, fcns=1000, optimizer=SGD())

        patch_size = get_padding(model.layers)
        if not augment:
            offset = int(DatasetConfig.MAX_PADDING / 2) - int(patch_size / 2)
        else:
            offset = 0

        kfold_netname = str(i + 1) + '_' + network_name

        # serialize model to JSON
        model_json = model.to_json()
        with open(store_dir + kfold_netname + ".json", "w") as json_file:
            json_file.write(model_json)

        filepath = kfold_netname + "-weights-improvement-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5"

        accuracy_history = AccuracyHistory()
        early_stopping = EarlyStopping(patience=5, verbose=5, mode="auto", monitor='val_acc')
        checkpoint = ModelCheckpoint(join(store_dir, "temp/", filepath), monitor='val_acc', verbose=1,
                                     save_best_only=False, mode='max')

        train_idxs, validation_idxs = divide_indexes(dataset_idxs[train], val_percentage=0.15)

        train_generator = IndexBasedGenerator(batch_size=NetworkParameters.BATCH_SIZE,
                                              dataset=dataset,
                                              dataset_gt=dataset_gt,
                                              indexes=train_idxs,
                                              patch_size=patch_size,
                                              offset=offset,
                                              augment=augment,
                                              aug_granularity=aug_granularity,
                                              aug_patch_size=dataset_padding)
        val_generator = IndexBasedGenerator(batch_size=NetworkParameters.BATCH_SIZE,
                                            dataset=dataset,
                                            dataset_gt=dataset_gt,
                                            indexes=validation_idxs,
                                            patch_size=patch_size,
                                            offset=offset,
                                            augment=False,
                                            aug_granularity=None,
                                            aug_patch_size=dataset_padding)

        model.fit_generator(train_generator,
                            steps_per_epoch=len(train_idxs) // NetworkParameters.BATCH_SIZE,
                            # steps_per_epoch=NetworkParameters.BATCH_SIZE,
                            epochs=epochs,
                            verbose=1,
                            callbacks=[accuracy_history, early_stopping, checkpoint],
                            validation_data=val_generator,
                            validation_steps=len(validation_idxs) // NetworkParameters.BATCH_SIZE,
                            use_multiprocessing=True)
        # validation_steps=NetworkParameters.BATCH_SIZE)

        plt.plot(range(1, epochs + 1), accuracy_history.acc, color=cmap(i))
        plt.plot(range(1, epochs + 1), accuracy_history.loss, color=cmap(i), linestyle='dashed')

        # serialize weights to HDF5
        model_weights_name = kfold_netname + '_' + "{0:.4f}-{1:.4f}".format(accuracy_history.loss[-1],
                                                                            accuracy_history.acc[-1])
        model.save_weights(store_dir + model_weights_name + ".h5")
        print("Saved model to disk")

        evalgen = IndexBasedGenerator(batch_size=NetworkParameters.BATCH_SIZE,
                                      dataset=dataset,
                                      dataset_gt=dataset_gt,
                                      indexes=dataset_idxs[test],
                                      patch_size=patch_size,
                                      offset=offset)

        predict_out = model.predict_generator(generator=evalgen,
                                              steps=int(len(test) // NetworkParameters.BATCH_SIZE) + 1,
                                              use_multiprocessing=True,
                                              verbose=1)
        predict_out = np.argmax(predict_out, axis=1)

        expected = np.zeros(shape=(len(test),), dtype=np.uint8)
        for j, idx in enumerate(dataset_idxs[test]):
            expected[j] = dataset_gt[idx[0], idx[1], idx[2]]

        cm = plot_confusion_matrix(expected, predict_out, np.array(['No Forest', 'Forest']), plot=False)

        print(classification_report(expected, predict_out, target_names=np.array(['no forest', 'forest'])))

        confmat_file = store_dir + kfold_netname + ".conf_mat.npz"

        if not exists(confmat_file):
            np.savez_compressed(confmat_file, cm=cm)

    plt.xlabel('Epochs')
    plt.ylabel('Validation Acc/Loss')
    plt.savefig(store_dir + 'train_kfold.png', bbox_inches='tight')
    plt.savefig(store_dir + 'train_kfold.pdf', bbox_inches='tight')
    plt.clf()

    return model, accuracy_history.acc[-1]


def test_network_kfold(dataset_folder, network_name, models_folder, splits, augment=False, aug_granularity=1):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    if not augment:
        dataset_padding = DatasetConfig.MAX_PADDING
    else:
        dataset_padding = (get_padding_from_params(cls=11, clks=5) * 2) + 1

    dataset, dataset_gt, dataset_idxs = prepare_generator_dataset(dataset_folder, dataset_padding)

    kfold = KFold(n_splits=splits, shuffle=True, random_state=seed)

    kfold_splits = kfold.split(X=dataset_idxs)

    cmap = plt.cm.get_cmap('hsv', splits + 1)

    store_dir = 'storage/search-plan-comb/test'

    models_files = [f for f in listdir(models_folder) if isfile(join(models_folder, f)) and f.endswith('json')]
    models_weights = [f for f in listdir(models_folder) if isfile(join(models_folder, f)) and f.endswith('h5')]

    models_files = natsorted(models_files, key=lambda y: y.lower())
    models_weights = natsorted(models_weights, key=lambda y: y.lower())

    print(models_files)
    print(models_weights)

    for i, (train, test) in enumerate(kfold_splits):
        print("=========================================")
        print("===== K Fold Test step => %d/5 =====" % (i + 1))
        print("=========================================")

        json_file = open(join(models_folder, models_files[i]), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(join(models_folder, models_weights[i]))

        patch_size = get_padding(model.layers)
        if not augment:
            offset = int(DatasetConfig.MAX_PADDING / 2) - int(patch_size / 2)
        else:
            offset = 0

        kfold_netname = str(i + 1) + '_' + network_name

        # serialize model to JSON
        model_json = model.to_json()
        with open(store_dir + kfold_netname + ".json", "w") as json_file:
            json_file.write(model_json)

        evalgen = IndexBasedGenerator(batch_size=NetworkParameters.BATCH_SIZE,
                                      dataset=dataset,
                                      dataset_gt=dataset_gt,
                                      indexes=dataset_idxs[test],
                                      patch_size=patch_size,
                                      offset=offset)

        predict_out = model.predict_generator(generator=evalgen,
                                              steps=int(len(test) // NetworkParameters.BATCH_SIZE) + 1,
                                              use_multiprocessing=True,
                                              verbose=1)
        predict_out = np.argmax(predict_out, axis=1)

        expected = np.zeros(shape=(len(test),), dtype=np.uint8)
        for j, idx in enumerate(dataset_idxs[test]):
            expected[j] = dataset_gt[idx[0], idx[1], idx[2]]

        test_acc = accuracy_score(expected, predict_out)

        cm = plot_confusion_matrix(expected, predict_out, np.array(['No Forest', 'Forest']), plot=False)

        class_report = classification_report(expected, predict_out, target_names=np.array(['no forest', 'forest']))

        confmat_file = store_dir + kfold_netname + ".conf_mat.npz"

        if not exists(confmat_file):
            np.savez_compressed(confmat_file, cm=cm)

        metrics_filename = join(store_dir, kfold_netname + '-score_{test_acc:.4f}'.format(test_acc=test_acc) + '.txt')

        with open(metrics_filename, 'w') as output:
            output.write(str(cm))

        print(class_report)

        with open(metrics_filename, 'a') as output:
            output.write('\n\n' + str(class_report))

    return None


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


def get_padding_from_params(cls, clks, rot_ang=45, shear_ang=45):
    ws = 1

    # calculate patch size for network
    for cl in range(cls):
        if cl == 0:
            kern_size = clks
        else:
            kern_size = 3
        # padding = int(kern_size/2)
        # ws = (ws - 1) - (2 * padding) + kern_size
        ws = (ws - 1) + kern_size

    # calculate patch size for rotation
    rws = int((ws * np.sin(np.deg2rad(rot_ang))) + (ws * np.cos(np.deg2rad(rot_ang))))

    # calculate patch size for shearing
    sws = rws + abs(np.sin(np.deg2rad(shear_ang)) * rws)
    pad = int((sws - rws)) + 1

    return pad


def test(model_name, dataset_file, tif_sample, tif_real, result_name):
    if result_name != '':
        store_with_name = result_name
    else:
        store_with_name = model_name

    model_design_file = [f for f in listdir('storage') if f == model_name + '.json']
    model_weights_file = [f for f in listdir('storage') if f.startswith(model_name) and f.endswith('.h5')]

    assert (len(model_design_file) == 1)
    assert (len(model_weights_file) == 1)
    json_file = open(join("storage", model_design_file[0]), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(join("storage", model_weights_file[0]))

    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

    bigdata = None
    item_getter = itemgetter('bigdata')
    with np.load(dataset_file) as df:
        bigdata = item_getter(df)

    # bigdata_clip = bigdata[:, :900, :900]
    bigdata_clip = bigdata[:, :RasterParams.FNF_MAX_X, :RasterParams.FNF_MAX_Y]

    pad_amount = get_padding(loaded_model.layers)
    half_padding = int(pad_amount / 2)

    bigdata_clip = np.pad(bigdata_clip, [(0, 0), (half_padding, half_padding), (half_padding, half_padding)],
                          mode='constant')

    fnf_handler = GTiffHandler()
    # fnf_handler.readFile("storage/test_fullsize_train_pred.tif")
    fnf_handler.readFile(tif_sample)

    predict_generator = SortedPredictGenerator(batch_size=NetworkParameters.BATCH_SIZE, dataset=bigdata_clip,
                                               patch_size=pad_amount)

    start = time.time()
    predict_mask = loaded_model.predict_generator(generator=predict_generator, steps=int(
        (RasterParams.FNF_MAX_X * RasterParams.FNF_MAX_Y) / NetworkParameters.BATCH_SIZE) + 1, use_multiprocessing=True,
                                                  verbose=1)
    predict_mask = np.argmax(predict_mask, axis=1)
    predict_mask = predict_mask.reshape(RasterParams.FNF_MAX_X, RasterParams.FNF_MAX_Y)
    end = time.time()
    print(end - start)

    bigdata_gt = None
    item_getter = itemgetter('bigdata_gt')
    with np.load(dataset_file) as df:
        bigdata_gt = item_getter(df)

    bigdata_gt_clip = bigdata_gt

    error_mask = np.logical_xor(predict_mask, bigdata_gt_clip)

    temp_pred = predict_mask.reshape(predict_mask.shape[0] * predict_mask.shape[1])
    temp_gt = bigdata_gt_clip.reshape(bigdata_gt_clip.shape[0] * bigdata_gt_clip.shape[1])

    ax, cm = plot_confusion_matrix(temp_gt, temp_pred, np.array(['No Forest', 'Forest']), plot=True)

    plt.savefig('storage/' + store_with_name + '_conf_plot.pdf', bbox_inches='tight')

    print(classification_report(temp_gt, temp_pred, target_names=np.array(['no forest', 'forest'])))

    unique, counts = np.unique(error_mask, return_counts=True)
    print("Test accuracy ", counts[0] / (counts[0] + counts[1]))

    np.savez_compressed('storage/' + store_with_name + 'confusion_matrix.npz', cm=cm)

    plt.clf()

    fnf_handler.src_Z = predict_mask
    fnf_handler.writeNewFile('storage/test_' + store_with_name + '_prediction.tif')

    fnf_handler.src_Z = error_mask
    fnf_handler.writeNewFile('storage/test_' + store_with_name + '_error.tif')

    fnf_handler.closeFile()

    if tif_real != '':
        real_fnf_handler = GTiffHandler()
        real_fnf_handler.readFile(tif_real)

        real_mask = np.array(real_fnf_handler.src_Z)

        unique, counts = np.unique(real_mask, return_counts=True)
        if (unique.shape[0] > 2):
            real_mask[real_mask > 1] = 0

        predict_mask_portion = predict_mask[:real_mask.shape[0], :real_mask.shape[1]]
        bigdata_gt_portion = bigdata_gt_clip[:real_mask.shape[0], :real_mask.shape[1]]

        our_error_mask = np.logical_xor(predict_mask_portion, real_mask)
        fnf_error_mask = np.logical_xor(bigdata_gt_portion, real_mask)

        temp_real = real_mask.reshape(real_mask.shape[0] * real_mask.shape[1])
        temp_pred = predict_mask_portion.reshape(real_mask.shape[0] * real_mask.shape[1])
        temp_gt = bigdata_gt_portion.reshape(real_mask.shape[0] * real_mask.shape[1])

        ax, cm = plot_confusion_matrix(temp_real, temp_pred, np.array(['No Forest', 'Forest']), plot=True)
        plt.savefig('storage/' + store_with_name + '_conf_plot_real_vs_pred.pdf', bbox_inches='tight')
        print(classification_report(temp_real, temp_pred, target_names=np.array(['no forest', 'forest'])))
        np.savez_compressed('storage/' + store_with_name + 'confusion_matrix_real_vs_pred.npz', cm=cm)
        plt.clf()

        ax, cm = plot_confusion_matrix(temp_real, temp_gt, np.array(['No Forest', 'Forest']), plot=True)
        plt.savefig('storage/' + store_with_name + '_conf_plot_real_vs_gt.pdf', bbox_inches='tight')
        print(classification_report(temp_real, temp_gt, target_names=np.array(['no forest', 'forest'])))
        np.savez_compressed('storage/' + store_with_name + 'confusion_matrix_real_vs_gt.npz', cm=cm)
        plt.clf()

        unique, counts = np.unique(our_error_mask, return_counts=True)
        print("Test our accuracy ", counts[0] / (counts[0] + counts[1]))

        unique, counts = np.unique(fnf_error_mask, return_counts=True)
        print("Test gt accuracy ", counts[0] / (counts[0] + counts[1]))

        real_fnf_handler.src_Z = predict_mask_portion
        real_fnf_handler.writeNewFile('storage/test_' + store_with_name + '_prediction_portion.tif')

        real_fnf_handler.src_Z = our_error_mask
        real_fnf_handler.writeNewFile('storage/test_' + store_with_name + '_prediction_vs_real_error.tif')

        real_fnf_handler.src_Z = bigdata_gt_portion
        real_fnf_handler.writeNewFile('storage/test_' + store_with_name + '_gt_portion.tif')

        real_fnf_handler.src_Z = fnf_error_mask
        real_fnf_handler.writeNewFile('storage/test_' + store_with_name + '_gt_vs_real_error.tif')


def test_load_fullsize(model_filename, weights_filename, dataset_folder):
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_filename)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    test_fullsize(loaded_model, dataset_folder)


def test_fullsize(model, dataset_folder):
    path_to_pck = join(dataset_folder, 'dataset.npz')

    bigdata = None
    item_getter = itemgetter('bigdata')
    with np.load(path_to_pck) as df:
        bigdata = item_getter(df)

    pad_amount = int(SamplesConfig.PATCH_SIZE / 2)
    bigdata = np.pad(bigdata, [(0, 0), (pad_amount, pad_amount), (pad_amount, pad_amount)], mode='constant')

    predict_mask = predict_fullsize_mask(model, bigdata)

    pckmetadata_gt = None
    item_getter = itemgetter('pckmetadata_gt')
    with np.load(path_to_pck) as df:
        pckmetadata_gt = item_getter(df)

    fnf_handler = GTiffHandler()
    fnf_handler.readFile(pckmetadata_gt[0])

    error_mask = np.logical_xor(predict_mask, np.array(fnf_handler.src_Z, dtype=np.uint8))

    fnf_handler.src_Z = predict_mask
    fnf_handler.writeNewFile('storage/test_fullsize_pred.tif')

    fnf_handler.src_Z = error_mask
    fnf_handler.writeNewFile('storage/test_fullsize_error.tif')

    fnf_handler.closeFile()


def predict_fullsize_mask(model, input):
    # predict_mask = np.empty((RasterParams.FNF_MAX_X * RasterParams.FNF_MAX_Y), dtype=np.uint8)
    predict_mask = []
    patches_batch = []

    total = RasterParams.FNF_MAX_Y * RasterParams.FNF_MAX_X
    for y in range(0, RasterParams.FNF_MAX_Y):
        for x in range(0, RasterParams.FNF_MAX_X):
            item_num = (y * RasterParams.FNF_MAX_Y) + x

            patch = []
            for i in range(0, input.shape[0]):
                patch.append(input[i, y:y + SamplesConfig.PATCH_SIZE, x:x + SamplesConfig.PATCH_SIZE])

            patches_batch.append(patch)

            if len(patches_batch) == TestParameters.FULL_SIZE_BATCH_SIZE:
                patches_batch = np.array(patches_batch).reshape(len(patches_batch), SamplesConfig.PATCH_SIZE,
                                                                SamplesConfig.PATCH_SIZE, input.shape[0])
                result = model.predict_classes(patches_batch)
                predict_mask.extend(result)
                patches_batch = []

                simple_progress_bar(item_num, total)

    if len(patches_batch) > 0:
        patches_batch = np.array(patches_batch).reshape(len(patches_batch), SamplesConfig.PATCH_SIZE,
                                                        SamplesConfig.PATCH_SIZE, input.shape[0])
        result = model.predict_classes(patches_batch)
        predict_mask.extend(result)
        simple_progress_bar(total, total)

    predict_mask = np.array(predict_mask, dtype=np.uint8)
    predict_mask = predict_mask.reshape(RasterParams.FNF_MAX_X, RasterParams.FNF_MAX_Y)

    return predict_mask


def test_network(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def play():
    json_file = open("storage/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("storage/model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    dataset_file = "data/processed/train/12/dataset.npz"

    bigdata = None
    item_getter = itemgetter('bigdata')
    with np.load(dataset_file) as df:
        bigdata = item_getter(df)

    bigdata_clip = bigdata[:, 2700:, 2700:]

    pad_amount = int(SamplesConfig.PATCH_SIZE / 2)
    bigdata_clip = np.pad(bigdata_clip, [(0, 0), (pad_amount, pad_amount), (pad_amount, pad_amount)], mode='constant')

    predict_mask = predict_fullsize_mask(loaded_model, bigdata_clip)

    fnf_handler = GTiffHandler()
    fnf_handler.readFile("storage/test_fullsize_train_pred.tif")

    bigdata_gt = None
    item_getter = itemgetter('bigdata_gt')
    with np.load(dataset_file) as df:
        bigdata_gt = item_getter(df)

    bigdata_gt_clip = bigdata_gt[2700:, 2700:]

    error_mask = np.logical_xor(predict_mask, bigdata_gt_clip)

    fnf_handler.src_Z = predict_mask
    fnf_handler.writeNewFile('storage/test_fullsize_train12_pred.tif')

    fnf_handler.src_Z = error_mask
    fnf_handler.writeNewFile('storage/test_fullsize_train12_error.tif')

    fnf_handler.closeFile()


def play2(samples_file):
    train_x, train_y, test_x, test_y = divide_samples(samples_file)

    train_x = train_x.reshape(train_x.shape[0], SamplesConfig.PATCH_SIZE, SamplesConfig.PATCH_SIZE,
                              train_x.shape[1])
    test_x = test_x.reshape(test_x.shape[0], SamplesConfig.PATCH_SIZE, SamplesConfig.PATCH_SIZE,
                            test_x.shape[1])

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')

    train_y = keras.utils.to_categorical(train_y, 2)
    test_y = keras.utils.to_categorical(test_y, 2)

    json_file = open("storage/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("storage/model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    test_network(loaded_model, test_x, test_y)


def play3():
    json_file = open("storage/full_upsample.json", 'r')
    # json_file = open("storage/factor-5p-upsample.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("storage/balanced-10p-upsample_0.8539.h5")
    # loaded_model.load_weights("storage/temp/full-upsample-weights-improvement-04-0.98.hdf5")
    # loaded_model.load_weights("storage/factor-5p-upsample_0.9692.h5")

    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    dataset_file = "data/processed/test/3/dataset.npz"

    bigdata = None
    item_getter = itemgetter('bigdata')
    with np.load(dataset_file) as df:
        bigdata = item_getter(df)

    # bigdata_clip = bigdata[:, :900, :900]
    bigdata_clip = bigdata

    pad_amount = int(SamplesConfig.PATCH_SIZE / 2)
    bigdata_clip = np.pad(bigdata_clip, [(0, 0), (pad_amount, pad_amount), (pad_amount, pad_amount)], mode='constant')

    start = time.time()
    predict_mask = predict_fullsize_mask(loaded_model, bigdata_clip)
    end = time.time()
    print(end - start)

    sys.exit()

    fnf_handler = GTiffHandler()
    # fnf_handler.readFile("storage/test_fullsize_train_pred.tif")
    fnf_handler.readFile("storage/FNF_BALANCED_15.tif")

    bigdata_gt = None
    item_getter = itemgetter('bigdata_gt')
    with np.load(dataset_file) as df:
        bigdata_gt = item_getter(df)

    # bigdata_gt_clip = bigdata_gt[:900, :900]
    bigdata_gt_clip = bigdata_gt

    error_mask = np.logical_xor(predict_mask, bigdata_gt_clip)

    temp_pred = predict_mask.reshape(predict_mask.shape[0] * predict_mask.shape[1])
    temp_gt = bigdata_gt_clip.reshape(bigdata_gt_clip.shape[0] * bigdata_gt_clip.shape[1])

    ax, cm = plot_confusion_matrix(temp_gt, temp_pred, np.array(['No Forest', 'Forest']))

    plt.show()

    print(classification_report(temp_gt, temp_pred, target_names=np.array(['no forest', 'forest'])))

    unique, counts = np.unique(error_mask, return_counts=True)
    print("Test accuracy ", counts[0] / (counts[0] + counts[1]))

    np.savez_compressed('storage/test_balanced_test15_10p_conf.npz', cm=cm)

    fnf_handler.src_Z = predict_mask
    fnf_handler.writeNewFile('storage/test_balanced_test15_10p_pred.tif')

    fnf_handler.src_Z = error_mask
    fnf_handler.writeNewFile('storage/test_balanced_test15_10p_error.tif')

    fnf_handler.closeFile()


def play4():
    json_file = open("storage/full_upsample.json", 'r')
    # json_file = open("storage/factor-5p-upsample.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("storage/balanced-10p-upsample_0.8539.h5")
    # loaded_model.load_weights("storage/temp/full-upsample-weights-improvement-04-0.98.hdf5")
    # loaded_model.load_weights("storage/factor-5p-upsample_0.9692.h5")

    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    dataset_file = "data/processed/test/3/dataset.npz"

    bigdata = None
    item_getter = itemgetter('bigdata')
    with np.load(dataset_file) as df:
        bigdata = item_getter(df)

    # bigdata_clip = bigdata[:, :900, :900]
    bigdata_clip = bigdata[:, :RasterParams.FNF_MAX_X, :RasterParams.FNF_MAX_Y]

    pad_amount = int(SamplesConfig.PATCH_SIZE / 2)
    bigdata_clip = np.pad(bigdata_clip, [(0, 0), (pad_amount, pad_amount), (pad_amount, pad_amount)], mode='constant')

    predict_generator = SortedPredictGenerator(NetworkParameters.BATCH_SIZE, bigdata_clip)

    start = time.time()
    predict_mask = loaded_model.predict_generator(generator=predict_generator, steps=int(
        (RasterParams.FNF_MAX_X * RasterParams.FNF_MAX_Y) / NetworkParameters.BATCH_SIZE) + 1, use_multiprocessing=True,
                                                  verbose=1)
    predict_mask = np.argmax(predict_mask, axis=1)
    predict_mask = predict_mask.reshape(RasterParams.FNF_MAX_X, RasterParams.FNF_MAX_Y)
    end = time.time()
    print(end - start)

    fnf_handler = GTiffHandler()
    # fnf_handler.readFile("storage/test_fullsize_train_pred.tif")
    fnf_handler.readFile("storage/FNF_BALANCED_15.tif")

    bigdata_gt = None
    item_getter = itemgetter('bigdata_gt')
    with np.load(dataset_file) as df:
        bigdata_gt = item_getter(df)

    # bigdata_gt_clip = bigdata_gt[:900, :900]
    bigdata_gt_clip = bigdata_gt

    error_mask = np.logical_xor(predict_mask, bigdata_gt_clip)

    temp_pred = predict_mask.reshape(predict_mask.shape[0] * predict_mask.shape[1])
    temp_gt = bigdata_gt_clip.reshape(bigdata_gt_clip.shape[0] * bigdata_gt_clip.shape[1])

    ax, cm = plot_confusion_matrix(temp_gt, temp_pred, np.array(['No Forest', 'Forest']))

    plt.show()

    print(classification_report(temp_gt, temp_pred, target_names=np.array(['no forest', 'forest'])))

    unique, counts = np.unique(error_mask, return_counts=True)
    print("Test accuracy ", counts[0] / (counts[0] + counts[1]))

    np.savez_compressed('storage/test_balanced_test15_10p_conf_fff.npz', cm=cm)

    fnf_handler.src_Z = predict_mask
    fnf_handler.writeNewFile('storage/test_balanced_test15_10p_pred_fff.tif')

    fnf_handler.src_Z = error_mask
    fnf_handler.writeNewFile('storage/test_balanced_test15_10p_error_fff.tif')

    fnf_handler.closeFile()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          plot=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax, cm
    else:
        return cm


def simple_progress_bar(current_value, total):
    increments = 50
    percentual = ((current_value / total) * 100)
    i = int(percentual // (100 / increments))
    text = "\r[{0: <{1}}] {2}%".format('=' * i, increments, "{0:.2f}".format(percentual))
    print(text, end="\n" if percentual == 100 else "")




main(sys.argv[1:])
