import sys, getopt
import gc
import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np

from functools import partial
from operator import itemgetter
from natsort import natsorted
from os import listdir
from os.path import exists, isfile, join
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.multiclass import unique_labels
from config import DatasetConfig, NetworkParameters, RasterParams, SamplesConfig
from raster_rw import GTiffHandler

OPERATION_CREATE_SVM_WITH_GENERATOR_AND_KFOLD = 33
OPERATION_TEST = 50
OPERATION_PLAY = 99


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Entering SVM manager')

    operation = None

    dataset_file = None

    neighbors = None
    use_vector = False
    reduction_factor = None
    model_name = ''
    model_directory = ''
    store_directory = ''
    finish_earlier = False
    tif_sample = ''
    tif_real = ''
    result_name = ''
    augment = False
    aug_granularity = 1

    try:
        opts, args = getopt.getopt(argv, "hc:a:n:t:vl:d:fe:r:o:g:pi:z:",
                                   ["help", "create_svm_gen_kfold=", "neighbors=", "model_name=", "use_vector",
                                    "reduction_factor", "storage_directory=", "tif_sample=", "tif_real", "result_name=",
                                    "augment=", "play", "model_directory", "test="])
    except getopt.GetoptError:
        print('svm_manager.py -h')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('svm_manager.py -c <dataset_folder>\n')
            sys.exit()
        elif opt in ["-a", "--create_svm_gen_kfold"]:
            dataset_folder = arg
            operation = OPERATION_CREATE_SVM_WITH_GENERATOR_AND_KFOLD
        elif opt in ["-z", "--test"]:
            dataset_file = arg
            operation = OPERATION_TEST
        elif opt in ["-n", "--neighbors"]:
            neighbors = int(arg)
        elif opt in ["-v", "--use_vector"]:
            use_vector = True
        elif opt in ["-l", "--reduction_factor"]:
            reduction_factor = int(arg)
        elif opt in ["-t", "--model_name"]:
            model_name = arg
        elif opt in ["-i", "--model_directory"]:
            model_directory = arg
        elif opt in ["-d", "--storage_directory"]:
            store_directory = arg
        elif opt in ["-e", "--tif_sample"]:
            tif_sample = arg
        elif opt in ["-r", "--tif_real"]:
            tif_real = arg
        elif opt in ["-o", "--result_name"]:
            result_name = arg
        elif opt in ["-g", "--augment"]:
            augment = True
            aug_granularity = int(arg)
        elif opt in ["-p", "--play"]:
            operation = OPERATION_PLAY

    print('Working with dataset file %s' % dataset_folder)
    print('Using %s neighbors' % neighbors)
    if operation == OPERATION_CREATE_SVM_WITH_GENERATOR_AND_KFOLD:
        train_svm_kfold(dataset_folder, model_name, 5, neighbors, use_vector, reduction_factor, augment, aug_granularity)
    elif operation == OPERATION_TEST:
        pass
        # test(model_name, dataset_file, tif_sample, tif_real, result_name, model_directory, neighbors, use_vector, reduction_factor)
    elif operation == OPERATION_PLAY:
        pass
        # play()

    sys.exit()


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


def train_svm_kfold(dataset_folder, svm_name, splits, neighbors=9, vec=False, reduction_factor=4, augment=False,
                   aug_granularity=1):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    if not augment:
        dataset_padding = neighbors
    else:
        dataset_padding = neighbors * 2

    global dataset
    global dataset_gt
    dataset, dataset_gt, dataset_idxs = prepare_generator_dataset(dataset_folder, dataset_padding)

    kfold = KFold(n_splits=splits, shuffle=True, random_state=seed)

    kfold_splits = kfold.split(X=dataset_idxs)

    store_dir = 'storage/svm-kfold/'

    if vec:
        center = int(np.floor(neighbors / 2))
        paths = calculate_feature_paths(center, reduction_factor)
    else:
        center = None
        paths = None

    for j, (train, test) in enumerate(kfold_splits):
        print("=========================================")
        print("===== K Fold Validation step => %d/5 =====" % (j + 1))
        print("=========================================")

        model = SGDClassifier(loss='hinge', alpha=1e-5, penalty='elasticnet', random_state=seed, verbose=1, n_jobs=-1,
                          max_iter=1500, early_stopping=False, validation_fraction=0.15, n_iter_no_change=20)

        train_idxs, validation_idxs = divide_indexes(dataset_idxs[train], val_percentage=0.15)

        feature_groups = int(neighbors / 2) + 1

        estimator_step_size = 500
        batch_size = (NetworkParameters.BATCH_SIZE * 2)

        steps = int(train_idxs.shape[0] / batch_size) + 1
        estimator_steps = int(steps / estimator_step_size) + 1

        if paths is not None:
            feature_extraction_fn = get_item_features_vector
            n_features = (paths.shape[0] + 1) * dataset.shape[1]
        else:
            feature_extraction_fn = get_item_features
            n_features = dataset.shape[1] * feature_groups

        print('Starting training phase...')

        print('Train progress: ', end='')

        epochs = 30
        for e in range(epochs):
            print('\n========== EPOCH %s ===========\n' % str(e + 1))

            train_idxs = shuffle_in_unison(train_idxs)
            # np.random.shuffle(X_train)
            for i in range(estimator_steps):
                print('{0}/{1} - '.format(i + 1, estimator_steps), end='')
                start = (i * estimator_step_size) * batch_size
                end = ((i + 1) * estimator_step_size) * batch_size
                end = end if end < train_idxs.shape[0] else train_idxs.shape[0]

                X_preprocessed_train_bag = np.zeros(shape=(end - start, n_features), dtype=np.float32)
                Y_preprocessed_train_bag = np.zeros(shape=(end - start), dtype=np.uint8)

                for pxt_i, pxt_item in enumerate(train_idxs[start:end, :]):
                    X_preprocessed_train_bag[pxt_i] = feature_extraction_fn(pxt_item,
                                                                                 feature_groups=feature_groups,
                                                                                 neighbors=neighbors, paths=paths,
                                                                                 center=center)
                    Y_preprocessed_train_bag[pxt_i] = get_item_gt(pxt_item)

                # pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()/2))
                # X_preprocessed_train_bag = np.array(pool.map(partial(self.feature_extraction_fn, feature_groups=self.feature_groups, neighbors=self.neighbors, paths=self.paths, center=self.center), X_train[start:end,:]))
                # Y_preprocessed_train_bag = np.array(pool.map(get_item_gt, X_train[start:end,:]))
                # pool.close()
                # pool.join()

                model.partial_fit(X_preprocessed_train_bag, Y_preprocessed_train_bag, classes=[0, 1])
                X_preprocessed_train_bag = None
                Y_preprocessed_train_bag = None

                gc.collect()

            print('Epoch Finished!\n')

            X_partial_preprocessed_val_bag = np.zeros(shape=(validation_idxs.shape[0], dataset.shape[1] * n_features),
                                                      dtype=np.float32)
            expected_val = np.zeros(shape=(validation_idxs.shape[0],), dtype=np.uint8)
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            X_partial_preprocessed_val_bag = np.array(
                pool.map(
                    partial(feature_extraction_fn, feature_groups=feature_groups,
                            neighbors=neighbors, paths=paths, center=center), validation_idxs))
            expected_val = np.array(pool.map(get_item_gt, validation_idxs))
            pool.close()
            pool.join()

            predicted_val = model.predict(X_partial_preprocessed_val_bag)

            val_acc = accuracy_score(expected_val, predicted_val)

            X_partial_preprocessed_val_bag = None
            expected_val = None
            predicted_val = None

            print('Validation score: {val_acc:.4f}\n'.format(val_acc=val_acc))

            # print('Storing model...')
            joblib.dump(model, join(store_dir,
                                         svm_name + '-epoch_{epoch:02d}-val_acc_{val_acc:.4f}'.format(epoch=e,
                                                                                                       val_acc=val_acc) + '.pkl'))

        train_idxs = None
        validation_idxs = None
        gc.collect()

        print('Finished!\n')

        print('Starting validation phase...')

        test_idx = dataset_idxs[test]

        print('Validation progress: ', end='')

        X_preprocessed_test_bag = np.zeros(shape=(test_idx.shape[0], dataset.shape[1] * feature_groups),
                                           dtype=np.float32)
        expected_test = np.zeros(shape=(test_idx.shape[0],), dtype=np.uint8)

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        X_preprocessed_test_bag = np.array(
            pool.map(partial(feature_extraction_fn, feature_groups=feature_groups,
                             neighbors=neighbors, paths=paths, center=center), test_idx))
        expected_test = np.array(pool.map(get_item_gt, test_idx))
        pool.close()
        pool.join()
        # for item_i, test_item in enumerate(X_test):
        #    X_preprocessed_test_bag[item_i] = get_patch_features(
        #        dataset[test_item[0], :, test_item[1]: test_item[1] + neighbors, test_item[2]: test_item[2] + neighbors],
        #        feature_groups)
        #    Y_preprocessed_test_bag[item_i] = dataset_gt[test_item[0], test_item[1], test_item[2]]

        # print(clf.score(X_preprocessed_test_bag, Y_preprocessed_test_bag))

        predicted_test = model.predict(X_preprocessed_test_bag)

        print('Calculating reports and metrics...')

        test_acc = accuracy_score(expected_test, predicted_test)
        print('Test score: {test_acc:.4f}'.format(test_acc=test_acc))

        print('Storing value accuracy...')
        metrics_filename = join(store_dir,
                                svm_name + '-score_{test_acc:.4f}'.format(test_acc=test_acc) + '.txt')

        cm = print_confusion_matrix(expected_test, predicted_test, np.array(['No Forest', 'Forest']))

        with open(metrics_filename, 'w') as output:
            output.write(str(cm))

        class_report = classification_report(expected_test, predicted_test,
                                             target_names=np.array(['no forest', 'forest']))

        print(class_report)

        with open(metrics_filename, 'a') as output:
            output.write('\n\n' + str(class_report))

        confmat_file = join(store_dir, svm_name + '.conf_mat.npz')

        print('Storing confusion matrix...')
        if not exists(confmat_file):
            np.savez_compressed(confmat_file, cm=cm)


def print_confusion_matrix(y_true, y_pred, classes,
                           normalize=False,
                           title=None):
    """
    This function prints the confusion matrix.
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
    return cm


def shuffle_in_unison(a):
    assert len(a)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
    return shuffled_a


def divide_indexes(indexes_file, val_percentage=SamplesConfig.TEST_PERCENTAGE):
    split_1 = int(val_percentage * len(indexes_file))

    train_idxs = indexes_file[split_1:]
    validation_idxs = indexes_file[:split_1]

    return train_idxs, validation_idxs


def path_values(paths, path_i, center, traslation_x, traslation_y):
    for i in range(1, center + 1):
        r = np.around(np.around((i * traslation_x) + center, decimals=1))
        h = np.around(np.around((i * traslation_y) + center, decimals=1))
        paths[path_i, i - 1] = [r, h]


def calculate_feature_paths(center, step):
    paths = np.zeros(shape=(int(center * 2 * 4 / step), center, 2), dtype=np.int8)

    # top of the matrix
    for path_i, h in enumerate(range(-1 * center, center, step)):
        v = -1 * center
        traslation_x = h / center
        traslation_y = v / center
        path_values(paths, path_i, center, traslation_x, traslation_y)

    # right of the matrix
    path_current = path_i + 1
    for path_i, v in enumerate(range(-1 * center, center, step)):
        h = 1 * center
        traslation_x = h / center
        traslation_y = v / center
        path_values(paths, path_current + path_i, center, traslation_x, traslation_y)

    # bottom of the matrix
    path_current += path_i + 1
    for path_i, h in enumerate(range(center, -1 * center, -1 * step)):
        v = 1 * center
        traslation_x = h / center
        traslation_y = v / center
        path_values(paths, path_current + path_i, center, traslation_x, traslation_y)

    # right of the matrix
    path_current += path_i + 1
    for path_i, v in enumerate(range(center, -1 * center, -1 * step)):
        h = -1 * center
        traslation_x = h / center
        traslation_y = v / center
        path_values(paths, path_current + path_i, center, traslation_x, traslation_y)

    paths = np.apply_along_axis(np.append, 1, paths, center)
    return paths


def get_item_patch(item, neighbors):
    return dataset[item[0], :, item[1]: item[1] + neighbors, item[2]: item[2] + neighbors]


def get_patch_features(data_patch, feature_groups=None, neighbors=None, paths=None, center=None):
    np_feat = np.zeros(shape=(data_patch.shape[0] * feature_groups,), dtype=np.float32)
    patch_sliced = data_patch
    for i in range(feature_groups - 1):
        top_row = patch_sliced[:, 0, :]
        bottom_row = patch_sliced[:, -1, :]

        patch_sliced = patch_sliced[:, 1:-1, :]
        left_col = patch_sliced[:, :, 0]
        right_col = patch_sliced[:, :, -1]

        patch_sliced = patch_sliced[:, :, 1:-1]

        border_values = np.concatenate((left_col, right_col, top_row, bottom_row), axis=1)
        np_feat[(i * data_patch.shape[0]):((i + 1) * data_patch.shape[0])] = np.mean(border_values, axis=1)

    np_feat[(-1 * data_patch.shape[0]):] = np.reshape(patch_sliced, data_patch.shape[0])

    return np_feat


def get_item_features(item, feature_groups=None, neighbors=None, paths=None, center=None):
    # Paths and center are not used in this function para we keep the parameter to maintain the same function definition
    # as get_item_features_vector and avoid an if statement being executed billion of times

    data_patch = get_item_patch(item, neighbors)

    return get_patch_features(data_patch, feature_groups=feature_groups, neighbors=neighbors, paths=paths,
                              center=center)


def get_item_gt(item):
    return dataset_gt[item[0], item[1], item[2]]


def get_patch_features_vector(data_patch, feature_groups=None, neighbors=None, paths=None, center=None):
    data_feat = np.array([np.mean([data_patch[:, g[0], g[1]] for g in f], axis=0) for f in paths], dtype=np.float32)

    data_center = data_patch[:, center, center]
    data_center = data_center.reshape(1, data_center.shape[0])
    data_feat = np.append(data_feat, data_center, axis=0)

    data_feat = data_feat.reshape(data_feat.shape[0] * data_feat.shape[1])
    return data_feat


def get_item_features_vector(item, feature_groups=None, neighbors=None, paths=None, center=None):
    # Feature groups are not used in this function para we keep the parameter to maintain the same function definition
    # as get_item_features and avoid an if statement being executed billion of times

    data_patch = get_item_patch(item, neighbors)

    return get_patch_features_vector(data_patch, feature_groups=feature_groups, neighbors=neighbors, paths=paths,
                                     center=center)


def get_batch(idx, batch_size, total_samples, patch_size):
    patches_batch = []
    left_lim = idx * batch_size
    right_lim = (idx + 1) * batch_size if (idx + 1) * batch_size <= total_samples else total_samples

    y_start = int(left_lim / RasterParams.FNF_MAX_Y)
    y_end = int(right_lim / RasterParams.FNF_MAX_Y)

    for y in range(y_start, y_end + 1):
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
            for i in range(0, dataset.shape[0]):
                patch.append(dataset[i, y:y + patch_size, x:x + patch_size])

            patches_batch.append(patch)

    # patches_batch = np.array(patches_batch).reshape(len(patches_batch), self.patch_size, self.patch_size, self.dataset.shape[0])
    patches_batch = np.array(patches_batch)
    patches_batch = patches_batch.astype('float32')
    return patches_batch


main(sys.argv[1:])