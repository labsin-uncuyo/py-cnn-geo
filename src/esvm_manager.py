import sys, getopt
import gc
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np

from functools import partial
from natsort import natsorted
from operator import itemgetter
from os import listdir
from os.path import exists, isfile, join
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils.multiclass import unique_labels
from config import DatasetConfig, NetworkParameters, RasterParams

OPERATION_CREATE_ESVM = 30
OPERATION_PLAY = 99

import warnings
warnings.filterwarnings("ignore")

def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Entering Ensemble SVM manager')

    operation = None
    finish_earlier = False

    try:
        opts, args = getopt.getopt(argv, "hc:n:t:d:fp", ["create_esvm=", "neighbors=", "model_name=", "storage_directory=", "finish_earlier", "play"])
    except getopt.GetoptError:
        print('esvm_manager.py -h')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('esvm_manager.py -c <dataset_folder>\n')
            sys.exit()
        elif opt in ["-c", "--create_esvm"]:
            dataset_folder = arg
            operation = OPERATION_CREATE_ESVM
        elif opt in ["-n", "--neighbors"]:
            neighbors = int(arg)
        elif opt in ["-t", "--model_name"]:
            model_name = arg
        elif opt in ["-d", "--storage_directory"]:
            store_directory = arg
        elif opt in ["-f", "--finish_earlier"]:
            finish_earlier = True
        elif opt in ["-p", "--play"]:
            operation = OPERATION_PLAY

    if operation == OPERATION_CREATE_ESVM:
        print('Working with dataset file %s' % dataset_folder)
        print('Using %s neighbors' % neighbors)
        create_esvm(dataset_folder, neighbors, store_directory, model_name, finish_earlier)
    elif operation == OPERATION_PLAY:
        play()

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

def get_item_features(item, feature_groups=None, neighbors=None):
    data_patch = dataset[item[0], :, item[1]: item[1] + neighbors, item[2]: item[2] + neighbors]

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

    np_feat[(-1 * data_patch.shape[0]):] = np.reshape(patch_sliced,data_patch.shape[0])

    return np_feat

def get_item_gt(item):
    return dataset_gt[item[0], item[1], item[2]]

def get_patch_features(data_patch, feature_groups):
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

    np_feat[(-1 * data_patch.shape[0]):] = np.reshape(patch_sliced,data_patch.shape[0])

    return np_feat

def shuffle_train(a):
    assert len(a)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
    return shuffled_a

def process_particular_svm(X_train_process_tuple, feature_groups=None, neighbors=None, store_dir=None, model_n=None, seed=7, process_total=None):

    process_number = X_train_process_tuple[1]
    process_X_train = X_train_process_tuple[0]
    print('{0}/{1} - '.format(process_number + 1, process_total), end='')

    X_preprocessed_train_bag = np.zeros(shape=(process_X_train.shape[0], dataset.shape[1] * feature_groups), dtype=np.float32)
    Y_preprocessed_train_bag = np.zeros(shape=(process_X_train.shape[0]), dtype=np.uint8)

    for pxt_i, pxt_item in enumerate(process_X_train):
        X_preprocessed_train_bag[pxt_i] = get_item_features(pxt_item, feature_groups=feature_groups, neighbors=neighbors)
        Y_preprocessed_train_bag[pxt_i] = get_item_gt(pxt_item)

    svm = SVC(kernel='rbf', cache_size=5000, gamma='auto', verbose=0, random_state=seed, max_iter=-1)
    svm.fit(X_preprocessed_train_bag, Y_preprocessed_train_bag)

    # print('Storing model...')
    joblib.dump(svm, join(store_dir, model_n + '-{step:03d}'.format(step=process_number) + '.pkl'))

    X_preprocessed_train_bag = None
    Y_preprocessed_train_bag = None
    svm = None

    gc.collect()


def predict_particular_batch(X_test, feature_groups=None, neighbors=None, svm=None):

    X_preprocessed_test_bag = np.zeros(shape=(X_test.shape[0], dataset.shape[1] * feature_groups), dtype=np.float32)

    for pxt_i, pxt_item in enumerate(X_test):
        X_preprocessed_test_bag[pxt_i] = get_item_features(pxt_item, feature_groups=feature_groups, neighbors=neighbors)

    return svm.predict(X_preprocessed_test_bag)


def create_esvm(dataset_folder, neighbors, store_directory, model_name, finish_earlier):
    print('Starting operation of Ensembled SVM creation')

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    global dataset
    global dataset_gt
    dataset, dataset_gt, dataset_idxs = prepare_generator_dataset(dataset_folder, neighbors)

    feature_groups = int(neighbors / 2) + 1

    X_train, X_test = train_test_split(dataset_idxs, test_size=0.15, random_state=7)

    if store_directory is not '':
        store_dir = store_directory
    else:
        store_dir = 'storage/ersvm/'

    if model_name is not '':
        model_n = model_name
    else:
        model_n = 'svm_model'

    estimator_step_size = 50
    batch_size = (NetworkParameters.BATCH_SIZE)

    steps = int(X_train.shape[0] / batch_size) + 1
    estimator_steps = int(steps / estimator_step_size) + 1

    #svm = SVC(C=1.0, kernel='rbf', cache_size=1000, verbose=1, probability=True, random_state=seed)

    #sgd_lsvc = linear_model.SGDClassifier(random_state=seed, warm_start=True, verbose=1, n_jobs=-1, alpha=0.00001) # max_features=int(dataset.shape[1]/3)
    epochs=1

    if not True:
        estimator_steps_array = np.arange(estimator_steps)
        for e in range(epochs):
            print('\n==========================')
            print('         EPOCH %s         ' % str(e+1))
            print('==========================\n\n')
            np.random.shuffle(X_train)

            print('Train progress: ', end='')
            for i in range(estimator_steps):
                print('{0}/{1} - '.format(i + 1, estimator_steps), end='')
                start = (i * estimator_step_size) * batch_size
                end = ((i + 1) * estimator_step_size) * batch_size
                end = end if end < X_train.shape[0] else X_train.shape[0]

                X_preprocessed_train_bag = np.zeros(shape=(end-start, dataset.shape[1]*feature_groups), dtype=np.float32)
                Y_preprocessed_train_bag = np.zeros(shape=(end-start), dtype=np.uint8)
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                X_preprocessed_train_bag = np.array(pool.map(partial(get_item_features, feature_groups=feature_groups, neighbors=neighbors), X_train[start:end,:]))
                Y_preprocessed_train_bag = np.array(pool.map(get_item_gt, X_train[start:end,:]))
                pool.close()
                pool.join()

                #svm = SVC(C=1.0, kernel='rbf', cache_size=5000, gamma='auto', verbose=0, probability=True, random_state=seed, max_iter=1000)
                #svm = SVC(C=1.0, kernel='linear', cache_size=5000, gamma='auto', verbose=0, probability=True, random_state=seed, max_iter=1000)
                svm = SVC(kernel='rbf', cache_size=5000, gamma='auto', verbose=0, random_state=seed, max_iter=-1)
                svm.fit(X_preprocessed_train_bag, Y_preprocessed_train_bag)

                # print('Storing model...')
                joblib.dump(svm, join(store_dir, model_n + '-{step:03d}'.format(step=i) + '.pkl'))

                X_preprocessed_train_bag = None
                Y_preprocessed_train_bag = None
                svm = None

                gc.collect()
                if finish_earlier and i == 10:
                    break

            print('Finished!\n')

    if not True:

        Xts = np.split(X_train[:(batch_size * (estimator_steps - 1) * estimator_step_size)], estimator_steps - 1)
        Xts.append(X_train[(batch_size * (estimator_steps - 1) * estimator_step_size):])

        n_est_arange = np.arange(estimator_steps)

        Xts = zip(Xts, n_est_arange)

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
        pool.map(partial(process_particular_svm, feature_groups=feature_groups, neighbors=neighbors, store_dir=store_dir, model_n=model_n, seed=seed, process_total=estimator_steps), Xts)
        pool.close()
        pool.join()

        Xts = None
        gc.collect()

        print('Finished!\n')

    random_trees_files = [f for f in listdir(store_dir) if not isfile(f) and f.endswith('.pkl')]
    random_trees_files = natsorted(random_trees_files, key=lambda y: y.lower())
    n_estimators = len(random_trees_files)

    steps = len(random_trees_files)
    batch_size = int(X_test.shape[0] / steps) + 1

    predicted_test = np.zeros(shape=(X_test.shape[0],), dtype=np.uint8)
    expected_test = np.zeros(shape=(X_test.shape[0],), dtype=np.uint8)

    print('Starting testing phase...')

    values_votes = np.zeros(shape=(X_test.shape[0], ), dtype=np.float32)

    print('Test progress: ', end='')

    Xtes = np.split(X_test[:(batch_size * (steps - 1))], steps - 1)
    Xtes.append(X_test[(batch_size * (steps - 1)):])

    for i in range(steps):
        print('{0}/{1} - '.format(i + 1, steps), end='')

        start = i * batch_size
        end = (i + 1) * batch_size
        end = end if end < X_test.shape[0] else X_test.shape[0]
        for file_i, rf_filename in enumerate(random_trees_files):

            rf_full_filename = join(store_dir, rf_filename)
            svm = joblib.load(rf_full_filename)

            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            svm_predictions = np.concatenate(np.array(
                pool.map(partial(predict_particular_batch, feature_groups=feature_groups, neighbors=neighbors, svm=svm), Xtes))[:])
            pool.close()
            pool.join()

            if i == 0:
                values_votes = svm_predictions
            else:
                values_votes = np.add(values_votes, svm_predictions)

            svm = None
            svm_predictions = None
            gc.collect()

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        expected_test[start:end] = np.array(pool.map(get_item_gt, Xtes[i]))
        pool.close()
        pool.join()

    if not True:
        for i in range(steps):
            print('{0}/{1} - '.format(i + 1, steps), end='')
            start = (i) * batch_size
            end = (i + 1) * batch_size
            end = end if end < X_test.shape[0] else X_test.shape[0]

            X_partial_preprocessed_test_bag = np.zeros(shape=(end - start, dataset.shape[1] * feature_groups), dtype=np.float32)
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            X_partial_preprocessed_test_bag = np.array(pool.map(partial(get_item_features, feature_groups=feature_groups, neighbors=neighbors), X_test[start:end,:]))
            expected_test[start:end] = np.array(pool.map(get_item_gt, X_test[start:end, :]))
            pool.close()
            pool.join()

            for file_i, rf_filename in enumerate(random_trees_files):
                rf_full_filename = join(store_dir, rf_filename)
                svm = joblib.load(rf_full_filename)

                if file_i == 0:
                    batch_votes = svm.predict(X_partial_preprocessed_test_bag)
                else:
                    batch_votes = np.add(batch_votes, svm.predict(X_partial_preprocessed_test_bag))

                svm = None
                gc.collect()

            values_votes[start:end] = batch_votes
    print('Finished!\n')

    predicted_test = np.divide(values_votes, steps)
    predicted_test[predicted_test<0.5] = 0
    predicted_test[predicted_test>=0.5] = 1
    X_partial_preprocessed_test_bag = None
    batch_votes = None
    values_votes = None
    gc.collect()
    #predicted_test = np.argmax(predicted_test, axis=1)

    print('Calculating reports and metrics...')

    test_acc = accuracy_score(expected_test, predicted_test)
    print('Test score: {test_acc:.4f}'.format(test_acc=test_acc))

    print('Storing value accuracy...')
    metrics_filename = join(store_dir, model_n + '-score_{test_acc:.4f}'.format(test_acc=test_acc) + '.txt')

    cm = print_confusion_matrix(expected_test, predicted_test, np.array(['No Forest', 'Forest']))

    with open(metrics_filename, 'w') as output:
        output.write(str(cm))

    class_report = classification_report(expected_test, predicted_test, target_names=np.array(['no forest', 'forest']))

    print(class_report)

    with open(metrics_filename, 'a') as output:
        output.write('\n\n' + str(class_report))

    confmat_file = join(store_dir, model_n + '.conf_mat.npz')

    print('Storing confusion matrix...')
    if not exists(confmat_file):
        np.savez_compressed(confmat_file, cm=cm)

    print('All done!!!')


def play():
    print('Starting operation of play :)')
    pass


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

main(sys.argv[1:])