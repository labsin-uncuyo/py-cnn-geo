import sys, getopt
import gc
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np

from functools import partial
from operator import itemgetter
from os import listdir
from os.path import exists, isfile, join
from sklearn import svm
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from config import DatasetConfig, NetworkParameters, RasterParams

OPERATION_CREATE_SVM = 30
OPERATION_PLAY = 99

def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Entering SVM manager')

    operation = None
    finish_earlier = False

    try:
        opts, args = getopt.getopt(argv, "hc:n:t:d:fp", ["create_svm=", "neighbors=", "model_name=", "storage_directory=", "finish_earlier", "play"])
    except getopt.GetoptError:
        print('svm_manager.py -h')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('svm_manager.py -c <dataset_folder>\n')
            sys.exit()
        elif opt in ["-c", "--create_svm"]:
            dataset_folder = arg
            operation = OPERATION_CREATE_SVM
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

    if operation == OPERATION_CREATE_SVM:
        print('Working with dataset file %s' % dataset_folder)
        print('Using %s neighbors' % neighbors)
        create_svm(dataset_folder, neighbors, store_directory, model_name, finish_earlier)
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

def create_svm(dataset_folder, neighbors, store_directory, model_name, finish_earlier):
    print('Starting operation of SVM creation')

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
        store_dir = 'storage/svm/'

    if model_name is not '':
        model_n = model_name
    else:
        model_n = 'svm_model'

    estimator_step_size = 500
    batch_size = (NetworkParameters.BATCH_SIZE*2)

    steps = int(X_train.shape[0] / batch_size) + 1
    estimator_steps = int(steps / estimator_step_size) + 1

    sgd_lsvc = linear_model.SGDClassifier(random_state=seed, warm_start=True, verbose=1, n_jobs=-1, alpha=0.00001) # max_features=int(dataset.shape[1]/3)
    epochs=1
    for e in range(epochs):
        print('\n==========================')
        print('         EPOCH %s         ' % str(e+1))
        print('==========================\n\n')
        np.random.shuffle(X_train)
        for i in range(estimator_steps):
            print('Step %s of %s' % (str(i+1), str(estimator_steps)))
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

            sgd_lsvc.partial_fit(X_preprocessed_train_bag, Y_preprocessed_train_bag, classes=[0,1])

            gc.collect()

            if finish_earlier and i == 10:
                break

    print('Storing model...')
    joblib.dump(sgd_lsvc, join(store_dir, model_n + '.pkl'))

    print('Testing...')
    X_preprocessed_test_bag = np.zeros(shape=(X_test.shape[0], dataset.shape[1] * feature_groups), dtype=np.float32)
    Y_preprocessed_test_bag = np.zeros(shape=(X_test.shape[0],), dtype=np.uint8)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    X_preprocessed_test_bag = np.array(
        pool.map(partial(get_item_features, feature_groups=feature_groups, neighbors=neighbors), X_test))
    Y_preprocessed_test_bag = np.array(pool.map(get_item_gt, X_test))
    pool.close()
    pool.join()
    #for item_i, test_item in enumerate(X_test):
    #    X_preprocessed_test_bag[item_i] = get_patch_features(
    #        dataset[test_item[0], :, test_item[1]: test_item[1] + neighbors, test_item[2]: test_item[2] + neighbors],
    #        feature_groups)
    #    Y_preprocessed_test_bag[item_i] = dataset_gt[test_item[0], test_item[1], test_item[2]]

    #print(clf.score(X_preprocessed_test_bag, Y_preprocessed_test_bag))

    predict_out = sgd_lsvc.predict(X_preprocessed_test_bag)

    print('Calculating reports and metrics...')

    cm = plot_confusion_matrix(Y_preprocessed_test_bag, predict_out, np.array(['No Forest', 'Forest']), plot=False)

    print(classification_report(Y_preprocessed_test_bag, predict_out, target_names=np.array(['no forest', 'forest'])))

    confmat_file = join(store_dir, model_n + '.conf_mat.npz')

    print('Storing confusion matrix...')
    if not exists(confmat_file):
        np.savez_compressed(confmat_file, cm=cm)

    print('All done!!!')


def play():
    print('Starting operation of play :)')
    pass

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

main(sys.argv[1:])