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
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.multiclass import unique_labels
from config import DatasetConfig, NetworkParameters, RasterParams, SamplesConfig
from raster_rw import GTiffHandler

OPERATION_CREATE_RF = 30
OPERATION_CREATE_RF_WITH_GENERATOR_AND_KFOLD = 33
OPERATION_TEST = 50
OPERATION_PLAY = 99


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Entering random forest manager')

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
                                   ["help", "create_rf=", "create_rf_gen_kfold=", "neighbors=", "model_name=",
                                    "use_vector", "reduction_factor", "storage_directory=", "finish_earlier",
                                    "tif_sample=", "tif_real", "result_name=", "augment=", "play", "model_directory",
                                    "test="])
    except getopt.GetoptError:
        print('random_forest_manager.py -h')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('random_forest_manager.py -c <dataset_folder>\n')
            sys.exit()
        elif opt in ["-c", "--create_rf"]:
            dataset_folder = arg
            operation = OPERATION_CREATE_RF
        elif opt in ["-a", "--create_rf_gen_kfold"]:
            dataset_folder = arg
            operation = OPERATION_CREATE_RF_WITH_GENERATOR_AND_KFOLD
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
        elif opt in ["-f", "--finish_earlier"]:
            finish_earlier = True
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

    if operation == OPERATION_CREATE_RF:
        print('Working with dataset file %s' % dataset_folder)
        print('Using %s neighbors' % neighbors)
        create_random_forest(dataset_folder, neighbors, store_directory, model_name, finish_earlier)
    elif operation == OPERATION_CREATE_RF_WITH_GENERATOR_AND_KFOLD:
        train_rf_kfold(dataset_folder, model_name, 5, neighbors, use_vector, reduction_factor, augment, aug_granularity)
    elif operation == OPERATION_TEST:
        test(model_name, dataset_file, tif_sample, tif_real, result_name, model_directory, neighbors, use_vector,
             reduction_factor)
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


def create_random_forest(dataset_folder, neighbors, store_directory, model_name, finish_earlier):
    print('Starting operation of random forest creation')

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
        store_dir = 'storage/rf/'

    if model_name is not '':
        model_n = model_name
    else:
        model_n = 'rf_model'

    estimator_step_size = 100
    batch_size = (NetworkParameters.BATCH_SIZE * 2)

    steps = int(X_train.shape[0] / batch_size) + 1
    estimator_steps = int(steps / estimator_step_size) + 1

    clf = RandomForestClassifier(n_estimators=estimator_step_size, max_features=int(dataset.shape[1] / 3),
                                 warm_start=True, verbose=1, n_jobs=-1)
    for i in range(estimator_steps):
        print('Step %s of %s' % (str(i + 1), str(estimator_steps)))
        print('Fitting %s estimators' % str(clf.n_estimators))
        start = (i * estimator_step_size) * batch_size
        end = ((i + 1) * estimator_step_size) * batch_size
        end = end if end < X_train.shape[0] else X_train.shape[0]

        X_preprocessed_train_bag = np.zeros(shape=(end - start, dataset.shape[1] * feature_groups), dtype=np.float32)
        Y_preprocessed_train_bag = np.zeros(shape=(end - start), dtype=np.uint8)
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        X_preprocessed_train_bag = np.array(
            pool.map(partial(get_item_features, feature_groups=feature_groups, neighbors=neighbors),
                     X_train[start:end, :]))
        Y_preprocessed_train_bag = np.array(pool.map(get_item_gt, X_train[start:end, :]))
        pool.close()
        pool.join()

        clf.fit(X_preprocessed_train_bag, Y_preprocessed_train_bag)

        gc.collect()

        if finish_earlier and clf.n_estimators == 1000:
            break

        if i + 1 != estimator_steps:
            if i + 1 != estimator_steps - 1:
                clf.set_params(n_estimators=clf.n_estimators + estimator_step_size)
            else:
                estimators_left = int((X_train.shape[0] - end) / batch_size) + 1
                clf.set_params(n_estimators=clf.n_estimators + estimators_left)

    print('Storing model...')
    joblib.dump(clf, join(store_dir, model_n + '.pkl'))

    print('Testing...')
    X_preprocessed_test_bag = np.zeros(shape=(X_test.shape[0], dataset.shape[1] * feature_groups), dtype=np.float32)
    Y_preprocessed_test_bag = np.zeros(shape=(X_test.shape[0],), dtype=np.uint8)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    X_preprocessed_test_bag = np.array(
        pool.map(partial(get_item_features, feature_groups=feature_groups, neighbors=neighbors), X_test))
    Y_preprocessed_test_bag = np.array(pool.map(get_item_gt, X_test))
    pool.close()
    pool.join()
    # for item_i, test_item in enumerate(X_test):
    #    X_preprocessed_test_bag[item_i] = get_patch_features(
    #        dataset[test_item[0], :, test_item[1]: test_item[1] + neighbors, test_item[2]: test_item[2] + neighbors],
    #        feature_groups)
    #    Y_preprocessed_test_bag[item_i] = dataset_gt[test_item[0], test_item[1], test_item[2]]

    # print(clf.score(X_preprocessed_test_bag, Y_preprocessed_test_bag))

    predict_out = clf.predict(X_preprocessed_test_bag)

    print('Calculating reports and metrics...')

    cm = plot_confusion_matrix(Y_preprocessed_test_bag, predict_out, np.array(['No Forest', 'Forest']), plot=False)

    print(classification_report(Y_preprocessed_test_bag, predict_out, target_names=np.array(['no forest', 'forest'])))

    confmat_file = join(store_dir, model_n + '.conf_mat.npz')

    print('Storing confusion matrix...')
    if not exists(confmat_file):
        np.savez_compressed(confmat_file, cm=cm)

    print('All done!!!')


def train_rf_kfold(dataset_folder, rf_name, splits, neighbors=9, vec=False, reduction_factor=4, augment=False,
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

    store_dir = 'storage/rf-kfold/'

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

        model = RandomForestClassifier(n_estimators=50, max_features=37, criterion='gini', min_samples_split=2,
                                       min_samples_leaf=1, warm_start=True, verbose=0, n_jobs=-1)

        train_idxs, validation_idxs = divide_indexes(dataset_idxs[train], val_percentage=0.15)

        feature_groups = int(neighbors / 2) + 1

        estimator_step_size = 50
        n_estimators = 250

        steps = int(n_estimators / estimator_step_size) + 1
        batch_size = int(train_idxs.shape[0] / steps) + 1

        if paths is not None:
            feature_extraction_fn = get_item_features_vector
            n_features = (paths.shape[0] + 1) * paths.shape[1]
        else:
            feature_extraction_fn = get_item_features
            n_features = dataset.shape[1] * feature_groups

        print('Starting training phase...')

        final_estimators = n_estimators

        print('Train progress: ', end='')

        if False:
            for i in range(steps):
                print('{0}/{1} - '.format(i + 1, steps), end='')
                # self.progress_bar(steps, i + 1, 'Train', stdscr)
                start = (i) * batch_size
                end = (i + 1) * batch_size
                end = end if end < train_idxs.shape[0] else train_idxs.shape[0]

                X_preprocessed_train_bag = np.zeros(shape=(end - start, n_features),
                                                    dtype=np.float32)
                Y_preprocessed_train_bag = np.zeros(shape=(end - start), dtype=np.uint8)
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                X_preprocessed_train_bag = np.array(
                    pool.map(
                        partial(feature_extraction_fn, feature_groups=feature_groups,
                                neighbors=neighbors,
                                paths=paths, center=center), train_idxs[start:end, :]))
                Y_preprocessed_train_bag = np.array(pool.map(get_item_gt, train_idxs[start:end, :]))
                pool.close()
                pool.join()

                model.fit(X_preprocessed_train_bag, Y_preprocessed_train_bag)
                X_preprocessed_train_bag = None
                Y_preprocessed_train_bag = None
                gc.collect()

                # print('Storing model...')
                joblib.dump(model, join(store_dir, str(j + 1) + '_' + rf_name + '-{step:03d}'.format(step=i) + '.pkl'))

                model = None
                gc.collect()

                if i + 1 != steps:
                    model = RandomForestClassifier(n_estimators=50, max_features=37, criterion='gini',
                                                   min_samples_split=2, min_samples_leaf=1, warm_start=True, verbose=0,
                                                   n_jobs=-1)

        print('Finished!\n')

        print('Starting validation phase...')

        val_batch_size = int(test.shape[0] / steps) + 1

        random_trees_files = [f for f in listdir(store_dir) if
                              isfile(join(store_dir, f)) and f.startswith(str(j + 1) + '_') and f.endswith('.pkl')]
        random_trees_files = natsorted(random_trees_files, key=lambda y: y.lower())

        values_votes = np.zeros(shape=(test.shape[0], 2), dtype=np.float32)
        expected_val = np.zeros(shape=(test.shape[0],), dtype=np.uint8)

        print('Validation progress: ', end='')

        for i in range(steps):
            print('{0}/{1} - '.format(i + 1, steps), end='')
            start = (i) * val_batch_size
            end = (i + 1) * val_batch_size
            end = end if end < test.shape[0] else test.shape[0]

            X_partial_preprocessed_val_bag = np.zeros(shape=(end - start, n_features), dtype=np.float32)
            # expected_val[start:end] = np.zeros(shape=(end - start), dtype=np.uint8)
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            X_partial_preprocessed_val_bag = np.array(
                pool.map(
                    partial(feature_extraction_fn, feature_groups=feature_groups,
                            neighbors=neighbors, paths=paths, center=center), test[start:end, :]))
            expected_val[start:end] = np.array(pool.map(get_item_gt, test[start:end, :]))
            pool.close()
            pool.join()

            for file_i, rf_filename in enumerate(random_trees_files):
                rf_full_filename = join(store_dir, rf_filename)

                model = joblib.load(rf_full_filename)

                if file_i == 0:
                    batch_votes = np.multiply(model.predict_proba(X_partial_preprocessed_val_bag),
                                              model.n_estimators)
                else:
                    batch_votes = np.add(batch_votes,
                                         np.multiply(model.predict_proba(X_partial_preprocessed_val_bag),
                                                     model.n_estimators))

                model = None
                gc.collect()

            values_votes[start:end] = batch_votes

        print('Finished!')

        predicted_val = np.divide(values_votes, final_estimators)
        X_partial_preprocessed_val_bag = None
        batch_votes = None
        values_votes = None
        gc.collect()
        predicted_val = np.argmax(predicted_val, axis=1)

        print('Calculating reports and metrics...')

        val_acc = accuracy_score(expected_val, predicted_val)
        print('Val score: {val_acc:.4f}'.format(val_acc=val_acc))

        print('Storing value accuracy...')
        metrics_filename = join(store_dir,
                                str(j + 1) + '_' + rf_name + '-score_{val_acc:.4f}'.format(val_acc=val_acc) + '.txt')

        cm = plot_confusion_matrix(expected_val, predicted_val, np.array(['No Forest', 'Forest']), plot=False)

        with open(metrics_filename, 'w') as output:
            output.write(str(cm))

        class_report = classification_report(expected_val, predicted_val,
                                             target_names=np.array(['no forest', 'forest']))

        print(class_report)

        with open(metrics_filename, 'a') as output:
            output.write('\n\n' + str(class_report))

        confmat_file = join(store_dir, str(j + 1) + '_' + rf_name + '.conf_mat.npz')

        print('Storing confusion matrix...')
        if not exists(confmat_file):
            np.savez_compressed(confmat_file, cm=cm)


def test(model_name, dataset_file, tif_sample, tif_real, result_name, model_directory, neighbors, vec,
         reduction_factor):
    if result_name != '':
        store_with_name = result_name
    else:
        store_with_name = model_name

    if model_directory is not '':
        model_dir = model_directory
    else:
        model_dir = 'storage'

    model_files = [f for f in listdir(model_dir) if f == model_name + '.pkl']
    model_files = natsorted(model_files, key=lambda y: y.lower())

    print("Loading models...")

    assert (len(model_files) >= 1)
    models = []
    for model_file in model_files:
        models.append(joblib.load(join(model_dir, model_file)))

    print("Models loaded from disk!")

    # evaluate loaded model on test data

    dataset_padding = neighbors

    if vec:
        center = int(np.floor(neighbors / 2))
        paths = calculate_feature_paths(center, reduction_factor)
    else:
        center = None
        paths = None

    feature_groups = int(neighbors / 2) + 1

    global dataset
    item_getter = itemgetter('bigdata')
    with np.load(dataset_file) as df:
        dataset = item_getter(df)

    # bigdata_clip = bigdata[:, :900, :900]
    dataset = dataset[:, :RasterParams.FNF_MAX_X, :RasterParams.FNF_MAX_Y]

    if paths is not None:
        feature_extraction_fn = get_patch_features_vector
        n_features = (paths.shape[0] + 1) * paths.shape[1]
    else:
        feature_extraction_fn = get_patch_features
        n_features = dataset.shape[0] * feature_groups

    half_padding = int(dataset_padding / 2)

    dataset = np.pad(dataset, [(0, 0), (half_padding, half_padding), (half_padding, half_padding)],
                     mode='constant')

    fnf_handler = GTiffHandler()
    # fnf_handler.readFile("storage/test_fullsize_train_pred.tif")
    fnf_handler.readFile(tif_sample)

    estimator_step_size = 50
    n_estimators = estimator_step_size * len(models)

    n_items_dataset = (RasterParams.FNF_MAX_X * RasterParams.FNF_MAX_Y)

    batch_size = NetworkParameters.BATCH_SIZE
    steps = int(n_items_dataset / batch_size) + 1

    predicted_test = np.zeros(shape=(n_items_dataset,), dtype=np.uint8)

    print('Starting testing phase...')
    start = time.time()

    values_votes = np.zeros(shape=(n_items_dataset, 2), dtype=np.float32)

    print('Test progress: ', end='')
    for i in range(steps):
        print('{0}/{1} - '.format(i + 1, steps), end='')
        start = (i) * batch_size
        end = (i + 1) * batch_size
        end = end if end < n_items_dataset else n_items_dataset

        batch = get_batch(start, batch_size, n_items_dataset, neighbors)

        X_partial_preprocessed_test_bag = np.zeros(shape=(end - start, n_features), dtype=np.float32)
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        X_partial_preprocessed_test_bag = np.array(
            pool.map(partial(feature_extraction_fn, feature_groups=feature_groups, neighbors=neighbors, paths=paths,
                             center=center), batch[:]))
        pool.close()
        pool.join()

        for model_i, model in enumerate(models):
            if model_i == 0:
                batch_votes = np.multiply(model.predict_proba(X_partial_preprocessed_test_bag),
                                          model.n_estimators)
            else:
                batch_votes = np.add(batch_votes,
                                     np.multiply(model.predict_proba(X_partial_preprocessed_test_bag),
                                                 model.n_estimators))

            # gc.collect()

        values_votes[start:end] = batch_votes

    predicted_test = np.divide(values_votes, n_estimators)
    X_partial_preprocessed_test_bag = None
    batch_votes = None
    values_votes = None
    gc.collect()
    predicted_test = np.argmax(predicted_test, axis=1)

    predict_mask = predicted_test.reshape(RasterParams.FNF_MAX_X, RasterParams.FNF_MAX_Y)
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
