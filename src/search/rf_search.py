import sys, getopt

sys.path.append("..")
import gc
import numpy as np
import _pickle as pkl

from operator import itemgetter
from os import listdir
from os.path import isfile, join
from search.rf_keras_batch_classifier import KerasBatchClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from config import DatasetConfig, RasterParams
from entities.parameter_search_status import ParameterSearchStatus


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Preparing for random forest parameter search')
    dataset_folder = None
    neighbors = None
    feature_reduction = None
    finish_earlier = False
    use_vector_features = False

    try:
        opts, args = getopt.getopt(argv, "hs:n:r:vc:k:f",
                                   ["dataset_folder=", "neighbors=", "continue_parameter=", "continue_kfold=",
                                    "feature_reduction="])
    except getopt.GetoptError:
        print('rf_search.py -s <dataset_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('rf_search.py -s <dataset_folder>')
            sys.exit()
        elif opt in ["-s", "--dataset_folder"]:
            dataset_folder = arg
        elif opt in ["-n", "--neighbors"]:
            neighbors = int(arg)
        elif opt in ["-r", "--feature_reduction"]:
            feature_reduction = int(arg)
        elif opt in ["-v", "--use_vector"]:
            use_vector_features = True
        elif opt in ["-c", "--continue_parameter"]:
            continue_parameter = arg
        elif opt in ["-k", "--continue_kfold"]:
            continue_kfold = int(arg)
        elif opt in ["-f", "--finish_earlier"]:
            finish_earlier = True
    print('Working with dataset folder %s' % dataset_folder)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    if feature_reduction is None:
        feature_reduction = 1

    # This is a very rudimentary system to continue the parameter search from different points
    search_status = ParameterSearchStatus()

    store_sub_path = '../storage/rf-search/vec/'
    if continue_parameter is not None:
        search_status.set_continue_parameter(continue_parameter)

        if continue_kfold is not None:
            search_status.set_continue_kfold(continue_kfold)
        else:
            search_status.set_continue_kfold(1)

        search_status.set_status(ParameterSearchStatus.CONTINUE_PREVIOUS_FAIL)

        with open(join(store_sub_path, 'search-status.pkl'), 'wb') as output:
            pkl.dump(search_status, output, -1)

    parameter_searcher(dataset_folder, neighbors, feature_reduction, use_vector_features, search_status, store_sub_path,
                       finish_earlier)

    sys.exit()


def parameter_searcher(dataset_folder, neighbors, feature_reduction, use_vector_features, search_status, store_sub_path,
                       finish_earlier):
    dataset, dataset_gt, dataset_idxs = prepare_generator_dataset(dataset_folder, neighbors)

    if use_vector_features:
        center = int(np.floor(neighbors / 2))
        paths = calculate_feature_paths(neighbors, center, feature_reduction)
    else:
        center = None
        paths = None

    # create model
    model = KerasBatchClassifier(build_fn=create_model, verbose=1)
    # define the grid search parameters
    n_ft = [3, 5, 8, 'sqrt']
    n_est = [500, 1000, 1500]
    min_smpl_spl = [2, 5, 10]
    min_smpl_leaf = [1, 2, 4]

    name_args = ['n_ft', 'n_est', 'min_smpl_spl', 'min_smpl_leaf']
    param_grid = dict(n_ft=n_ft, n_est=n_est, min_smpl_spl=min_smpl_spl, min_smpl_leaf=min_smpl_leaf)

    total_kfold = 3
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, error_score='raise', verbose=1,
                        cv=total_kfold)

    grid_result = grid.fit(dataset_idxs, dataset_idxs, dataset=dataset, dataset_gt=dataset_gt, name_args=name_args,
                           neighbors=neighbors, paths=paths, center=center, finish_earlier=finish_earlier,
                           store_sub_path=store_sub_path, total_kfold=total_kfold)

    search_status.set_status = ParameterSearchStatus.DONE

    with open(join(store_sub_path, 'search-status.pkl'), 'wb') as output:
        pkl.dump(search_status, output, pkl.HIGHEST_PROTOCOL)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    # print(model.summary())


def create_model(n_ft='auto', n_est=5000, crit='gini', min_smpl_spl=2, min_smpl_leaf=1):
    # print('---> Create model parameters: features %s, estimators %s, criterion %s, sample split %s, sample leaf %s' % (str(n_ft), str(n_est), str(crit), str(min_smpl_spl), str(min_smpl_leaf)))

    model = RandomForestClassifier(n_estimators=50, max_features=n_ft, criterion=crit, min_samples_split=min_smpl_spl,
                                   min_samples_leaf=min_smpl_leaf, warm_start=True, verbose=0, n_jobs=-1)
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


def path_values(paths, path_i, center, traslation_x, traslation_y):
    for i in range(1, center + 1):
        r = np.around(np.around((i * traslation_x) + center, decimals=1))
        h = np.around(np.around((i * traslation_y) + center, decimals=1))
        paths[path_i, i - 1] = [r, h]


def calculate_feature_paths(neighbors, center, step):
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


main(sys.argv[1:])
