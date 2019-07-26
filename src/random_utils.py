import gc
import getopt
import sys
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from os import listdir
from os.path import isfile, join
from operator import itemgetter

from config import DatasetConfig, RasterParams


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Preparing for... something')
    dataset_folder = None
    idx_filepath = None
    npz_filepath = None

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #session = tf.Session(config=config)

    try:
        opts, args = getopt.getopt(argv, "hs:i:n:", ["dataset_folder=","idx_filepath=","npz_filepath="])
    except getopt.GetoptError:
        print('random_utils.py -s <dataset_folder> -i <idx_filepath> -n <npz_filepath>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('random_utils.py -s <dataset_folder> -i <idx_filepath> -n <npz_filepath>')
            sys.exit()
        elif opt in ["-s", "--dataset_folder"]:
            dataset_folder = arg
        elif opt in ["-i", "--idx_filepath"]:
            idx_filepath = arg
        elif opt in ["-n", "--npz_filepath"]:
            npz_filepath = arg

    print('Working with NPZ file %s' % npz_filepath)
    print('Working with IDX file %s' % idx_filepath)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    plot_full(dataset_folder, idx_filepath, npz_filepath)

    sys.exit()


def plot_full(dataset_folder, idx_filepath, npz_filepath):
    print('Retrieving datasets...')

    rasters_folders = [f for f in listdir(dataset_folder) if not isfile(join(dataset_folder, f))]

    rasters_folders = natsorted(rasters_folders, key=lambda y: y.lower())

    bigdata = np.zeros(shape=(len(rasters_folders), DatasetConfig.DATASET_LST_BANDS_USED, RasterParams.SRTM_MAX_X,
                              RasterParams.SRTM_MAX_Y), dtype=np.float32)

    for i, pck in enumerate(rasters_folders):
        path_to_pck = join(dataset_folder, pck, 'dataset.npz')

        print('Loading dataset folder ', pck)

        pck_bigdata = None
        item_getter = itemgetter('bigdata')
        with np.load(path_to_pck) as df:
            pck_bigdata = item_getter(df)

        bigdata[i] = pck_bigdata

        del pck_bigdata

        gc.collect()

    bigdata_idx_0 = None
    bigdata_idx_1 = None

    item_getter = itemgetter('bigdata_idx_0', 'bigdata_idx_1')
    with np.load(idx_filepath) as df:
        bigdata_idx_0, bigdata_idx_1 = item_getter(df)

    values_0 = None
    values_1 = None
    edges_0 = None
    edges_1 = None
    lower_0 = None
    lower_1 = None
    upper_0 = None
    upper_1 = None
    lower_outliers_0 = None
    lower_outliers_1 = None
    upper_outliers_0 = None
    upper_outliers_1 = None
    percentages_0 = None
    percentages_1 = None


    item_getter = itemgetter('values_0', 'values_1', 'edges_0', 'edges_1', 'lower_0', 'lower_1', 'upper_0', 'upper_1',
    'lower_outliers_0', 'lower_outliers_1', 'upper_outliers_0', 'upper_outliers_1', 'percentages_0', 'percentages_1')
    with np.load(npz_filepath) as df:
        values_0, values_1, edges_0, edges_1, lower_0, lower_1, upper_0, upper_1, lower_outliers_0, lower_outliers_1, upper_outliers_0, upper_outliers_1, percentages_0, percentages_1 = item_getter(df)

    items = np.zeros(shape=(bigdata_idx_0.shape[0]), dtype=np.float32)

    for i, item in enumerate(bigdata_idx_0):
        items[i] = bigdata[item[0]][1][item[1]][item[2]]

    bounds = calculate_bounds(items, z_thresh=4.5)
    outlier_aware_hist(items, 'fd', *bounds, data_min=-1, data_max=1, title='Class Forest - Layer ' + str(5))

    print('Lets check until here')

    #plt.hist(edges_0[0][:-1], bins=edges_0[0], weights=np.multiply(np.divide(values_0[0], np.sum(values_0[0])), 100.0))
    plt.hist(edges_0[1][:-1], bins=edges_0[1], weights=values_0[1])

    plt.show()

    print('was it plotted???')

def outlier_aware_hist(data, sel_bins, lower=None, upper=None, data_min=None, data_max=None, title=''):
    if not lower or lower < data.min():
        lower = data.min() if data_min is None else data_min
        lower_outliers = False
    else:
        lower_outliers = True

    if not upper or upper > data.max():
        upper = data.max() if data_max is None else data_max
        upper_outliers = False
    else:
        upper_outliers = True

    n, bins, patches = plt.hist(data, range=(lower, upper), bins=sel_bins)

    if lower_outliers:
        n_lower_outliers = (data < lower).sum()
        #patches[0].set_height(patches[0].get_height() + n_lower_outliers)
        patches[0].set_facecolor('c')
        patches[0].set_label(
            'Lower outliers: ({:.2f}, {:.2f})'.format(-1, lower))

    if upper_outliers:
        n_upper_outliers = (data > upper).sum()
        #patches[-1].set_height(patches[-1].get_height() + n_upper_outliers)
        patches[-1].set_facecolor('m')
        patches[-1].set_label(
            'Upper outliers: ({:.2f}, {:.2f})'.format(upper, 1))

    if lower_outliers or upper_outliers:
        plt.legend()

    plt.xlabel("Bins")
    plt.ylabel("Frequency")
    plt.title('Histogram: ' + title)
    plt.show()

def mad(data):
    median = np.median(data)
    diff = np.abs(data - median)
    mad = np.median(diff)
    return mad

def calculate_bounds(data, z_thresh=3.5):
    MAD = mad(data)
    median = np.median(data)
    const = z_thresh * MAD / 0.6745
    return (median - const, median + const)

main(sys.argv[1:])