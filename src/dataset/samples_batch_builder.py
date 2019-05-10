import sys
import getopt
import gdal
import numpy as np
from operator import itemgetter
from os import listdir
from os.path import isfile, join


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Preparing for sample batch creation')
    gdal.UseExceptions()
    dataset_folder = ''

    try:
        opts, args = getopt.getopt(argv, "hs:", ["dataset_folder="])
    except getopt.GetoptError:
        print('sample_batch_builder.py -s <dataset_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('sample_batch_builder.py -s <dataset_folder>')
            sys.exit()
        elif opt in ["-s", "--dataset_folder"]:
            dataset_folder = arg

    print('Working with dataset folder %s' % dataset_folder)

    batch_sample_generator(dataset_folder)

    sys.exit()


def batch_sample_generator(dataset_folder):
    sample_rasters_folders = [f for f in listdir(dataset_folder) if not isfile(join(dataset_folder, f))]

    sample_rasters_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print('Folders to work with: ', sample_rasters_folders)

    print('Checking number of indexes...')
    cnt_idx_0 = 0
    cnt_idx_1 = 0
    for i, pck in enumerate(sample_rasters_folders):
        path_to_pck = join(dataset_folder, pck, 'dataset.npz')

        pck_metadata_gt = None
        item_getter = itemgetter('pckmetadata_gt')
        with np.load(path_to_pck) as df:
            pck_metadata_gt = item_getter(df)

        cnt_idx_0 += pck_metadata_gt[1]
        cnt_idx_1 += pck_metadata_gt[2]


    bigdata_idx_0 = np.empty(shape=(cnt_idx_0+1,3), dtype=np.uint16)
    bigdata_idx_1 = np.empty(shape=(cnt_idx_1+1,3), dtype=np.uint16)

    print('Number of indexes for No Forest: %s' % (str(len(bigdata_idx_0))))
    print('Number of indexes for Forest: %s' % (str(len(bigdata_idx_1))))

    print('Copying and appending index values...')

    current_0_idx = 0
    #current_1_idx = 1
    current_1_idx = 0
    for i, pck in enumerate(sample_rasters_folders):
        path_to_pck = join(dataset_folder, pck, 'dataset.npz')

        pck_bigdata_idx_0 = pck_bigdata_idx_1 = None
        item_getter = itemgetter('bigdata_idx_0', 'bigdata_idx_1')
        with np.load(path_to_pck) as df:
            pck_bigdata_idx_0, pck_bigdata_idx_1 = item_getter(df)

        bigdata_idx_0[current_0_idx:current_0_idx+len(pck_bigdata_idx_0),1:] = pck_bigdata_idx_0
        bigdata_idx_1[current_1_idx:current_1_idx+len(pck_bigdata_idx_1),1:] = pck_bigdata_idx_1
        bigdata_idx_0[current_0_idx:current_0_idx+len(pck_bigdata_idx_0),0] = i
        bigdata_idx_1[current_1_idx:current_1_idx+len(pck_bigdata_idx_1),0] = i

        current_0_idx += len(pck_bigdata_idx_0)
        current_1_idx += len(pck_bigdata_idx_1)


    if len(bigdata_idx_0) > len(bigdata_idx_1):
        repetitions = int(len(bigdata_idx_0) / len(bigdata_idx_1))
        print('Upsampling Forest indexes %s times' % (str(repetitions)))

        if repetitions > 0:
            bigdata_idx_1 = bigdata_idx_1.repeat(repetitions, axis=0)

        left_to_complete = len(bigdata_idx_0) - len(bigdata_idx_1)

        if left_to_complete > 0:
            bigdata_idx_1 = np.append(bigdata_idx_1, bigdata_idx_1[:left_to_complete], axis=0)
    elif len(bigdata_idx_1) > len(bigdata_idx_0):
        repetitions = int(len(bigdata_idx_1) / len(bigdata_idx_0))

        print('Upsampling No Forest indexes %s times' % (str(repetitions)))

        if repetitions > 0:
            bigdata_idx_0 = bigdata_idx_0.repeat(repetitions, axis=0)

        left_to_complete = len(bigdata_idx_1) - len(bigdata_idx_0)

        if left_to_complete > 0:
            bigdata_idx_0 = np.append(bigdata_idx_0, bigdata_idx_0[:left_to_complete], axis=0)

    print('Shuffling No Forest indexes...')
    np.random.shuffle(bigdata_idx_0)
    print('Shuffling Forest indexes...')
    np.random.shuffle(bigdata_idx_1)

    print('Storing data...')
    dataset_path = join(dataset_folder, 'samples_shuffled_idx.npz')
    np.savez_compressed(dataset_path, bigdata_idx_0=bigdata_idx_0, bigdata_idx_1=bigdata_idx_1)

    print('Done!')

main(sys.argv[1:])