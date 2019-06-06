import gdal
import sys, gc
import getopt
import redis
import time
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
from operator import itemgetter
from natsort import natsorted
from config import DatasetConfig, RasterParams
from dataset.threads.band_analyzer_thread import BandAnalyzerThread
from dataset.threads.sample_selector_thread_status import SampleSelectorThreadStatus

TACTIC_UPSAMPLE = 'upsample'
TACTIC_DOWNSAMPLE = 'downsample'

OPERATION_CREATE = 1000
OPERATION_ANALYZE = 2000
OPERATION_FULLANALYZE = 2500
OPERATION_MIX = 3000


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Preparing for balanced downsampler indexer by factor')
    gdal.UseExceptions()
    dataset_folder = None
    storage_folder = None
    tactic = TACTIC_DOWNSAMPLE
    operation = OPERATION_MIX
    beginning = 5
    ending = 100
    jump = 5
    iterations = 10
    partitions = 20

    try:
        opts, args = getopt.getopt(argv, "hs:d:t:cafmb:e:j:i:p:",
                                   ["dataset_folder=", "storage_folder=", "tactic=", "create", "analyze",
                                    "full_analyze", "mix", "begin=", "end=", "jump=", "iterations=", "partitions="])
    except getopt.GetoptError:
        print(
            'balanced_factor_indexer.py -s <dataset_folder> -d <storage_folder> -t {upsample/downsample} -m -b <beginning_percentage> -e <ending_percentage -j <jump_between_samples> -i <number_of_iterations> -p <partition_bins>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(
                'balanced_factor_indexer.py -s <dataset_folder> -d <storage_folder> -t {upsample/downsample} -m -b <beginning_percentage> -e <ending_percentage -j <jump_between_samples> -i <number_of_iterations> -p <partition_bins>')
            sys.exit()
        elif opt in ["-s", "--dataset_folder"]:
            dataset_folder = arg
        elif opt in ["-d", "--storage_folder"]:
            storage_folder = arg
        elif opt in ["-t", "--tactic"]:
            if arg == 'upsample':
                tactic = TACTIC_UPSAMPLE
            else:
                tactic = TACTIC_DOWNSAMPLE
        elif opt in ["-c", "--create"]:
            operation = OPERATION_CREATE
        elif opt in ["-a", "--analyze"]:
            operation = OPERATION_ANALYZE
        elif opt in ["-f", "--full_analyze"]:
            operation = OPERATION_FULLANALYZE
        elif opt in ["-m", "--mix"]:
            operation = OPERATION_MIX
        elif opt in ["-b", "--beginning"]:
            beginning = int(arg)
        elif opt in ["-e", "--ending"]:
            ending = int(arg)
        elif opt in ["-j", "--jump"]:
            jump = int(arg)
        elif opt in ["-i", "--iterations"]:
            iterations = int(arg)
        elif opt in ["-p", "--partitions"]:
            partitions = int(arg)

    print('Working with dataset folder %s' % dataset_folder)

    if operation == OPERATION_CREATE or operation == OPERATION_MIX:
        indexes_creator(dataset_folder, tactic, storage_folder, beginning, ending, jump, iterations)
    elif operation == OPERATION_ANALYZE or operation == OPERATION_MIX:
        dataset_analyzer(dataset_folder, storage_folder, beginning, ending, jump, partitions)
    elif operation == OPERATION_FULLANALYZE or operation == OPERATION_MIX:
        full_dataset_analyzer(dataset_folder, storage_folder, tactic, partitions)

    sys.exit()


def indexes_creator(dataset_folder, tactic, storage_folder, beginning, ending, jump, iterations):
    sample_rasters_folders = [f for f in listdir(dataset_folder) if not isfile(join(dataset_folder, f))]

    sample_rasters_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print('Folders to work with: ', sample_rasters_folders)

    print('Checking number of indexes...')
    cnt_idx_0 = 0
    cnt_idx_1 = 0
    for i, pck in enumerate(sample_rasters_folders):
        path_to_pck = join(dataset_folder, pck, 'idxs.npz')
        pck_data_idx_0 = pck_data_idx_1 = None
        item_getter = itemgetter('bigdata_idx_0', 'bigdata_idx_1')
        with np.load(path_to_pck) as df:
            pck_data_idx_0, pck_data_idx_1 = item_getter(df)

        cnt_idx_0 += pck_data_idx_0.shape[0]
        cnt_idx_1 += pck_data_idx_1.shape[0]

    forest_dominance = False if cnt_idx_0 > cnt_idx_1 else True

    class_total = 0
    if tactic == TACTIC_UPSAMPLE:
        if not forest_dominance:
            class_total = cnt_idx_0
        else:
            class_total = cnt_idx_1
    else:
        if not forest_dominance:
            class_total = cnt_idx_1
        else:
            class_total = cnt_idx_0

    # Retrieving all indexes from the different zones and putting them in memory

    bigdata_idx_0 = np.empty(shape=(cnt_idx_0 + 1, 3), dtype=np.uint16)
    bigdata_idx_1 = np.empty(shape=(cnt_idx_1 + 1, 3), dtype=np.uint16)

    print('Number of indexes for No Forest: %s' % (str(len(bigdata_idx_0))))
    print('Number of indexes for Forest: %s' % (str(len(bigdata_idx_1))))

    print('Copying and appending index values...')

    current_0_idx = 0
    current_1_idx = 0
    for i, pck in enumerate(sample_rasters_folders):
        path_to_pck = join(dataset_folder, pck, 'idxs.npz')

        pck_bigdata_idx_0 = pck_bigdata_idx_1 = None
        item_getter = itemgetter('bigdata_idx_0', 'bigdata_idx_1')
        with np.load(path_to_pck) as df:
            pck_bigdata_idx_0, pck_bigdata_idx_1 = item_getter(df)

        bigdata_idx_0[current_0_idx:current_0_idx + len(pck_bigdata_idx_0), 1:] = pck_bigdata_idx_0
        bigdata_idx_1[current_1_idx:current_1_idx + len(pck_bigdata_idx_1), 1:] = pck_bigdata_idx_1
        bigdata_idx_0[current_0_idx:current_0_idx + len(pck_bigdata_idx_0), 0] = i
        bigdata_idx_1[current_1_idx:current_1_idx + len(pck_bigdata_idx_1), 0] = i

        current_0_idx += len(pck_bigdata_idx_0)
        current_1_idx += len(pck_bigdata_idx_1)

    # Now we go through each percentage sampling

    for percentage in range(beginning, ending, jump):

        # Calculating sampling amount and determining if upsampling is needed
        upsample_required = False
        upsample_amount = 0

        class_percentage = None
        if tactic == TACTIC_UPSAMPLE:
            class_percentage = int(class_total * percentage / 100.0)
            if not forest_dominance:
                if class_percentage > cnt_idx_1:
                    upsample_required = True
                    upsample_amount = class_percentage - cnt_idx_1
            else:
                if class_percentage > cnt_idx_0:
                    upsample_required = True
                    upsample_amount = class_percentage - cnt_idx_0
        else:
            class_percentage = int(class_total * percentage / 100.0)

        folder_subfix = (TACTIC_UPSAMPLE if tactic == TACTIC_UPSAMPLE else TACTIC_DOWNSAMPLE) + '-' + str(
            int(percentage)) + 'p'

        analysis_path = join(storage_folder, 'train-balanced-' + folder_subfix)

        if not exists(analysis_path):
            makedirs(analysis_path)

        print('Performing initial shuffle of the full datasets')

        print('Shuffling No Forest indexes...')
        np.random.shuffle(bigdata_idx_0)
        print('Shuffling Forest indexes...')
        np.random.shuffle(bigdata_idx_1)

        p_bigdata_idx_0 = bigdata_idx_0.copy()
        p_bigdata_idx_1 = bigdata_idx_1.copy()

        if upsample_required:
            if not forest_dominance:
                if upsample_amount / cnt_idx_1 > 1:
                    repetitions = int(upsample_amount / cnt_idx_1)
                    print('Upsampling Forest indexes %s times' % (str(repetitions)))

                    if repetitions > 0:
                        p_bigdata_idx_1 = p_bigdata_idx_1.repeat(repetitions, axis=0)

                left_to_complete = upsample_amount % cnt_idx_1

                if left_to_complete > 0:
                    p_bigdata_idx_1 = np.append(p_bigdata_idx_1, p_bigdata_idx_1[:left_to_complete], axis=0)
            else:
                if upsample_amount / cnt_idx_0 > 1:
                    repetitions = int(upsample_amount / cnt_idx_0)
                    print('Upsampling No Forest indexes %s times' % (str(repetitions)))

                    if repetitions > 0:
                        p_bigdata_idx_0 = p_bigdata_idx_0.repeat(repetitions, axis=0)

                left_to_complete = upsample_amount % cnt_idx_0

                if left_to_complete > 0:
                    p_bigdata_idx_0 = np.append(p_bigdata_idx_0, p_bigdata_idx_0[:left_to_complete], axis=0)

        # For each iteration we shuffle, upsample if required and retrieve a percentage of the indexes
        for i in range(iterations):
            print('Performing shuffle before collecting iteration %d' % i)

            print('Shuffling No Forest indexes...')
            np.random.shuffle(p_bigdata_idx_0)
            print('Shuffling Forest indexes...')
            np.random.shuffle(p_bigdata_idx_1)

            final_idx_0 = p_bigdata_idx_0[:class_percentage]
            final_idx_1 = p_bigdata_idx_1[:class_percentage]

            analysis_idx_path = join(analysis_path,
                                     "{:02d}_{:02d}_{:}_samples_factor_idx.npz".format(percentage, i, tactic))
            print('Storing data: ' + analysis_idx_path)
            np.savez_compressed(analysis_idx_path, bigdata_idx_0=final_idx_0, bigdata_idx_1=final_idx_1)

    print('Done!')


def dataset_analyzer(dataset_folder, storage_folder, beginning, ending, jump, partitions):
    print('Retrieving datasets...')

    rasters_folders = [f for f in listdir(dataset_folder) if not isfile(join(dataset_folder, f))]

    rasters_folders = natsorted(rasters_folders, key=lambda y: y.lower())

    bigdata = np.zeros(shape=(len(rasters_folders), DatasetConfig.DATASET_LST_BANDS_USED, RasterParams.SRTM_MAX_X,
                              RasterParams.SRTM_MAX_Y), dtype=np.float32)
    bigdata_gt = np.zeros(shape=(len(rasters_folders), RasterParams.FNF_MAX_X, RasterParams.FNF_MAX_Y), dtype=np.uint8)

    for i, pck in enumerate(rasters_folders):
        path_to_pck = join(dataset_folder, pck, 'dataset.npz')

        print('Loading dataset folder ', pck)

        pck_bigdata = None
        item_getter = itemgetter('bigdata')
        with np.load(path_to_pck) as df:
            pck_bigdata = item_getter(df)

        bigdata[i] = pck_bigdata

        pck_bigdata_gt = None
        item_getter = itemgetter('bigdata_gt')
        with np.load(path_to_pck) as df:
            pck_bigdata_gt = item_getter(df)

        bigdata_gt[i] = pck_bigdata_gt

        del pck_bigdata
        del pck_bigdata_gt

        gc.collect()

    print('Procesing percentage sampled index files...\n')

    partition_range = 2.0 / partitions

    bigdata = np.divide(np.add(bigdata, 1.0), partition_range)
    gc.collect()
    bigdata = bigdata.astype(np.uint32)
    for percentage in range(beginning, ending, jump):
        print('Starting with percentage %d' % percentage)
        percentage_idxs_folder = [d for d in listdir(storage_folder) if
                                  not isfile(join(storage_folder, d)) and str(d).endswith("{:02d}p".format(percentage))]

        if len(percentage_idxs_folder) != 0:
            percentage_idxs_files = [f for f in listdir(join(storage_folder, percentage_idxs_folder[0])) if
                                     isfile(join(storage_folder, percentage_idxs_folder[0], f)) and str(f).endswith(
                                         'factor_idx.npz')]

            percentage_idxs_files = natsorted(percentage_idxs_files, key=lambda y: y.lower())

            item_counter_0 = np.zeros(
                shape=(len(percentage_idxs_files), DatasetConfig.DATASET_LST_BANDS_USED, partitions + 1),
                dtype=np.uint32)
            item_counter_1 = np.zeros(
                shape=(len(percentage_idxs_files), DatasetConfig.DATASET_LST_BANDS_USED, partitions + 1),
                dtype=np.uint32)

            for i, idx_file in enumerate(percentage_idxs_files):
                path_to_idx = join(storage_folder, percentage_idxs_folder[0], idx_file)
                print('Processing idx file %s' % path_to_idx)

                iter_bigdata_idx_0 = iter_bigdata_idx_1 = None
                item_getter = itemgetter('bigdata_idx_0', 'bigdata_idx_1')
                with np.load(path_to_idx) as df:
                    iter_bigdata_idx_0, iter_bigdata_idx_1 = item_getter(df)

                redises = []
                threads = list()
                band_analyzers = []

                for band in range(DatasetConfig.DATASET_LST_BANDS_USED):
                    redis_db = redis.Redis(db=band)
                    redis_db.delete('item_counter_0', 'item_counter_1', 'status')
                    redises.append(redis_db)

                    band_analyzer = BandAnalyzerThread(band, redis_db, bigdata, iter_bigdata_idx_0, iter_bigdata_idx_1,
                                                       partitions, band)
                    band_analyzers.append(band_analyzer)

                    t = band_analyzer
                    threads.append(t)
                    t.start()

                all_thread_processed = False
                thrds_processed = [False for t_i in range(len(threads))]

                while not all_thread_processed:
                    # progress_bar(redises, stdscr)

                    for thrd in range(len(threads)):
                        if redises[thrd].get('status').decode('utf-8') == SampleSelectorThreadStatus.STATUS_DONE:
                            if not thrds_processed[thrd]:
                                item_counter_0[i][thrd][:] = redises[thrd].get('item_counter_0').decode('utf-8').split(
                                    ';')
                                item_counter_1[i][thrd][:] = redises[thrd].get('item_counter_1').decode('utf-8').split(
                                    ';')
                                thrds_processed[thrd] = True

                    all_thread_processed = True
                    for elem in thrds_processed:
                        if not elem:
                            all_thread_processed = False

                    if not all_thread_processed:
                        time.sleep(1)

            n_item_counter_0 = np.zeros(
                shape=(len(percentage_idxs_files), DatasetConfig.DATASET_LST_BANDS_USED, partitions), dtype=np.uint32)
            n_item_counter_1 = np.zeros(
                shape=(len(percentage_idxs_files), DatasetConfig.DATASET_LST_BANDS_USED, partitions), dtype=np.uint32)

            n_item_counter_0[:, :, 0:partitions - 1] = item_counter_0[:, :, 0:partitions - 1]
            n_item_counter_0[:, :, partitions - 1] = item_counter_0[:, :, partitions - 1] + item_counter_0[:, :,
                                                                                            partitions]

            del (item_counter_0)

            n_item_counter_1[:, :, 0:partitions - 1] = item_counter_1[:, :, 0:partitions - 1]
            n_item_counter_1[:, :, partitions - 1] = item_counter_1[:, :, partitions - 1] + item_counter_1[:, :,
                                                                                            partitions]

            del (item_counter_1)

            analysis_cntr_path = join(storage_folder, percentage_idxs_folder[0],
                                      "{:02d}_{:02d}_histogram_info.npz".format(percentage, partitions))
            print('Storing data: ' + analysis_cntr_path)
            np.savez_compressed(analysis_cntr_path, item_counter_0=n_item_counter_0, item_counter_1=n_item_counter_1)

    print('Done!')


def full_dataset_analyzer(dataset_folder, storage_folder, tactic, partitions):
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

    print('Checking number of indexes...')
    cnt_idx_0 = 0
    cnt_idx_1 = 0
    for i, pck in enumerate(rasters_folders):
        path_to_pck = join(dataset_folder, pck, 'idxs.npz')
        pck_data_idx_0 = pck_data_idx_1 = None
        item_getter = itemgetter('bigdata_idx_0', 'bigdata_idx_1')
        with np.load(path_to_pck) as df:
            pck_data_idx_0, pck_data_idx_1 = item_getter(df)

        cnt_idx_0 += pck_data_idx_0.shape[0]
        cnt_idx_1 += pck_data_idx_1.shape[0]

    forest_dominance = False if cnt_idx_0 > cnt_idx_1 else True

    class_total = 0
    if tactic == TACTIC_UPSAMPLE:
        if not forest_dominance:
            class_total = cnt_idx_0
        else:
            class_total = cnt_idx_1
    else:
        if not forest_dominance:
            class_total = cnt_idx_1
        else:
            class_total = cnt_idx_0

    # Retrieving all indexes from the different zones and putting them in memory

    bigdata_idx_0 = np.empty(shape=(cnt_idx_0 + 1, 3), dtype=np.uint16)
    bigdata_idx_1 = np.empty(shape=(cnt_idx_1 + 1, 3), dtype=np.uint16)

    print('Number of indexes for No Forest: %s' % (str(len(bigdata_idx_0))))
    print('Number of indexes for Forest: %s' % (str(len(bigdata_idx_1))))

    print('Copying and appending index values...')

    current_0_idx = 0
    current_1_idx = 0
    for i, pck in enumerate(rasters_folders):
        path_to_pck = join(dataset_folder, pck, 'idxs.npz')

        pck_bigdata_idx_0 = pck_bigdata_idx_1 = None
        item_getter = itemgetter('bigdata_idx_0', 'bigdata_idx_1')
        with np.load(path_to_pck) as df:
            pck_bigdata_idx_0, pck_bigdata_idx_1 = item_getter(df)

        bigdata_idx_0[current_0_idx:current_0_idx + len(pck_bigdata_idx_0), 1:] = pck_bigdata_idx_0
        bigdata_idx_1[current_1_idx:current_1_idx + len(pck_bigdata_idx_1), 1:] = pck_bigdata_idx_1
        bigdata_idx_0[current_0_idx:current_0_idx + len(pck_bigdata_idx_0), 0] = i
        bigdata_idx_1[current_1_idx:current_1_idx + len(pck_bigdata_idx_1), 0] = i

        current_0_idx += len(pck_bigdata_idx_0)
        current_1_idx += len(pck_bigdata_idx_1)

    upsample_required = False
    if tactic == TACTIC_UPSAMPLE:
        if not forest_dominance:
            if class_total > cnt_idx_1:
                upsample_required = True
                upsample_amount = class_total - cnt_idx_1
        else:
            if class_total > cnt_idx_0:
                upsample_required = True
                upsample_amount = class_total - cnt_idx_0

    if upsample_required:
        if not forest_dominance:
            if upsample_amount / cnt_idx_1 > 1:
                repetitions = int(upsample_amount / cnt_idx_1)
                print('Upsampling Forest indexes %s times' % (str(repetitions)))

                if repetitions > 0:
                    bigdata_idx_1 = bigdata_idx_1.repeat(repetitions, axis=0)

            left_to_complete = upsample_amount % cnt_idx_1

            if left_to_complete > 0:
                bigdata_idx_1 = np.append(bigdata_idx_1, bigdata_idx_1[:left_to_complete], axis=0)
        else:
            if upsample_amount / cnt_idx_0 > 1:
                repetitions = int(upsample_amount / cnt_idx_0)
                print('Upsampling No Forest indexes %s times' % (str(repetitions)))

                if repetitions > 0:
                    bigdata_idx_0 = bigdata_idx_0.repeat(repetitions, axis=0)

            left_to_complete = upsample_amount % cnt_idx_0

            if left_to_complete > 0:
                bigdata_idx_0 = np.append(bigdata_idx_0, bigdata_idx_0[:left_to_complete], axis=0)

    print('Procesing full dataset distribution sampled index files...\n')

    partition_range = 2.0 / partitions

    bigdata = np.divide(np.add(bigdata, 1.0), partition_range)
    gc.collect()
    bigdata = bigdata.astype(np.uint32)

    item_counter_0 = np.zeros(shape=(DatasetConfig.DATASET_LST_BANDS_USED, partitions + 1), dtype=np.uint32)
    item_counter_1 = np.zeros(shape=(DatasetConfig.DATASET_LST_BANDS_USED, partitions + 1), dtype=np.uint32)

    redises = []
    threads = list()
    band_analyzers = []

    for band in range(DatasetConfig.DATASET_LST_BANDS_USED):
        redis_db = redis.Redis(db=band)
        redis_db.delete('item_counter_0', 'item_counter_1', 'status')
        redises.append(redis_db)

        band_analyzer = BandAnalyzerThread(band, redis_db, bigdata, bigdata_idx_0, bigdata_idx_1,
                                           partitions, band)
        band_analyzers.append(band_analyzer)

        t = band_analyzer
        threads.append(t)
        t.start()

    all_thread_processed = False
    thrds_processed = [False for t_i in range(len(threads))]

    while not all_thread_processed:
        # progress_bar(redises, stdscr)

        for thrd in range(len(threads)):
            if redises[thrd].get('status').decode('utf-8') == SampleSelectorThreadStatus.STATUS_DONE:
                if not thrds_processed[thrd]:
                    item_counter_0[thrd][:] = redises[thrd].get('item_counter_0').decode('utf-8').split(';')
                    item_counter_1[thrd][:] = redises[thrd].get('item_counter_1').decode('utf-8').split(';')
                    thrds_processed[thrd] = True

        all_thread_processed = True
        for elem in thrds_processed:
            if not elem:
                all_thread_processed = False

        if not all_thread_processed:
            time.sleep(1)

    n_item_counter_0 = np.zeros(shape=(DatasetConfig.DATASET_LST_BANDS_USED, partitions), dtype=np.uint32)
    n_item_counter_1 = np.zeros(shape=(DatasetConfig.DATASET_LST_BANDS_USED, partitions), dtype=np.uint32)

    n_item_counter_0[:, :, 0:partitions - 1] = item_counter_0[:, :, 0:partitions - 1]
    n_item_counter_0[:, :, partitions - 1] = item_counter_0[:, :, partitions - 1] + item_counter_0[:, :, partitions]

    del item_counter_0

    n_item_counter_1[:, :, 0:partitions - 1] = item_counter_1[:, :, 0:partitions - 1]
    n_item_counter_1[:, :, partitions - 1] = item_counter_1[:, :, partitions - 1] + item_counter_1[:, :, partitions]

    del item_counter_1

    analysis_cntr_path = join(storage_folder, "full_{:02d}_histogram_info.npz".format(partitions))
    print('Storing data: ' + analysis_cntr_path)
    np.savez_compressed(analysis_cntr_path, item_counter_0=n_item_counter_0, item_counter_1=n_item_counter_1)


main(sys.argv[1:])
