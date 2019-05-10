import sys
import gdal
import getopt
import curses
import redis
import time
import numpy as np
from os import listdir
from os.path import isfile, join
from operator import itemgetter
from config import DatasetConfig
from raster_rw import GTiffHandler
from dataset.threads.sample_selector_thread import SampleSelectorThread
from dataset.threads.sample_selector_thread_status import SampleSelectorThreadStatus


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Preparing for sample factor upsampler')
    gdal.UseExceptions()
    dataset_folder = ''

    try:
        opts, args = getopt.getopt(argv, "hs:", ["dataset_folder="])
    except getopt.GetoptError:
        print('factor_upsampler_indexer.py -s <dataset_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('factor_upsampler_indexer.py -s <dataset_folder>')
            sys.exit()
        elif opt in ["-s", "--dataset_folder"]:
            dataset_folder = arg

    print('Working with dataset folder %s' % dataset_folder)

    factor_upsampler_indexer(dataset_folder)

    sys.exit()


def factor_upsampler_indexer(dataset_folder):
    sample_rasters_folders = [f for f in listdir(dataset_folder) if not isfile(join(dataset_folder, f))]

    sample_rasters_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print('Folders to work with: ', sample_rasters_folders)

    print('Checking number of indexes...')
    cnt_idx_0 = 0
    cnt_idx_1 = 0
    for i, pck in enumerate(sample_rasters_folders):
        path_to_pck = join(dataset_folder, pck)
        raster_files = [f for f in listdir(path_to_pck) if isfile(join(path_to_pck, f)) and 'FNF' in f[0:4]]
        fnf_handler = GTiffHandler()
        fnf_handler.readFile(join(path_to_pck, raster_files[0]))

        fnf_np_array = np.array(fnf_handler.src_Z)

        unique, counts = np.unique(fnf_np_array, return_counts=True)

        cnt_idx_0 += counts[0]
        cnt_idx_1 += counts[1]

        fnf_handler.closeFile()

    upsample_forest = False
    if cnt_idx_0 > cnt_idx_1:
        upsample_forest = True

    class_total = None
    upsample_rate = 1
    upsample_limit = int((cnt_idx_0 / cnt_idx_1)) if upsample_forest else int((cnt_idx_1 / cnt_idx_0))
    folder_subfix = '100p-' + str(upsample_limit) + 't'
    if DatasetConfig.UPSAMPLE_PERCERTAGE > 0:
        upsample_rate = int(upsample_limit * (DatasetConfig.UPSAMPLE_PERCERTAGE / 100.0))
        folder_subfix = str(int(DatasetConfig.UPSAMPLE_PERCERTAGE)) + 'p-' + str(upsample_rate) + 't'
    elif DatasetConfig.UPSAMPLE_RATE > 1:
        upsample_rate = DatasetConfig.UPSAMPLE_RATE
        percentage = (int(upsample_limit / upsample_rate * 100))
        folder_subfix = str(percentage) + 'p-' + str(upsample_rate) + 't'

    class_total = upsample_rate * cnt_idx_1 if upsample_forest else upsample_rate * cnt_idx_0

    class_total_per_process = None

    if upsample_rate > 1:
        class_total_per_process = [int(class_total / len(sample_rasters_folders)) for i in
                                   range(0, len(sample_rasters_folders))]
        class_total_per_process[-1] = class_total_per_process[-1] + (class_total % len(sample_rasters_folders))
    else:
        class_total_per_process = [-1 for i in range(0, len(sample_rasters_folders))]

    processed_path = '../data/processed/train-factor-' + folder_subfix + '/'

    redises = []
    threads = list()
    for i, pck in enumerate(sample_rasters_folders):
        path_to_pck = join(dataset_folder, pck)

        redis_db = redis.Redis(db=i)
        redis_db.delete('processed', 'total', 'status')
        redises.append(redis_db)

        spl_selector = SampleSelectorThread(i, path_to_pck, redis_db, class_total_per_process[i], processed_path)

        t = spl_selector
        threads.append(t)
        t.start()

    print('All folders were loaded')

    try:
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
    except curses.error as e:
        print('Expected curses error... just ignore it')

    all_thread_processed = False
    thrds_processed = [False for t_i in range(len(threads))]

    while not all_thread_processed:
        progress_bar(redises, stdscr)

        for thrd in range(len(threads)):
            if redises[thrd].get('status').decode('utf-8') == SampleSelectorThreadStatus.STATUS_DONE:
                if not thrds_processed[thrd]:
                    thrds_processed[thrd] = True

        all_thread_processed = True
        for elem in thrds_processed:
            if not elem:
                all_thread_processed = False

        if not all_thread_processed:
            time.sleep(1)

    try:
        curses.echo()
        curses.nocbreak()
        curses.endwin()
    except curses.error as e:
        print('Expected curses error... just ignore it')
    finally:
        pass

    bigdata_idx_0 = None
    bigdata_idx_1 = None
    if upsample_forest:
        bigdata_idx_0 = np.empty(shape=(class_total, 3), dtype=np.uint16)
        bigdata_idx_1 = np.empty(shape=(cnt_idx_1, 3), dtype=np.uint16)
    else:
        bigdata_idx_0 = np.empty(shape=(cnt_idx_0, 3), dtype=np.uint16)
        bigdata_idx_1 = np.empty(shape=(class_total, 3), dtype=np.uint16)

    print('Number of indexes for No Forest: %s' % (str(bigdata_idx_0.shape[0])))
    print('Number of indexes for Forest: %s' % (str(bigdata_idx_1.shape[0])))

    print('Copying and appending index values...')


    current_0_idx = 0
    # current_1_idx = 1
    current_1_idx = 0
    for i, pck in enumerate(sample_rasters_folders):
        path_to_pck = join(processed_path, pck, 'idxs.npz')

        pck_bigdata_idx_0 = pck_bigdata_idx_1 = None
        item_getter = itemgetter('bigdata_idx_0', 'bigdata_idx_1')
        with np.load(path_to_pck) as df:
            pck_bigdata_idx_0, pck_bigdata_idx_1 = item_getter(df)

        bigdata_idx_0[current_0_idx:current_0_idx + pck_bigdata_idx_0.shape[0], 1:] = pck_bigdata_idx_0
        bigdata_idx_1[current_1_idx:current_1_idx + pck_bigdata_idx_1.shape[0], 1:] = pck_bigdata_idx_1
        bigdata_idx_0[current_0_idx:current_0_idx + pck_bigdata_idx_0.shape[0], 0] = i
        bigdata_idx_1[current_1_idx:current_1_idx + pck_bigdata_idx_1.shape[0], 0] = i

        current_0_idx += pck_bigdata_idx_0.shape[0]
        current_1_idx += pck_bigdata_idx_1.shape[0]

    repetitions = upsample_rate
    if upsample_forest:
        print('Upsampling Forest indexes %s times' % (str(repetitions)))

        if repetitions > 0:
            bigdata_idx_1 = bigdata_idx_1.repeat(repetitions, axis=0)

        left_to_complete = len(bigdata_idx_0) - len(bigdata_idx_1)

        print('Left to complete: ', left_to_complete)

        if left_to_complete > 0:
            bigdata_idx_1 = np.append(bigdata_idx_1, bigdata_idx_1[:left_to_complete], axis=0)
    else:
        print('Upsampling No Forest indexes %s times' % (str(repetitions)))

        if repetitions > 0:
            bigdata_idx_0 = bigdata_idx_0.repeat(repetitions, axis=0)

        left_to_complete = len(bigdata_idx_1) - len(bigdata_idx_0)

        print('Left to complete: ', left_to_complete)

        if left_to_complete > 0:
            bigdata_idx_0 = np.append(bigdata_idx_0, bigdata_idx_0[:left_to_complete], axis=0)

    print('Shuffling No Forest indexes...')
    np.random.shuffle(bigdata_idx_0)
    print('Shuffling Forest indexes...')
    np.random.shuffle(bigdata_idx_1)

    print('Storing data...')
    dataset_path = join(processed_path, 'samples_shuffled_factor_idx.npz')
    np.savez_compressed(dataset_path, bigdata_idx_0=bigdata_idx_0, bigdata_idx_1=bigdata_idx_1)

    print('Done!')

def progress_bar(working_redis_db, stdscr):
    try:
        for working_t_i, working_t in enumerate(working_redis_db):

            t_current_value = working_t.get("processed")
            t_total = working_t.get("total")
            t_status = working_t.get("status").decode('utf-8')

            current_value = int(t_current_value) if t_current_value is not None else None
            total = int(t_total) if t_total is not None else None
            status = str(t_status) if t_status is not None else None

            if total is not None and current_value is not None and current_value != total:
                increments = 50
                percentual = ((current_value / total) * 100)
                i = int(percentual // (100 / increments))
                text = "\rData folder {0} - {1} points [{2: <{3}}] {4}%".format(working_t_i + 1, total, '=' * i, increments,
                                                                                "{0:.2f}".format(percentual))
                stdscr.addstr(working_t_i, 0, text)
                # print(text, end="\n" if percentual == 100 else "")
            else:
                text = "\rData folder {0} - {1} points - {2}".format(working_t_i + 1, total if total is not None else 0, status)
                stdscr.addstr(working_t_i, 0, text)
        stdscr.refresh()
    except curses.error as e:
        pass

main(sys.argv[1:])
