
import gdal
import sys
import getopt
import redis
import curses
import time
import numpy as np
from os import listdir
from os.path import isfile, join
from operator import itemgetter
from raster_rw import GTiffHandler
from config import DatasetConfig
from dataset.threads.sample_extractor_thread import SampleExtractorThread
from dataset.threads.sample_selector_thread_status import SampleSelectorThreadStatus

TACTIC_UPSAMPLE = 'upsample'
TACTIC_DOWNSAMPLE = 'downsample'

def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Preparing for balanced downsampler indexer by factor')
    gdal.UseExceptions()
    dataset_folder = ''
    tactic = TACTIC_DOWNSAMPLE
    test_only = False

    try:
        opts, args = getopt.getopt(argv, "hs:t:x", ["dataset_folder=", "tactic=", "test"])
    except getopt.GetoptError:
        print('balanced_factor_indexer.py -s <dataset_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('balanced_factor_indexer.py -s <dataset_folder>')
            sys.exit()
        elif opt in ["-s", "--dataset_folder"]:
            dataset_folder = arg
        elif opt in ["-t", "--tactic"]:
            if arg == 'upsample':
                tactic = TACTIC_UPSAMPLE
            else:
                tactic = TACTIC_DOWNSAMPLE
        elif opt in ["-x", "--test"]:
            test_only = True

    print('Working with dataset folder %s' % dataset_folder)

    balanced_downsample_indexer(dataset_folder, tactic, test_only)

    sys.exit()


def balanced_downsample_indexer(dataset_folder, tactic, test_only):
    sample_rasters_folders = [f for f in listdir(dataset_folder) if not isfile(join(dataset_folder, f))]

    sample_rasters_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print('Folders to work with: ', sample_rasters_folders)

    if not test_only:

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

        upsample_required = False
        upsample_amount = 0

        class_percentage = None
        if tactic == TACTIC_UPSAMPLE:
            class_percentage = int(class_total * DatasetConfig.BALANCED_PERCENTAGE / 100.0)
            if not forest_dominance:
                if class_percentage > cnt_idx_1:
                    upsample_required = True
                    upsample_amount = class_percentage - cnt_idx_1
            else:
                if class_percentage > cnt_idx_0:
                    upsample_required = True
                    upsample_amount = class_percentage - cnt_idx_0
        else:
            class_percentage = int(class_total * DatasetConfig.BALANCED_PERCENTAGE / 100.0)

        folder_subfix = (TACTIC_UPSAMPLE if tactic == TACTIC_UPSAMPLE else TACTIC_DOWNSAMPLE) + '-' + str(int(DatasetConfig.BALANCED_PERCENTAGE)) + 'p'

        processed_path = '../data/processed/train-balanced-' + folder_subfix
    else:
        processed_path = '../data/processed/test'

    redises = []
    threads = list()
    sample_extractors = []
    for i, pck in enumerate(sample_rasters_folders):
        path_to_pck = join(dataset_folder, pck)

        redis_db = redis.Redis(db=i)
        redis_db.delete('processed', 'total', 'status')
        redises.append(redis_db)

        spl_selector = SampleExtractorThread(i, path_to_pck, redis_db, processed_path)
        sample_extractors.append(spl_selector)

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

    if not test_only:
        bigdata_idx_0 = np.empty(shape=(cnt_idx_0 + 1, 3), dtype=np.uint16)
        bigdata_idx_1 = np.empty(shape=(cnt_idx_1 + 1, 3), dtype=np.uint16)

        print('Number of indexes for No Forest: %s' % (str(len(bigdata_idx_0))))
        print('Number of indexes for Forest: %s' % (str(len(bigdata_idx_1))))

        print('Copying and appending index values...')

        current_0_idx = 0
        current_1_idx = 0
        for i, pck in enumerate(sample_rasters_folders):
            path_to_pck = join(processed_path, pck, 'idxs.npz')

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

        print('Shuffling No Forest indexes...')
        np.random.shuffle(bigdata_idx_0)
        print('Shuffling Forest indexes...')
        np.random.shuffle(bigdata_idx_1)

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

        final_idx_0 = bigdata_idx_0[:class_percentage]
        del bigdata_idx_0
        final_idx_1 = bigdata_idx_1[:class_percentage]
        del bigdata_idx_1

        print('Storing data...')
        dataset_path = join(processed_path, 'samples_shuffled_factor_idx.npz')
        np.savez_compressed(dataset_path, bigdata_idx_0=final_idx_0, bigdata_idx_1=final_idx_1)

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