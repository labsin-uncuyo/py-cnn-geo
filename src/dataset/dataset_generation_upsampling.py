# os.environ["MKL_NUM_THREADS"] = "3"
# os.environ["NUMEXPR_NUM_THREADS"] = "3"
# os.environ["OMP_NUM_THREADS"] = "3"

import sys, getopt
import gdal
import time
import curses
import redis
import multiprocessing
from os import listdir
from os.path import isfile, join
from dataset.threads.sample_extractor_thread import SampleExtractorThread
from dataset.threads.sample_selector_thread_status import SampleSelectorThreadStatus


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Preparing for dataset creation')
    gdal.UseExceptions()
    src_folder = ''

    cpus = multiprocessing.cpu_count()

    try:
        opts, args = getopt.getopt(argv, "hs:", ["src_folder="])
    except getopt.GetoptError:
        print('dataset_generation_upsampling.py -s <source_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('dataset_generation_upsampling.py -s <source_folder>')
            sys.exit()
        elif opt in ["-s", "--src_folder"]:
            src_folder = arg

    print('Working with source folder %s' % src_folder)

    explore_dataset_folders(src_folder)

    sys.exit()


def explore_dataset_folders(src_folder):
    sample_rasters_folders = [f for f in listdir(src_folder) if not isfile(join(src_folder, f))]

    sample_rasters_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print(sample_rasters_folders)

    redises = []
    threads = list()
    sample_extractors = []
    for i, pck in enumerate(sample_rasters_folders):
        path_to_pck = join(src_folder, pck)

        redis_db = redis.Redis(db=i)
        redis_db.delete('processed', 'total', 'status')
        redises.append(redis_db)

        spl_selector = SampleExtractorThread(i, path_to_pck, redis_db)
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

    print('Storing processed files in disk...')

    print('stop here')


def scale(X, x_min, x_max):
    nom = (X - X.min()) * (x_max - x_min)
    denom = X.max() - X.min()
    denom = denom + (denom is 0)
    return x_min + nom / denom


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


# def progress_bar(current_value, total):
#    increments = 50
#    percentual = ((current_value/ total) * 100)
#    i = int(percentual // (100 / increments ))
#    text = "\r[{0: <{1}}] {2}%".format('=' * i, increments, "{0:.2f}".format(percentual))
#    print(text, end="\n" if percentual == 100 else "")


main(sys.argv[1:])
