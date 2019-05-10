import sys
import getopt
import gdal
from os import listdir
from os.path import isfile, join


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Preparing for samples creation')
    gdal.UseExceptions()
    src_folder = ''

    try:
        opts, args = getopt.getopt(argv, "hs:", ["src_folder="])
    except getopt.GetoptError:
        print('sample_generation_upsampling.py -s <source_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('sample_generation_upsampling.py -s <source_folder>')
            sys.exit()
        elif opt in ["-s", "--src_folder"]:
            src_folder = arg

    print('Working with source folder %s' % src_folder)

    explore_dataset_folders(src_folder)

    sys.exit()


def explore_dataset_folders(dataset_folder):
    sample_rasters_folders = [f for f in listdir(dataset_folder) if not isfile(join(dataset_folder, f))]

    sample_rasters_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print(sample_rasters_folders)

    noforest_cnt = 0
    forest_cnt = 0
    for i, pck in enumerate(sample_rasters_folders):
        path_to_pck = join(dataset_folder, pck, 'dataset.npz')

        pck_bigdata = None
        item_getter = itemgetter('bigdata')
        with np.load(path_to_pck) as df:
            pck_bigdata = item_getter(df)


main(sys.argv[1:])