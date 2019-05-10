"""
Author: Gabriel Caffaratti

Description:
This python script performs a binaryzation of a FNF raster changing all the no forest values to zero and accommodating
the color table accordingly.

Example:
    $ python fnf_binaryzer.py -i path/to/some/fnf_raster_file.tif
    $ python fnf_binaryzer.py -h
"""

import sys, getopt
sys.path.append("..")
import gdal
import numpy as np
from raster_rw import GTiffHandler
from os import listdir
from os.path import isfile, join


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """
    gdal.UseExceptions()
    dataset_folder = ''

    try:
        opts, args = getopt.getopt(argv, "hs:", ["dataset_folder="])
    except getopt.GetoptError:
        print('fnf_binaryzer.py -s <dataset_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('fnf_binaryzer.py -s <dataset_folder>')
            sys.exit()
        elif opt in ("-s", "--dataset_folder"):
            dataset_folder = arg

    fnf_filename = [f for f in listdir(dataset_folder) if isfile(join(dataset_folder, f)) and f[0:3] == 'FNF']

    assert len(fnf_filename) == 1

    fnf_filepath = join(dataset_folder, fnf_filename[0])

    print('Trying to binaryze the FNF raster file %s' % fnf_filepath)
    binaryzeFNF(fnf_filepath)
    sys.exit()


def binaryzeFNF(fnf_filename):
    """
    Opens the FNF raster file allocated in the received parameter, changes all the values bigger than 1 to 0, leaving
    the forest values as 1 and all the rest of the pixels as 0. It also accommodates the color table so now the 'sand'
    color is used for the 0 value, and the green for the 1 value.

    :param fnf_filename: (str) file path of the FNF raster
    """
    fnf_handler = GTiffHandler()
    fnf_handler.readFile(fnf_filename)

    if fnf_handler.src_filehandler is not None:
        fnf_np_array = np.array(fnf_handler.src_Z)

        # accommodating fnf array values so forest is 1, no forest is 0
        fnf_np_array[fnf_np_array > 1] = 0

        unique, counts = np.unique(fnf_np_array, return_counts=True)
        print('After accommodating values the distribution of no forest (0) and forest (1) is: ',
              dict(zip(unique, counts)))

        print('Changing color table...')
        colorTable = fnf_handler.src_band1.GetColorTable()

        zero_colorEntry = colorTable.GetColorEntry(2)
        null_colorEntry = colorTable.GetColorEntry(0)

        colorTable.SetColorEntry(0, zero_colorEntry)
        colorTable.SetColorEntry(2, null_colorEntry)
        colorTable.SetColorEntry(3, null_colorEntry)

        fnf_handler.src_filehandler.GetRasterBand(1).SetColorTable(colorTable)

        print('Saving changes and storing...')
        fnf_handler.src_Z = fnf_np_array
        fnf_handler.writeSrcFile()
        fnf_handler.closeFile()

        print('All done')


main(sys.argv[1:])
