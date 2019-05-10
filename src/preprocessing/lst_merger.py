"""
Author: Gabriel Caffaratti

Description:
This script merges different Landsat rasters by band creating a single file will all the information contained.

Example:
    $ python lst_merger.py -s path/to/landsat/files/
    $ python lst_merger.py -h
"""

import sys, getopt
import pathlib
import subprocess
from os import listdir
from os.path import isfile, join
from natsort import natsorted


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Entering to the Landsat merger')
    lst_folder = ''

    try:
        opts, args = getopt.getopt(argv, "hs:", ["lstfolder="])
    except getopt.GetoptError:
        print('lst_merger.py -s <landsat_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('lst_merger.py -s <landsat_folder>')
            sys.exit()
        elif opt in ["-s", "--lstfolder"]:
            lst_folder = arg

    print('Working with Landsat folder %s' % lst_folder)

    merge_rasters(lst_folder)

    sys.exit()


def merge_rasters(lst_folder):
    folderpath = str(pathlib.Path(lst_folder))

    if not isfile(folderpath):

        for i in range(11):
            file_sufix = 'B' + str(i + 1) + '_H.TIF'

            files_to_merge = [(folderpath + '/' + f) for f in listdir(folderpath) if
                              (isfile(join(folderpath, f)) and file_sufix in f)]

            files_to_merge = natsorted(files_to_merge, key=lambda y: y.lower())

            final_file_name = folderpath + '/' + files_to_merge[0].split('/')[-1][0:10] + 'MERGE_' + file_sufix

            print('Files with sufix %s' % file_sufix)
            print(files_to_merge)

            cmd = ['gdalwarp', '--config', 'GDAL_CACHEMAX', '5000', '-wm', '5000', '-multi', '-overwrite', '-srcnodata',
                   '-9999', '-r', 'bilinear']
            #cmd = ['gdalwarp', '--config', 'GDAL_CACHEMAX', '5000', '-wm', '5000', '-multi', '-overwrite', '-srcnodata',
            #       '-9999', '-r', 'max']
            cmd.extend(files_to_merge)
            cmd.append(final_file_name)

            # print(cmd)
            execution = subprocess.run(cmd)

            rmcmd = ['rm']
            rmcmd.extend(files_to_merge)
            execution = subprocess.run(rmcmd)


main(sys.argv[1:])
