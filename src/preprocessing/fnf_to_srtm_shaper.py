"""
Author: Gabriel Caffaratti

Description:
This python script changes the pixel size of the FNF file situated in the same folder of the SRTM raster file provided
in the parameters. It also converts the FNF raster to GTiff to match the SRTM and Landsat formats. The script assumes
the FNF file location and shape covers the SRTM file provided. It also assumes the FNF file will start with 'FNF'.

Example:
    $ python fnf_to_srtm_shaper.py -s path/to/srtm_raster_file.tif
    $ python fnf_to_srtm_shaper.py -h
"""

import sys, getopt
sys.path.append("..")
import pathlib
import subprocess
import gdal
from raster_utils import RasterUtils
from os import listdir
from os.path import isfile, join


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Entering to the FNF to SRTM shaper')
    gdal.UseExceptions()
    dataset_folder = ''

    try:
        opts, args = getopt.getopt(argv, "hs:", ["dataset_folder="])
    except getopt.GetoptError:
        print('fnf_to_srtm_shaper.py -s <dataset_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('fnf_to_srtm_shaper.py -s <dataset_folder>')
            sys.exit()
        elif opt in ["-s", "--dataset_folder"]:
            dataset_folder = arg

    srtm_filename = [f for f in listdir(dataset_folder) if isfile(join(dataset_folder, f)) and f[0:4] == 'SRTM']

    assert len(srtm_filename) == 1

    srtm_filepath = join(dataset_folder, srtm_filename[0])

    print('Working with SRTM file %s' % srtm_filepath)

    raster_utils = RasterUtils(srtm_filepath)
    srtm_pixelsize = raster_utils.retrieve_raster_pixelsize()

    change_fnf_raster(srtm_filepath, srtm_pixelsize)

    sys.exit()


def change_fnf_raster(srtm_filename, srtm_pixels):
    """
    Changes the FNF raster file with name starting with 'FNF' contained in the same folder of the SRTM file. The changes
    performed are basically the translation to GTiff format and the alignment of the pixel size according with the SRTM
    pixel.

    The script assumes the FNF raster's original surface covers all the SRTM surface.

    It worth to mention the original FNF file will be deleted after the complete modification

    :param srtm_filename: (str) File name or path to the SRTM raster file
    :param srtm_pixels: (array) List of pixel sizes (vertical and horizontal)
    """
    folderpath = str(pathlib.Path(srtm_filename).parent)

    onlyfiles = [f for f in listdir(folderpath) if isfile(join(folderpath, f)) and f[0:3] == 'FNF']

    files_to_remove = []
    for file in onlyfiles:
        if '.hdr' not in file and '.tar' not in file:
            print('Changing FNF raster with name %s' % str(file))

            lst_file_path = join(folderpath, file)
            lst_tif_file_path = lst_file_path + ".tif"
            lst_tif_pix_file_path = join(folderpath, lst_tif_file_path.split('/')[-1].split(".")[-2] + "_PIX." +
                                         lst_tif_file_path.split('/')[-1].split(".")[1])

            print('Transforming GeoTIF file...')

            execution = subprocess.run(['gdal_translate', '-of', 'GTiff', lst_file_path, lst_tif_file_path])

            print('Modifying pixel size to %s' % str(srtm_pixels))

            execution = subprocess.run(
                ['gdalwarp', '-tr', str(srtm_pixels[0]), str(srtm_pixels[1]), '-r', 'bilinear', lst_tif_file_path,
                 lst_tif_pix_file_path])

            print('Removing previous files')

            execution = subprocess.run(['rm', lst_file_path, lst_tif_file_path, lst_tif_file_path + '.aux.xml'])
        elif '.hdr' in file or '.tar' in file:
            hdr_file_path = folderpath + '/' + file
            files_to_remove.append(hdr_file_path)

    for rmfile in files_to_remove:
        print('Removing hdr or tar file')

        execution = subprocess.run(['rm', rmfile])

    print('The FNF file found was SRTM-shaped')


main(sys.argv[1:])
