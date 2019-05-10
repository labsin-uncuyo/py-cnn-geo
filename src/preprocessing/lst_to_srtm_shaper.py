"""
Author: Gabriel Caffaratti

Description:
This python script changes the projection system and pixel size of the Landsat files situated in the same folder of the
SRTM raster file provided in the parameters. It also clips the Landsat rasters to the size of the SRTM ones. The script
assumes the Landsat files location and shape covers the SRTM file provided. It also assumes the Landsat files will start
with 'LC08'.

Example:
    $ python lst_to_srtm_shaper.py -s path/to/srtm_raster_file.tif
    $ python lst_to_srtm_shaper.py -h
"""

import sys, getopt
sys.path.append("..")
import pathlib
import subprocess
from natsort import natsorted
from raster_utils import RasterUtils
from os import listdir
from os.path import isfile, join


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Entering to the Landsat to SRTM shaper')
    dataset_folder = ''

    try:
        opts, args = getopt.getopt(argv, "hs:", ["dataset_folder="])
    except getopt.GetoptError:
        print('lst_to_srtm_shaper.py -s <dataset_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('lst_to_srtm_shaper.py -s <dataset_folder>')
            sys.exit()
        elif opt in ["-s", "--dataset_folder"]:
            dataset_folder = arg

    srtm_filename = [f for f in listdir(dataset_folder) if isfile(join(dataset_folder, f)) and f[0:4] == 'SRTM']

    assert len(srtm_filename) == 1

    srtm_filepath = join(dataset_folder, srtm_filename[0])

    print('Working with SRTM file %s' % srtm_filepath)

    raster_utils = RasterUtils(srtm_filepath)
    srtm_bounds = raster_utils.retrieve_raster_boundaries()
    srtm_pixelsize = raster_utils.retrieve_raster_pixelsize()

    change_lst_rasters(srtm_filepath, srtm_bounds, srtm_pixelsize)

    sys.exit()


def change_lst_rasters(srtm_filename, srtm_bounds, srtm_pixels):
    """
    Changes the Landsat raster files with name starting with 'LC08' contained in the same folder of the SRTM file. The
    changes performed are basically the transformation of the projection (from UTM to WGS84), the alignment of the pixel
    size according with the SRTM pixel, and the clipping of the raster based on the SRTM raster boundaries.

    The script assumes the Landsat rasters' original surface covers all the SRTM surface.

    It worth to mention the original Landsat files will be deleted after each file complete modification

    :param srtm_filename: (str) File name or path to the SRTM raster file
    :param srtm_bounds: (array) List of boundaries (corner points) of the SRTM file raster
    :param srtm_pixels: (array) List of pixel sizes (vertical and horizontal)
    """
    folderpath = str(pathlib.Path(srtm_filename).parent)

    onlyfiles = [f for f in listdir(folderpath) if isfile(join(folderpath, f))]

    onlyfiles = natsorted(onlyfiles, key=lambda y: y.lower())

    for file in onlyfiles:
        if 'LC08' in file:
            print('Changing Landsat raster with name %s' % str(file))

            lst_file_path = join(folderpath, file)
            lst_wgs_file_path = join(folderpath,
                                     file.split('/')[-1].split(".")[-2] + "_WGS." + file.split('/')[-1].split(".")[-1])
            lst_wgs_pix_file_path = join(folderpath, lst_wgs_file_path.split('/')[-1].split(".")[-2] + "_PIX." +
                                         lst_wgs_file_path.split('/')[-1].split(".")[-1])
            lst_wgs_pix_clip_file_path = join(folderpath,
                                              lst_wgs_pix_file_path.split('/')[-1].split(".")[-2] + "_CLIP." + \
                                              lst_wgs_pix_file_path.split('/')[-1].split(".")[-1])

            print('Transforming from UTM to WGS84...')

            execution = subprocess.call(
                'gdalwarp -t_srs \'+proj=longlat +ellps=WGS84\' ' + lst_file_path + ' ' + lst_wgs_file_path, shell=True)

            print('Modifying pixel size to %s' % str(srtm_pixels))

            execution = subprocess.run(
                ['gdalwarp', '-tr', str(srtm_pixels[0]), str(srtm_pixels[1]), '-r', 'bilinear', lst_wgs_file_path,
                 lst_wgs_pix_file_path])

            print('Clipping raster to SRTM size')

            execution = subprocess.run(
                ['gdalwarp', '-te', str(srtm_bounds[0]), str(srtm_bounds[1]), str(srtm_bounds[2]), str(srtm_bounds[3]),
                 lst_wgs_pix_file_path, lst_wgs_pix_clip_file_path])

            print('Removing previous files')

            execution = subprocess.run(['rm', lst_file_path, lst_wgs_file_path, lst_wgs_pix_file_path])

    print('All the Landsat files found were SRTM-shaped')


main(sys.argv[1:])
