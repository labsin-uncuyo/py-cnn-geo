import sys, getopt
sys.path.append("..")
import pathlib
import gdal
import subprocess
import numpy as np
from os import listdir
from os.path import isfile, join
from natsort import natsorted
from raster_rw import GTiffHandler


class Constants:
    RASTER_ID = 'LANDSAT_PRODUCT_ID'
    BAND_RADIANCE_MULT_COEF = 'RADIANCE_MULT_BAND_'
    BAND_RADIANCE_ADD_COEF = 'RADIANCE_ADD_BAND_'
    BAND_REFLECTANCE_MULT_COEF = 'REFLECTANCE_MULT_BAND_'
    BAND_REFLECTANCE_ADD_COEF = 'REFLECTANCE_ADD_BAND_'
    BAND_K1_CONSTANT = 'K1_CONSTANT_BAND_'
    BAND_K2_CONSTANT = 'K2_CONSTANT_BAND_'
    RASTER_SUN_ELEVATION = 'SUN_ELEVATION'
    NO_DATA_VALUE = -9999


def main(argv):
    """
    Main function which shows the usage, retrieves the command line parameters and invokes the required functions to do
    the expected job.

    :param argv: (dictionary) options and values specified in the command line
    """

    print('Entering to the Landsat homogenizer')
    lst_folder = ''

    gdal.UseExceptions()

    try:
        opts, args = getopt.getopt(argv, "hs:", ["lstfolder="])
    except getopt.GetoptError:
        print('lst_homogenizer.py -s <landsat_folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('lst_homogenizer.py -s <landsat_folder>')
            sys.exit()
        elif opt in ["-s", "--lstfolder"]:
            lst_folder = arg

    print('Working with Landsat folder %s' % lst_folder)

    homogenize_rasters(lst_folder)

    sys.exit()


def homogenize_rasters(lst_folder):
    # Radiance multiplicative scaling factor for the band

    folderpath = str(pathlib.Path(lst_folder))

    mtl_files = [f for f in listdir(folderpath) if isfile(join(folderpath, f)) and '_MTL' in f]

    rasters_metadata = []
    for mtl_file in mtl_files:

        d = {}
        mtl_file_path = join(folderpath, mtl_file)
        with open(mtl_file_path) as f:
            for line in f:
                if line != 'END\n':
                    (key, val) = [str.strip(i) for i in line.split("=")]
                    d[key] = val.replace('"', '')

        rasters_metadata.append(d)

    for metadata in rasters_metadata:
        raster_id = metadata[Constants.RASTER_ID]

        band_files = [f for f in listdir(folderpath) if
                      isfile(join(folderpath, f)) and raster_id in f and '.TIF' in f[
                                                                                   -4:] and '_BQA' not in f and '_H' not in f]

        band_files = natsorted(band_files, key=lambda y: y.lower())

        print(band_files)

        sun_elevation = float(metadata[Constants.RASTER_SUN_ELEVATION])
        for i, band_file in enumerate(band_files):

            band_file_path = join(folderpath, band_file)
            band = i + 1

            print("Homogenizing file %s" % (band_file_path))

            data_handler = GTiffHandler()
            data_handler.readFile(band_file_path)
            data = np.array(data_handler.src_Z, dtype=np.float32)
            if band < 10:
                band_refl_mult_coef = float(metadata[Constants.BAND_REFLECTANCE_MULT_COEF + str(band)])
                band_refl_add_coef = float(metadata[Constants.BAND_REFLECTANCE_ADD_COEF + str(band)])

                data = np.where(data != 0, calculate_P_lambda(data, band_refl_mult_coef, band_refl_add_coef, sun_elevation), Constants.NO_DATA_VALUE)
            else:
                band_rad_mult_coef = float(metadata[Constants.BAND_RADIANCE_MULT_COEF + str(band)])
                band_rad_add_coef = float(metadata[Constants.BAND_RADIANCE_ADD_COEF + str(band)])
                band_k1_constant = float(metadata[Constants.BAND_K1_CONSTANT + str(band)])
                band_k2_constant = float(metadata[Constants.BAND_K2_CONSTANT + str(band)])

                data = np.where(data != 0, calculate_T(data, band_rad_mult_coef, band_rad_add_coef, band_k1_constant, band_k2_constant), Constants.NO_DATA_VALUE)

            data_handler.src_datatype = gdal.GDT_Float32

            data_handler.src_Z = data

            data_handler.src_nodatavalue = Constants.NO_DATA_VALUE

            new_file_path = band_file_path[:-4] + '_H.TIF'
            data_handler.writeNewFile(new_file_path)

            data_handler.closeFile()

    files_to_del = [f for f in listdir(folderpath) if isfile(join(folderpath, f)) and 'LC08' in f and '_H' not in f]

    for file_to_del in files_to_del:
        file_to_del_path = join(folderpath, file_to_del)

        execution = subprocess.run(['rm', file_to_del_path])

    print('check')


# Refer to the equation 5.1 of the document
# https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1574_L8_Data_Users_Handbook.pdf
def calculate_L_lambda(pixel_DN, band_radiance_mult_coef, band_radiance_add_coef):
    return (pixel_DN * band_radiance_mult_coef) + band_radiance_add_coef


# Refer to the equation 5.2.a of the document
# https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1574_L8_Data_Users_Handbook.pdf
def calculate_P_lambda_prime(pixel_DN, band_reflectance_mult_coef, band_reflectance_add_coef):
    return (pixel_DN * band_reflectance_mult_coef) + band_reflectance_add_coef


# Refer to the equation 5.2.b of the document
# https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1574_L8_Data_Users_Handbook.pdf
def calculate_P_lambda(pixel_DN, band_reflectance_mult_coef, band_reflectance_add_coef, raster_sun_elevation):
    return calculate_P_lambda_prime(pixel_DN, band_reflectance_mult_coef, band_reflectance_add_coef) / np.sin(
        (raster_sun_elevation * np.pi)/180.0)


# Refer to the equation 5.3 of the document
# https://prd-wret.s3-us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1574_L8_Data_Users_Handbook.pdf
def calculate_T(pixel_DN, band_radiance_mult_coef, band_radiance_add_coef, band_K1_constant, band_K2_constant):
    return band_K2_constant / np.log(
        (band_K1_constant / calculate_L_lambda(pixel_DN, band_radiance_mult_coef, band_radiance_add_coef)) + 1)


main(sys.argv[1:])
