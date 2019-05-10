"""
Author:
    Gabriel Caffaratti

Description:
    This class contains utility methods to return different parameters of the raster file provided to the constructor.

Example:
    raster_util = RasterUtils('/path/to/some/raster_file.tif')
    raster_bounds = raster_util.retrieve_raster_boundaries()
    raster_pixelsize = raster_util.retrieve_raster_pixelsize()
"""
from raster_rw import GTiffHandler

class RasterUtils:

    def __init__(self, filename):
        """
        Constructor of the class

        :param filename: (str) Raster file name or path
        """
        self.filename = filename


    def retrieve_raster_boundaries(self):
        """
        Retrieves the boundaries of the raster file and the pixel size of the raster

        :return:
            raster_bounds: (array) List of boundaries (corner points) of the SRTM file raster
        """
        raster_handler = GTiffHandler()
        raster_handler.readFile(self.filename)

        raster_bounds = [raster_handler.src_geotransform[0],
                       raster_handler.src_geotransform[3] + (raster_handler.src_geotransform[5] * raster_handler.src_ysize),
                       raster_handler.src_geotransform[0] + (raster_handler.src_geotransform[1] * raster_handler.src_xsize),
                       raster_handler.src_geotransform[3]]

        raster_handler.closeFile()

        return raster_bounds


    def retrieve_raster_pixelsize(self):
        """
        Retrieves the pixel size of the raster file

        :return:
            raster_pixelsize: (array) List of pixel sizes (vertical and horizontal)
        """
        raster_handler = GTiffHandler()
        raster_handler.readFile(self.filename)

        raster_pixelsize = [raster_handler.src_geotransform[1], raster_handler.src_geotransform[5]]

        return raster_pixelsize