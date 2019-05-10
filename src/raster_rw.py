# -*- coding: utf-8 -*-

from osgeo import gdal
from gdalconst import *
import numpy as np

class GTiffHandler:
    """GTiff read/write handler of data and metadata.

    This class contains attributes and methods helpful for handling the GTiff files containing raster information. The
    methods and attributes allow to not only read its information but also override the recently read file, or write a
    new one based either on the information received or the original file.

    Constants:
        FORMAT (str): Defines the file format (GTiff)

    Attributes:
        src_filename (str): Filename or path to the GTiff
        src_filehandler (GDALDataset): Contains the GDAL dataset of the GTiff file once it's opened
        src_band1 (GDALRasterBand): Raster band contained in the GDAL dataset
        src_geotransform (array): Affine transformation coefficients of the raster
        src_geoproj (str): Projection definition string of the dataset
        src_Z (array): Array containing the data of the band 1 of the dataset
        src_xsize (int): Horizontal size of the band 1
        src_ysize (int): Vertical size of the band 1
        src_nodatavalue (Any): Any value used in case the field has no data
        src_datatype (GDALDataType): Type of data of the raster

        """

    def __init__(self):
        """
        Instantiation method of the class
        """

        # constants
        self.FORMAT = "GTiff"

        # source file data and metadata
        self.src_filename = None
        self.src_filehandler = None
        self.src_band1 = None
        self.src_geotransform = None
        self.src_geoproj = None
        self.src_Z = None
        self.src_xsize = None
        self.src_ysize = None
        self.src_nodatavalue = None
        self.src_datatype = None

    def readFile(self, filename):
        """
        This method loads a raster from the received raster file name or path

        :param filename: (str) Raster file name or file path

        """
        self.src_filename = filename
        self.src_filehandler = gdal.Open(self.src_filename, GA_Update)
        self.src_band1 = self.src_filehandler.GetRasterBand(1)
        self.src_geotransform = self.src_filehandler.GetGeoTransform()
        self.src_geoproj = self.src_filehandler.GetProjection()
        self.src_Z = self.src_band1.ReadAsArray()
        self.src_xsize = self.src_filehandler.RasterXSize
        self.src_ysize = self.src_filehandler.RasterYSize
        self.src_nodatavalue = self.src_band1.GetNoDataValue()
        self.src_datatype = self.src_band1.DataType

    def writeSrcFile(self):
        """
        Saves changes in the current handler's raster.

        """
        if self.src_filehandler is not None:

            if self.src_filehandler.GetRasterBand(1).DataType != self.src_datatype and self.src_datatype is not None:
                self.src_filehandler.GetRasterBand(1).DataType = self.src_datatype

            if self.src_filehandler.GetRasterBand(1).GetNoDataValue() != self.src_nodatavalue and self.src_nodatavalue is not None:
                self.src_filehandler.GetRasterBand(1).SetNoDataValue(self.src_nodatavalue)

            self.src_filehandler.GetRasterBand(1).WriteArray(self.src_Z)

            if self.src_filehandler.GetGeoTransform() != self.src_geotransform and self.src_geotransform is not None:
                self.src_filehandler.SetGeoTransform(self.src_geotransform)

            if self.src_filehandler.GetProjection() != self.src_geoproj and self.src_geoproj is not None:
                self.src_filehandler.SetProjection(self.src_geoproj)

            self.src_filehandler.FlushCache()


    def writeNewFile(self, filename, geotransform = None, geoprojection = None, datatype = None, nodatavalue = None, data = None):
        """
        Creates a new raster file with name or path as it was received in the filename parameter. The new file will have
        the same properties and data of the current handler source file, or it will be changed by the received
        parameters.

        :param filename: (str) New raster file name or path.
        :param geotransform: (array) Affine transformation coefficients of the raster [optional]
        :param geoprojection: (str) Projection definition string of the dataset [optional]
        :param datatype: (GDALDataType) Type of data of the raster [optional]
        :param nodatavalue: (Any) Any value used in case the field has no data [optional]
        :param data: (array) Array containing the data of the band 1 of the dataset [optional]

        """
        if data is not None:
            (x, y) = data.shape
        else:
            (x, y) = self.src_Z.shape

        driver = gdal.GetDriverByName(self.FORMAT)

        if datatype is not None:
            dst_datatype = datatype
        else:
            dst_datatype = self.src_datatype

        dst_ds = driver.Create(filename, y, x, 1, dst_datatype)

        if data is not None:
            dst_ds.GetRasterBand(1).WriteArray(data)
        else:
            dst_ds.GetRasterBand(1).WriteArray(self.src_Z)

        if geotransform is not None:
            dst_ds.SetGeoTransform(geotransform)
        else:
            dst_ds.SetGeoTransform(self.src_geotransform)

        if geoprojection is not None:
            dst_ds.SetProjection(geoprojection)
        else:
            dst_ds.SetProjection(self.src_geoproj)

        if nodatavalue is not None:
            dst_ds.GetRasterBand(1).SetNoDataValue(nodatavalue)
        elif self.src_nodatavalue is not None:
            dst_ds.GetRasterBand(1).SetNoDataValue(self.src_nodatavalue)


    def closeFile(self):
        """
        Fully dereferences the dataset
        """
        self.src_filehandler = None
        del self.src_filehandler
