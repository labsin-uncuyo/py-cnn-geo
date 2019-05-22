## Preprocessing
Each zone is preprocessed in order to accomodate pixel size, projection and coverage area according to SRTM raster parameters. The different steps are executed on each folder under the raw directory under src/data.

### Naming
The preprocessing scripts expect a naming convention for the different files of a zone.

- SRTM raster: It should start with "SRTM_". In this example we added the prefix and left only the coodenates specifications removing the rest of the data in the name:

```console
foo@bar:1$ mv s26_w053_1arc_v3.tf SRTM_S26W053.tif
```

- Landsat 8 files: These files should start with the prefix "LC08_". There's no need to perform any preprocessing step, just extracting the files and then removing the compressed files.

```console
foo@bar:1$ tar -xzf LC08*.tar.gz
foo@bar:1$ rm LC08*.tar.gz
```

- JAXA Forest - No-Forest files: These files should start with the prefix "FNF_". In this example we decompressed the tar.gz file and simply added the prefix to each file:

```console
foo@bar:1$ tar -xzf *_F02DAR*
foo@bar:1$ for filename in *_F02DAR*; do mv "$filename" "FNF_$filename"; done;
```

### Landsat Homogenizing

Landsat-8 files need to be converted from DNs to reflectance and radiance values. The following script under src/preprocessing does the job:

```console
foo@bar:preprocessing$ python3 lst_homogenizer.py -s ../data/raw/train/1/
Entering to the Landsat homogenizer
Working with Landsat folder ../data/raw/train/1/
['LC08_L1TP_222077_20140301_20170425_01_T1_B1.TIF', 'LC08_L1TP_222077_20140301_20170425_01_T1_B2.TIF', 'LC08_L1TP_222077_20140301_20170425_01_T1_B3.TIF', 'LC08_L1TP_222077_20140301_20170425_01_T1_B4.TIF', 'LC08_L1TP_222077_20140301_20170425_01_T1_B5.TIF', 'LC08_L1TP_222077_20140301_20170425_01_T1_B6.TIF', 'LC08_L1TP_222077_20140301_20170425_01_T1_B7.TIF', 'LC08_L1TP_222077_20140301_20170425_01_T1_B8.TIF', 'LC08_L1TP_222077_20140301_20170425_01_T1_B9.TIF', 'LC08_L1TP_222077_20140301_20170425_01_T1_B10.TIF', 'LC08_L1TP_222077_20140301_20170425_01_T1_B11.TIF']
Homogenizing file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B1.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B2.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B3.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B4.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B5.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B6.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B7.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B8.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B9.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B10.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B11.TIF
['LC08_L1TP_223078_20140204_20170426_01_T1_B1.TIF', 'LC08_L1TP_223078_20140204_20170426_01_T1_B2.TIF', 'LC08_L1TP_223078_20140204_20170426_01_T1_B3.TIF', 'LC08_L1TP_223078_20140204_20170426_01_T1_B4.TIF', 'LC08_L1TP_223078_20140204_20170426_01_T1_B5.TIF', 'LC08_L1TP_223078_20140204_20170426_01_T1_B6.TIF', 'LC08_L1TP_223078_20140204_20170426_01_T1_B7.TIF', 'LC08_L1TP_223078_20140204_20170426_01_T1_B8.TIF', 'LC08_L1TP_223078_20140204_20170426_01_T1_B9.TIF', 'LC08_L1TP_223078_20140204_20170426_01_T1_B10.TIF', 'LC08_L1TP_223078_20140204_20170426_01_T1_B11.TIF']
Homogenizing file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B1.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B2.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B3.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B4.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B5.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B6.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B7.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B8.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B9.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B10.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B11.TIF
['LC08_L1TP_223077_20140204_20170426_01_T1_B1.TIF', 'LC08_L1TP_223077_20140204_20170426_01_T1_B2.TIF', 'LC08_L1TP_223077_20140204_20170426_01_T1_B3.TIF', 'LC08_L1TP_223077_20140204_20170426_01_T1_B4.TIF', 'LC08_L1TP_223077_20140204_20170426_01_T1_B5.TIF', 'LC08_L1TP_223077_20140204_20170426_01_T1_B6.TIF', 'LC08_L1TP_223077_20140204_20170426_01_T1_B7.TIF', 'LC08_L1TP_223077_20140204_20170426_01_T1_B8.TIF', 'LC08_L1TP_223077_20140204_20170426_01_T1_B9.TIF', 'LC08_L1TP_223077_20140204_20170426_01_T1_B10.TIF', 'LC08_L1TP_223077_20140204_20170426_01_T1_B11.TIF']
Homogenizing file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B1.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B2.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B3.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B4.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B5.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B6.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B7.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B8.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B9.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B10.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B11.TIF
['LC08_L1TP_222078_20140301_20170425_01_T1_B1.TIF', 'LC08_L1TP_222078_20140301_20170425_01_T1_B2.TIF', 'LC08_L1TP_222078_20140301_20170425_01_T1_B3.TIF', 'LC08_L1TP_222078_20140301_20170425_01_T1_B4.TIF', 'LC08_L1TP_222078_20140301_20170425_01_T1_B5.TIF', 'LC08_L1TP_222078_20140301_20170425_01_T1_B6.TIF', 'LC08_L1TP_222078_20140301_20170425_01_T1_B7.TIF', 'LC08_L1TP_222078_20140301_20170425_01_T1_B8.TIF', 'LC08_L1TP_222078_20140301_20170425_01_T1_B9.TIF', 'LC08_L1TP_222078_20140301_20170425_01_T1_B10.TIF', 'LC08_L1TP_222078_20140301_20170425_01_T1_B11.TIF']
Homogenizing file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B1.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B2.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B3.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B4.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B5.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B6.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B7.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B8.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B9.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B10.TIF
Homogenizing file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B11.TIF
check
```

### Landsat Merging
In some zones multiple landsat files are required to cover a SRTM raster zone. In this case, each file corresponging to a band needs to be merged in one file. The following script under the src/preprocessing directory does the described job:

```console
foo@bar:preprocessing$ python3 lst_merger.py -s ../data/raw/train/1/
Entering to the Landsat merger
Working with Landsat folder ../data/raw/train/1/
Files with sufix B1_H.TIF
['../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B1_H.TIF', '../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B1_H.TIF', '../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B1_H.TIF', '../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B1_H.TIF']
Creating output file that is 14021P x 13101L.
Processing input file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B1_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B1_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B1_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B1_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B1_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B1_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Files with sufix B2_H.TIF
['../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B2_H.TIF', '../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B2_H.TIF', '../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B2_H.TIF', '../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B2_H.TIF']
Creating output file that is 14021P x 13101L.
Processing input file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B2_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B2_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B2_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B2_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B2_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B2_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Files with sufix B3_H.TIF
['../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B3_H.TIF', '../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B3_H.TIF', '../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B3_H.TIF', '../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B3_H.TIF']
Creating output file that is 14021P x 13101L.
Processing input file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B3_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B3_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B3_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B3_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B3_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B3_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Files with sufix B4_H.TIF
['../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B4_H.TIF', '../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B4_H.TIF', '../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B4_H.TIF', '../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B4_H.TIF']
Creating output file that is 14021P x 13101L.
Processing input file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B4_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B4_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B4_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B4_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B4_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B4_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Files with sufix B5_H.TIF
['../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B5_H.TIF', '../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B5_H.TIF', '../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B5_H.TIF', '../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B5_H.TIF']
Creating output file that is 14021P x 13101L.
Processing input file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B5_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B5_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B5_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B5_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B5_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B5_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Files with sufix B6_H.TIF
['../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B6_H.TIF', '../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B6_H.TIF', '../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B6_H.TIF', '../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B6_H.TIF']
Creating output file that is 14021P x 13101L.
Processing input file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B6_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B6_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B6_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B6_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B6_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B6_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Files with sufix B7_H.TIF
['../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B7_H.TIF', '../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B7_H.TIF', '../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B7_H.TIF', '../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B7_H.TIF']
Creating output file that is 14021P x 13101L.
Processing input file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B7_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B7_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B7_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B7_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B7_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B7_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Files with sufix B8_H.TIF
['../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B8_H.TIF', '../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B8_H.TIF', '../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B8_H.TIF', '../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B8_H.TIF']
Creating output file that is 28041P x 26201L.
Processing input file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B8_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B8_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B8_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B8_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B8_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B8_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Files with sufix B9_H.TIF
['../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B9_H.TIF', '../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B9_H.TIF', '../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B9_H.TIF', '../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B9_H.TIF']
Creating output file that is 14021P x 13101L.
Processing input file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B9_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B9_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B9_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B9_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B9_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B9_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Files with sufix B10_H.TIF
['../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B10_H.TIF', '../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B10_H.TIF', '../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B10_H.TIF', '../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B10_H.TIF']
Creating output file that is 14021P x 13101L.
Processing input file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B10_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B10_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B10_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B10_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B10_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B10_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Files with sufix B11_H.TIF
['../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B11_H.TIF', '../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B11_H.TIF', '../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B11_H.TIF', '../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B11_H.TIF']
Creating output file that is 14021P x 13101L.
Processing input file ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B11_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_222077_20140301_20170425_01_T1_B11_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B11_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_222078_20140301_20170425_01_T1_B11_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223077_20140204_20170426_01_T1_B11_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Processing input file ../data/raw/train/1/LC08_L1TP_223078_20140204_20170426_01_T1_B11_H.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
```

### Landsat Shapping
Once the Landsat files are homogenized and (if it's needed) merged, the files are ready to be converted from UTM projection to WGS-84, then their pixel size is accommodated to the SRTM raster pixel size, and finally the files are clipped to the SRTM coverage zone. The following script under the src/preprocessing folder does the respective job:

```console
foo@bar:preprocessing$ python3 lst_to_srtm_shaper.py -s ../data/raw/train/1/         
Entering to the Landsat to SRTM shaper
Working with SRTM file ../data/raw/train/1/SRTM_S26W053.tif
Changing Landsat raster with name LC08_L1TP_MERGE_B1_H.TIF
Transforming from UTM to WGS84...
Creating output file that is 14857P x 12574L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B1_H.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B1_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B1_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B1_H_WGS.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Modifying pixel size to [0.0002777777777777778, -0.0002777777777777778]
Creating output file that is 15260P x 12915L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B1_H_WGS.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B1_H_WGS.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B1_H_WGS.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B1_H_WGS_PIX.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Clipping raster to SRTM size
Creating output file that is 3601P x 3601L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B1_H_WGS_PIX.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B1_H_WGS_PIX.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B1_H_WGS_PIX.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B1_H_WGS_PIX_CLIP.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Removing previous files
Changing Landsat raster with name LC08_L1TP_MERGE_B2_H.TIF
Transforming from UTM to WGS84...
Creating output file that is 14857P x 12574L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B2_H.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B2_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B2_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B2_H_WGS.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Modifying pixel size to [0.0002777777777777778, -0.0002777777777777778]
Creating output file that is 15260P x 12915L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B2_H_WGS.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B2_H_WGS.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B2_H_WGS.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B2_H_WGS_PIX.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Clipping raster to SRTM size
Creating output file that is 3601P x 3601L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B2_H_WGS_PIX.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B2_H_WGS_PIX.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B2_H_WGS_PIX.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B2_H_WGS_PIX_CLIP.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Removing previous files
Changing Landsat raster with name LC08_L1TP_MERGE_B3_H.TIF
Transforming from UTM to WGS84...
Creating output file that is 14857P x 12574L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B3_H.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B3_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B3_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B3_H_WGS.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Modifying pixel size to [0.0002777777777777778, -0.0002777777777777778]
Creating output file that is 15260P x 12915L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B3_H_WGS.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B3_H_WGS.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B3_H_WGS.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B3_H_WGS_PIX.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Clipping raster to SRTM size
Creating output file that is 3601P x 3601L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B3_H_WGS_PIX.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B3_H_WGS_PIX.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B3_H_WGS_PIX.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B3_H_WGS_PIX_CLIP.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Removing previous files
Changing Landsat raster with name LC08_L1TP_MERGE_B4_H.TIF
Transforming from UTM to WGS84...
Creating output file that is 14857P x 12574L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B4_H.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B4_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B4_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B4_H_WGS.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Modifying pixel size to [0.0002777777777777778, -0.0002777777777777778]
Creating output file that is 15260P x 12915L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B4_H_WGS.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B4_H_WGS.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B4_H_WGS.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B4_H_WGS_PIX.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Clipping raster to SRTM size
Creating output file that is 3601P x 3601L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B4_H_WGS_PIX.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B4_H_WGS_PIX.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B4_H_WGS_PIX.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B4_H_WGS_PIX_CLIP.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Removing previous files
Changing Landsat raster with name LC08_L1TP_MERGE_B5_H.TIF
Transforming from UTM to WGS84...
Creating output file that is 14857P x 12574L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B5_H.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B5_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B5_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B5_H_WGS.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Modifying pixel size to [0.0002777777777777778, -0.0002777777777777778]
Creating output file that is 15260P x 12915L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B5_H_WGS.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B5_H_WGS.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B5_H_WGS.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B5_H_WGS_PIX.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Clipping raster to SRTM size
Creating output file that is 3601P x 3601L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B5_H_WGS_PIX.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B5_H_WGS_PIX.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B5_H_WGS_PIX.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B5_H_WGS_PIX_CLIP.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Removing previous files
Changing Landsat raster with name LC08_L1TP_MERGE_B6_H.TIF
Transforming from UTM to WGS84...
Creating output file that is 14857P x 12574L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B6_H.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B6_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B6_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B6_H_WGS.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Modifying pixel size to [0.0002777777777777778, -0.0002777777777777778]
Creating output file that is 15260P x 12915L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B6_H_WGS.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B6_H_WGS.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B6_H_WGS.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B6_H_WGS_PIX.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Clipping raster to SRTM size
Creating output file that is 3601P x 3601L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B6_H_WGS_PIX.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B6_H_WGS_PIX.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B6_H_WGS_PIX.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B6_H_WGS_PIX_CLIP.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Removing previous files
Changing Landsat raster with name LC08_L1TP_MERGE_B7_H.TIF
Transforming from UTM to WGS84...
Creating output file that is 14857P x 12574L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B7_H.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B7_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B7_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B7_H_WGS.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Modifying pixel size to [0.0002777777777777778, -0.0002777777777777778]
Creating output file that is 15260P x 12915L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B7_H_WGS.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B7_H_WGS.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B7_H_WGS.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B7_H_WGS_PIX.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Clipping raster to SRTM size
Creating output file that is 3601P x 3601L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B7_H_WGS_PIX.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B7_H_WGS_PIX.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B7_H_WGS_PIX.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B7_H_WGS_PIX_CLIP.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Removing previous files
Changing Landsat raster with name LC08_L1TP_MERGE_B8_H.TIF
Transforming from UTM to WGS84...
Creating output file that is 29712P x 25146L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B8_H.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B8_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B8_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B8_H_WGS.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Modifying pixel size to [0.0002777777777777778, -0.0002777777777777778]
Creating output file that is 15259P x 12914L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B8_H_WGS.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B8_H_WGS.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B8_H_WGS.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B8_H_WGS_PIX.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Clipping raster to SRTM size
Creating output file that is 3601P x 3601L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B8_H_WGS_PIX.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B8_H_WGS_PIX.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B8_H_WGS_PIX.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B8_H_WGS_PIX_CLIP.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Removing previous files
Changing Landsat raster with name LC08_L1TP_MERGE_B9_H.TIF
Transforming from UTM to WGS84...
Creating output file that is 14857P x 12574L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B9_H.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B9_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B9_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B9_H_WGS.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Modifying pixel size to [0.0002777777777777778, -0.0002777777777777778]
Creating output file that is 15260P x 12915L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B9_H_WGS.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B9_H_WGS.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B9_H_WGS.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B9_H_WGS_PIX.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Clipping raster to SRTM size
Creating output file that is 3601P x 3601L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B9_H_WGS_PIX.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B9_H_WGS_PIX.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B9_H_WGS_PIX.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B9_H_WGS_PIX_CLIP.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Removing previous files
Changing Landsat raster with name LC08_L1TP_MERGE_B10_H.TIF
Transforming from UTM to WGS84...
Creating output file that is 14857P x 12574L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B10_H.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B10_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B10_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B10_H_WGS.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Modifying pixel size to [0.0002777777777777778, -0.0002777777777777778]
Creating output file that is 15260P x 12915L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B10_H_WGS.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B10_H_WGS.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B10_H_WGS.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B10_H_WGS_PIX.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Clipping raster to SRTM size
Creating output file that is 3601P x 3601L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B10_H_WGS_PIX.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B10_H_WGS_PIX.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B10_H_WGS_PIX.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B10_H_WGS_PIX_CLIP.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Removing previous files
Changing Landsat raster with name LC08_L1TP_MERGE_B11_H.TIF
Transforming from UTM to WGS84...
Creating output file that is 14857P x 12574L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B11_H.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B11_H.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B11_H.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B11_H_WGS.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Modifying pixel size to [0.0002777777777777778, -0.0002777777777777778]
Creating output file that is 15260P x 12915L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B11_H_WGS.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B11_H_WGS.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B11_H_WGS.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B11_H_WGS_PIX.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Clipping raster to SRTM size
Creating output file that is 3601P x 3601L.
Processing input file ../data/raw/train/1/LC08_L1TP_MERGE_B11_H_WGS_PIX.TIF.
Using internal nodata values (e.g. -9999) for image ../data/raw/train/1/LC08_L1TP_MERGE_B11_H_WGS_PIX.TIF.
Copying nodata values from source ../data/raw/train/1/LC08_L1TP_MERGE_B11_H_WGS_PIX.TIF to destination ../data/raw/train/1/LC08_L1TP_MERGE_B11_H_WGS_PIX_CLIP.TIF.
0...10...20...30...40...50...60...70...80...90...100 - done.
Removing previous files
All the Landsat files found were SRTM-shaped
```

### FNF Shaping

The FNF file have different pixel size than the SRTM raster. Then a pixel resizing is required. The following script needs to be executed to perform the job:

```console
foo@bar:preprocessing$ python3 fnf_to_srtm_shaper.py -s ../data/raw/train/1/
Entering to the FNF to SRTM shaper
Working with SRTM file ../data/raw/train/1/SRTM_S26W053.tif
Changing FNF raster with name FNF_S25W053_17_C_F02DAR
Transforming GeoTIF file...
Input file size is 4500, 4500
0...10...20...30...40...50...60...70...80...90...100 - done.
Modifying pixel size to [0.0002777777777777778, -0.0002777777777777778]
Copying color table from ../data/raw/train/1/FNF_S25W053_17_C_F02DAR.tif to new file.
Creating output file that is 3600P x 3600L.
Processing input file ../data/raw/train/1/FNF_S25W053_17_C_F02DAR.tif.
Warning 1: Input file ../data/raw/train/1/FNF_S25W053_17_C_F02DAR.tif has a color table, which will likely lead to bad results when using a resampling method other than nearest neighbour or mode. Converting the dataset prior to 24/32 bit is advised.
0...10...20...30...40...50...60...70...80...90...100 - done.
Removing previous files
Removing hdr or tar file
The FNF file found was SRTM-shaped
```

### FNF Binarizing
The FNF raster have values for forest, no-forest and water pixels. In our project we convert the water pixels to no-forest in order to obtain a binary mask of forest/no-forest. The following script does the required job:

```console
foo@bar:preprocessing$ python3 fnf_binaryzer.py -s ../data/raw/train/1/
Trying to binaryze the FNF raster file ../data/raw/train/1/FNF_S25W053_17_C_F02DAR_PIX.tif
After accommodating values the distribution of no forest (0) and forest (1) is:  {0: 11025610, 1: 1934390}
Saving changes and storing...
All done
```
