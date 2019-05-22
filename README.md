# py-cnn-geo
CNN architecture for Forest-No Forest mask generation based on SRTM and Landsat-8 information.

## Dataset
Bellow there's a table describing the different files involved in the contruction of each zone of the dataset, the coverage, number of points for each class, and usage of the zone (training or testing).

<table>
  <thead>
    <tr>
      <th>Zone</th>
      <th>Coverage</th>
      <th>SRTM file</th>
      <th>Landsat-8 files</th>
      <th>FNF file</th>
      <th>Usage</th>
      <th>F-NF Points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><sub>1</sub></td>
      <td><sub>
        25º0'0"S - 53º0'0"W to<br>
        26º0'0"S - 52º0'0"W</sub></td>
      <td><sub>s26_w053_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_223077_20140204_20170426_01_T1<br>
        LC08_L1TP_223078_20140204_20170426_01_T1<br>
        LC08_L1TP_222077_20140301_20170425_01_T1<br>
        LC08_L1TP_222078_20140301_20170425_01_T1
      </sub></td>
      <td><sub>S25W053_17_FNF_F02DAR</sub></td>
      <td><sub>Training</sub></td>
      <td><sub>
        NF: 11025610<br>
        F: 1934390
      </sub></td>
    </tr>
    <tr>
      <td><sub>2</sub></td>
      <td><sub>
        25º0'0"S - 52º0'0"W to<br>
        26º0'0"S - 51º0'0"W</sub></td>
      <td><sub>s26_w052_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_222077_20140301_20170425_01_T1<br>
        LC08_L1TP_222078_20140301_20170425_01_T1
      </sub></td>
      <td><sub>S25W052_17_FNF_F02DAR</sub></td>
      <td><sub>Training</sub></td>
      <td><sub>
        NF: 6271924<br>
        F: 6688076
      </sub></td>
    </tr>
    <tr>
      <td><sub>3</sub></td>
      <td><sub>
        25º0'0"S - 51º0'0"W to<br>
        26º0'0"S - 50º0'0"W</sub></td>
      <td><sub>s26_w051_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_221077_20140206_20170426_01_T1<br>
        LC08_L1TP_221078_20140206_20170426_01_T1
      </sub></td>
      <td><sub>S25W051_17_FNF_F02DAR</sub></td>
      <td><sub>Training</sub></td>
      <td><sub>
        NF: 8493976<br>
        F: 4466024
      </sub></td>
    </tr>
    <tr>
      <td><sub>4</sub></td>
      <td><sub>
        25º0'0"S - 50º0'0"W to<br>
        26º0'0"S - 49º0'0"W</sub></td>
      <td><sub>s26_w050_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_221077_20140206_20170426_01_T1<br>
        LC08_L1TP_221078_20140206_20170426_01_T1<br>
        LC08_L1TP_220077_20140130_20170426_01_T1<br>
        LC08_L1TP_220078_20140130_20170426_01_T1
      </sub></td>
      <td><sub>S25W050_17_FNF_F02DAR</sub></td>
      <td><sub>Training</sub></td>
      <td><sub>
        NF: 5830415<br>
        F: 7129585
      </sub></td>
    </tr>
    <tr>
      <td><sub>5</sub></td>
      <td><sub>
        26º0'0"S - 53º0'0"W to<br>
        27º0'0"S - 52º0'0"W</sub></td>
      <td><sub>s27_w053_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_223078_20140204_20170426_01_T1<br>
        LC08_L1TP_222078_20140301_20170425_01_T1<br>
        LC08_L1TP_222079_20140301_20170425_01_T1
      </sub></td>
      <td><sub>S26W053_17_FNF_F02DAR</sub></td>
      <td><sub>Training</sub></td>
      <td><sub>
        NF: 10856099<br>
        F: 2103901
      </sub></td>
    </tr>
    <tr>
      <td><sub>6</sub></td>
      <td><sub>
        26º0'0"S - 52º0'0"W to<br>
        27º0'0"S - 51º0'0"W</sub></td>
      <td><sub>s27_w052_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_222078_20140301_20170425_01_T1<br>
        LC08_L1TP_222079_20140301_20170425_01_T1<br>
        LC08_L1TP_221078_20140206_20170426_01_T1<br>
        LC08_L1TP_221079_20140206_20170426_01_T1
      </sub></td>
      <td><sub>S26W052_17_FNF_F02DAR</sub></td>
      <td><sub>Training</sub></td>
      <td><sub>
        NF: 4772477<br>
        F: 8187523
      </sub></td>
    </tr>
    <tr>
      <td><sub>7</sub></td>
      <td><sub>
        26º0'0"S - 51º0'0"W to<br>
        27º0'0"S - 50º0'0"W</sub></td>
      <td><sub>s27_w051_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_221078_20140206_20170426_01_T1<br>
        LC08_L1TP_221079_20140206_20170426_01_T1
      </sub></td>
      <td><sub>S26W051_17_FNF_F02DAR</sub></td>
      <td><sub>Training</sub></td>
      <td><sub>
        NF: 6068250<br>
        F: 6891750
      </sub></td>
    </tr>
    <tr>
      <td><sub>8</sub></td>
      <td><sub>
        26º0'0"S - 50º0'0"W to<br>
        27º0'0"S - 49º0'0"W</sub></td>
      <td><sub>s27_w050_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_221078_20140206_20170426_01_T1<br>
        LC08_L1TP_221079_20140206_20170426_01_T1<br>
        LC08_L1TP_220078_20140130_20170426_01_T1<br>
        LC08_L1TP_220079_20140130_20170426_01_T1
      </sub></td>
      <td><sub>S26W050_17_FNF_F02DAR</sub></td>
      <td><sub>Training</sub></td>
      <td><sub>
        NF: 4066061<br>
        F: 8893939
      </sub></td>
    </tr>
    <tr>
      <td><sub>9</sub></td>
      <td><sub>
        27º0'0"S - 53º0'0"W to<br>
        28º0'0"S - 52º0'0"W</sub></td>
      <td><sub>s28_w053_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_222079_20140301_20170425_01_T1
      </sub></td>
      <td><sub>S27W053_17_FNF_F02DAR</sub></td>
      <td><sub>Training</sub></td>
      <td><sub>
        NF: 10928487<br>
        F: 2031513
      </sub></td>
    </tr>
    <tr>
      <td><sub>10</sub></td>
      <td><sub>
        27º0'0"S - 52º0'0"W to<br>
        28º0'0"S - 51º0'0"W</sub></td>
      <td><sub>s28_w052_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_222079_20140301_20170425_01_T1<br>
        LC08_L1TP_221079_20140206_20170426_01_T1
      </sub></td>
      <td><sub>S27W052_17_FNF_F02DAR</sub></td>
      <td><sub>Training</sub></td>
      <td><sub>
        NF: 9216286<br>
        F: 3743714
      </sub></td>
    </tr>
    <tr>
      <td><sub>11</sub></td>
      <td><sub>
        27º0'0"S - 51º0'0"W to<br>
        28º0'0"S - 50º0'0"W</sub></td>
      <td><sub>s28_w051_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_221079_20140206_20170426_01_T1
      </sub></td>
      <td><sub>S27W051_17_FNF_F02DAR</sub></td>
      <td><sub>Training</sub></td>
      <td><sub>
        NF: 6724179<br>
        F: 6235821
      </sub></td>
    </tr>
    <tr>
      <td><sub>12</sub></td>
      <td><sub>
        27º0'0"S - 50º0'0"W to<br>
        28º0'0"S - 49º0'0"W</sub></td>
      <td><sub>s28_w050_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_221079_20140206_20170426_01_T1<br>
        LC08_L1TP_220079_20140130_20170426_01_T1
      </sub></td>
      <td><sub>S27W050_17_FNF_F02DAR</sub></td>
      <td><sub>Training</sub></td>
      <td><sub>
        NF: 5403650<br>
        F: 7556350
      </sub></td>
    </tr>
    <tr>
      <td><sub>13</sub></td>
      <td><sub>
        28º0'0"S - 51º0'0"W to<br>
        29º0'0"S - 50º0'0"W</sub></td>
      <td><sub>s29_w051_1arc_v3</sub></td>
      <td><sub>
        LC08_L1TP_222080_20140301_20170425_01_T1<br>
        LC08_L1TP_221079_20140206_20170426_01_T1<br>
        LC08_L1TP_220080_20140130_20170426_01_T1
      </sub></td>
      <td><sub>S28W051_17_FNF_F02DAR</sub></td>
      <td><sub>Testing</sub></td>
      <td><sub>
        NF: 8546237<br>
        F: 4413763
      </sub></td>
    </tr>
  </tbody>
</table>

## Preprocessing
Preprocessing steps can be found in the following link:

https://github.com/labsin-uncuyo/py-cnn-geo/tree/master/src/preprocessing

## Dataset generation

To create the data-cubes required for training, an algorithm was created to randomly pick different locations of forest and no-forest points in the zones and conform the following processes and files:

- A processed folder is created in the data directory where the training and testing processed data is stored.
- For both, training and testing data, each zone is processed creating a file with the input cube of data composed by the preprocessed SRTM raster and Landsat 8 bands, and a gt matrix containing the labels (forest or no-forest) for each pixel in the input data cube. This data is stored in a file with name dataset.npz. Also, an index file is created were the locations of the different pixels for each label is stored. This data is stored in the file idxs.npz.
- For training data only, another file is created containing a random selection of locations from the index files of all the training zones. This selection is limited by a percentage of data to be retrieved, so the percentage of the data to be used for training can be defined, but assuring a random selection of the training samples. This information is stored in a file with name samples_shuffled_factor_idx.npz

The script to perform this task is executed by the following command:

- For train:

´´´ console
python3 balanced_factor_indexer.py -s ../data/raw/train/ -t upsample
Preparing for balanced downsampler indexer by factor
Working with dataset folder ../data/raw/train/
Folders to work with:  ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
Checking number of indexes...
All folders were loaded
Number of indexes for No Forest: 89657415
Number of indexes for Forest: 65862587
Copying and appending index values...
Shuffling No Forest indexes...
Shuffling Forest indexes...
Storing data...
Done!
´´´

For test:

´´´ console
python3 balanced_factor_indexer.py -s ../data/raw/test/ -x
Preparing for balanced downsampler indexer by factor
Working with dataset folder ../data/raw/test/
Folders to work with:  ['13']
All folders were loaded
Done!
´´´
