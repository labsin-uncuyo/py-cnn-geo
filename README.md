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

