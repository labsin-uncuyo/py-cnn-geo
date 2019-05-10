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
      <td>1</td>
      <td>
        25º0'0"S - 53º0'0"W to<br>
        26º0'0"S - 52º0'0"W</td>
      <td>s26_w053_1arc_v3</td>
      <td>
        LC08_L1TP_223077_20140204_20170426_01_T1<br>
        LC08_L1TP_223078_20140204_20170426_01_T1<br>
        LC08_L1TP_222077_20140301_20170425_01_T1<br>
        LC08_L1TP_222078_20140301_20170425_01_T1
      </td>
      <td>S25W053_17_FNF_F02DAR</td>
      <td>Training</td>
      <td>
        NF: 11025610<br>
        F: 1934390
      </td>
    </tr>
    <tr>
      <td>2</td>
      <td>
        25º0'0"S - 52º0'0"W to<br>
        26º0'0"S - 51º0'0"W</td>
      <td>s26_w052_1arc_v3</td>
      <td>
        LC08_L1TP_222077_20140301_20170425_01_T1<br>
        LC08_L1TP_222078_20140301_20170425_01_T1
      </td>
      <td>S25W052_17_FNF_F02DAR</td>
      <td>Training</td>
      <td>
        NF: 6271924<br>
        F: 6688076
      </td>
    </tr>
    <tr>
      <td>3</td>
      <td>
        25º0'0"S - 51º0'0"W to<br>
        26º0'0"S - 50º0'0"W</td>
      <td>s26_w051_1arc_v3</td>
      <td>
        LC08_L1TP_221077_20140206_20170426_01_T1<br>
        LC08_L1TP_221078_20140206_20170426_01_T1
      </td>
      <td>S25W051_17_FNF_F02DAR</td>
      <td>Training</td>
      <td>
        NF: 8493976<br>
        F: 4466024
      </td>
    </tr>
    <tr>
      <td>4</td>
      <td>
        25º0'0"S - 50º0'0"W to<br>
        26º0'0"S - 49º0'0"W</td>
      <td>s26_w050_1arc_v3</td>
      <td>
        LC08_L1TP_221077_20140206_20170426_01_T1<br>
        LC08_L1TP_221078_20140206_20170426_01_T1<br>
        LC08_L1TP_220077_20140130_20170426_01_T1<br>
        LC08_L1TP_220078_20140130_20170426_01_T1
      </td>
      <td>S25W050_17_FNF_F02DAR</td>
      <td>Training</td>
      <td>
        NF: 5830415<br>
        F: 7129585
      </td>
    </tr>
    <tr>
      <td>5</td>
      <td>
        26º0'0"S - 53º0'0"W to<br>
        27º0'0"S - 52º0'0"W</td>
      <td>s27_w053_1arc_v3</td>
      <td>
        LC08_L1TP_223078_20140204_20170426_01_T1<br>
        LC08_L1TP_222078_20140301_20170425_01_T1<br>
        LC08_L1TP_222079_20140301_20170425_01_T1
      </td>
      <td>S26W053_17_FNF_F02DAR</td>
      <td>Training</td>
      <td>
        NF: 10856099<br>
        F: 2103901
      </td>
    </tr>
    <tr>
      <td>6</td>
      <td>
        26º0'0"S - 52º0'0"W to<br>
        27º0'0"S - 51º0'0"W</td>
      <td>s27_w052_1arc_v3</td>
      <td>
        LC08_L1TP_222078_20140301_20170425_01_T1<br>
        LC08_L1TP_222079_20140301_20170425_01_T1<br>
        LC08_L1TP_221078_20140206_20170426_01_T1<br>
        LC08_L1TP_221079_20140206_20170426_01_T1
      </td>
      <td>S26W052_17_FNF_F02DAR</td>
      <td>Training</td>
      <td>
        NF: 4772477<br>
        F: 8187523
      </td>
    </tr>
    <tr>
      <td>7</td>
      <td>
        26º0'0"S - 51º0'0"W to<br>
        27º0'0"S - 50º0'0"W</td>
      <td>s27_w051_1arc_v3</td>
      <td>
        LC08_L1TP_221078_20140206_20170426_01_T1<br>
        LC08_L1TP_221079_20140206_20170426_01_T1
      </td>
      <td>S26W051_17_FNF_F02DAR</td>
      <td>Training</td>
      <td>
        NF: 6068250<br>
        F: 6891750
      </td>
    </tr>
    <tr>
      <td>8</td>
      <td>
        26º0'0"S - 50º0'0"W to<br>
        27º0'0"S - 49º0'0"W</td>
      <td>s27_w050_1arc_v3</td>
      <td>
        LC08_L1TP_221078_20140206_20170426_01_T1<br>
        LC08_L1TP_221079_20140206_20170426_01_T1<br>
        LC08_L1TP_220078_20140130_20170426_01_T1<br>
        LC08_L1TP_220079_20140130_20170426_01_T1
      </td>
      <td>S26W050_17_FNF_F02DAR</td>
      <td>Training</td>
      <td>
        NF: 4066061<br>
        F: 8893939
      </td>
    </tr>
    <tr>
      <td>9</td>
      <td>
        27º0'0"S - 53º0'0"W to<br>
        28º0'0"S - 52º0'0"W</td>
      <td>s28_w053_1arc_v3</td>
      <td>
        LC08_L1TP_222079_20140301_20170425_01_T1
      </td>
      <td>S27W053_17_FNF_F02DAR</td>
      <td>Training</td>
      <td>
        NF: 10928487<br>
        F: 2031513
      </td>
    </tr>
    <tr>
      <td>10</td>
      <td>
        27º0'0"S - 52º0'0"W to<br>
        28º0'0"S - 51º0'0"W</td>
      <td>s28_w052_1arc_v3</td>
      <td>
        LC08_L1TP_222079_20140301_20170425_01_T1<br>
        LC08_L1TP_221079_20140206_20170426_01_T1
      </td>
      <td>S27W052_17_FNF_F02DAR</td>
      <td>Training</td>
      <td>
        NF: 9216286<br>
        F: 3743714
      </td>
    </tr>
    <tr>
      <td>11</td>
      <td>
        27º0'0"S - 51º0'0"W to<br>
        28º0'0"S - 50º0'0"W</td>
      <td>s28_w051_1arc_v3</td>
      <td>
        LC08_L1TP_221079_20140206_20170426_01_T1
      </td>
      <td>S27W051_17_FNF_F02DAR</td>
      <td>Training</td>
      <td>
        NF: 6724179<br>
        F: 6235821
      </td>
    </tr>
    <tr>
      <td>12</td>
      <td>
        27º0'0"S - 50º0'0"W to<br>
        28º0'0"S - 49º0'0"W</td>
      <td>s28_w050_1arc_v3</td>
      <td>
        LC08_L1TP_221079_20140206_20170426_01_T1<br>
        LC08_L1TP_220079_20140130_20170426_01_T1
      </td>
      <td>S27W050_17_FNF_F02DAR</td>
      <td>Training</td>
      <td>
        NF: 5403650<br>
        F: 7556350
      </td>
    </tr>
    <tr>
      <td>13</td>
      <td>
        28º0'0"S - 51º0'0"W to<br>
        29º0'0"S - 50º0'0"W</td>
      <td>s29_w051_1arc_v3</td>
      <td>
        LC08_L1TP_222080_20140301_20170425_01_T1<br>
        LC08_L1TP_221079_20140206_20170426_01_T1<br>
        LC08_L1TP_220080_20140130_20170426_01_T1
      </td>
      <td>S28W051_17_FNF_F02DAR</td>
      <td>Testing</td>
      <td>
        NF: 8546237<br>
        F: 4413763
      </td>
    </tr>
  </tbody>
</table>
