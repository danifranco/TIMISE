# A **T**oolbox for **I**dentifying **M**itochondria **I**nstance **S**egmentation **E**rrors

The toolbox provides a set of size and morphological measures from the prediction and ground truth, including volume,
skeleton length and cable length; and a set of evaluation metrics, including the AP used from MitoEM, association
and matching metrics. Thus, you could analize the errors between these two volumes:

<table>
  <tr>
     <td>Ground Truth</td>
     <td>Prediction</td>
  </tr>
  <tr>
    <td><img src="https://github.com/danifranco/TIMISE/blob/main/examples/img/toy_gt.gif" width=200></td>
    <td><img src="https://github.com/danifranco/TIMISE/blob/main/examples/img/toy_pred.gif" width=200 ></td>
  </tr>
</table>

## Installation
Create a conda environment with all the dependencies:
```shell
conda create -n timise python=3.9
pip install -r requirements.txt
```

## Usage

```python
from timise import TIMISE

timise = TIMISE()

# Compute metrics
timise.evaluate("/home/user/model_xx_prediction_folder","/home/user/gt_folder", 
                "/home/user/output_folder", data_resolution=[30,8,8])

# Summarize the results as tables in the console
timise.summarize() 

# Plot errors and save in '/home/user/output'
timise.plot() 

# You can also create neuroglancer scripts to visualize them easily. 
# More info in 'examples/neuroglancer' folder
timise.create_neuroglancer_file("gt", categories=['large'])
```

Example of the tables printed in the console:

```
Stats in /home/user/analysis/output/model_xx_prediction


              Average Precision (AP)              
+---------------+----------+----------+----------+
| IoU=0.50:0.95 | IoU=0.50 | IoU=0.75 | IoU=0.90 |
+---------------+----------+----------+----------+
|     0.087     |  0.369   |  0.035   |   0.0    |
+---------------+----------+----------+----------+

                                                                                                                   Prediction   
                                         Ground truth associations                                              false positives 
+----------+-------+------------+---------+-------------------+--------------------+--------------+-------+------+------------+
| category |       | one-to-one | missing | over-segmentation | under-segmentation | many-to-many | Total |      | background |
+----------+-------+------------+---------+-------------------+--------------------+--------------+-------+------+------------+
|  small   | Count |     37     |    7    |         5         |         1          |      0       |   50  |      |     25     |
|          |   %   |    74.0    |   14.0  |        10.0       |        2.0         |     0.0      |       |      |            |
|  medium  | Count |     29     |    0    |         3         |         3          |      0       |   35  |      |     -      |
|          |   %   |   82.86    |   0.0   |        8.57       |        8.57        |     0.0      |       |      |            |
|  large   | Count |     13     |    0    |         5         |         1          |      0       |   19  |      |     -      |
|          |   %   |   68.42    |   0.0   |       26.32       |        5.26        |     0.0      |       |      |            |
|          |       |            |         |                   |                    |              |       |      |            |
|  TOTAL   | Count |     79     |    7    |         13        |         5          |      0       |  104  |      |     25     |
|          |   %   |   75.96    |   6.73  |        12.5       |        4.81        |     0.0      |  100  |      |   16.03    |
+----------+-------+------------+---------+-------------------+--------------------+--------------+-------+------+------------+

                                                                     Matching metrics                                                                    
+----------+--------+-----+----+----+-----------+--------+----------+-------+--------+--------+-----------------+--------------------+------------------+
| category | thresh |  fp | tp | fn | precision | recall | accuracy |   f1  | n_true | n_pred | mean_true_score | mean_matched_score | panoptic_quality |
+----------+--------+-----+----+----+-----------+--------+----------+-------+--------+--------+-----------------+--------------------+------------------+
|  small   |  0.3   |  41 | 36 | 14 |   0.468   |  0.72  |  0.396   | 0.567 |   50   |   77   |      0.362      |       0.503        |      0.285       |
|  small   |  0.5   |  57 | 20 | 30 |    0.26   |  0.4   |  0.187   | 0.315 |   50   |   77   |      0.233      |       0.582        |      0.183       |
|  small   |  0.75  |  77 | 0  | 50 |    0.0    |  0.0   |   0.0    |  0.0  |   50   |   77   |       0.0       |        0.0         |       0.0        |
|  medium  |  0.3   |  8  | 30 | 5  |   0.789   | 0.857  |  0.698   | 0.822 |   35   |   38   |       0.49      |       0.571        |       0.47       |
|  medium  |  0.5   |  16 | 22 | 13 |   0.579   | 0.629  |  0.431   | 0.603 |   35   |   38   |      0.391      |       0.622        |      0.375       |
|  medium  |  0.75  |  34 | 4  | 31 |   0.105   | 0.114  |  0.058   |  0.11 |   35   |   38   |      0.089      |       0.783        |      0.086       |
|  large   |  0.3   |  24 | 17 | 2  |   0.415   | 0.895  |  0.395   | 0.567 |   19   |   41   |      0.495      |       0.554        |      0.314       |
|  large   |  0.5   |  30 | 11 | 8  |   0.268   | 0.579  |  0.224   | 0.367 |   19   |   41   |      0.359      |        0.62        |      0.227       |
|  large   |  0.75  |  40 | 1  | 18 |   0.024   | 0.053  |  0.017   | 0.033 |   19   |   41   |      0.044      |       0.829        |      0.028       |
|  total   |  0.3   |  73 | 83 | 21 |   0.532   | 0.798  |  0.469   | 0.638 |  104   |  156   |      0.429      |       0.537        |      0.343       |
|  total   |  0.5   | 103 | 53 | 51 |    0.34   |  0.51  |  0.256   | 0.408 |  104   |  156   |      0.309      |       0.607        |      0.247       |
|  total   |  0.75  | 151 | 5  | 99 |   0.032   | 0.048  |   0.02   | 0.038 |  104   |  156   |      0.038      |       0.792        |       0.03       |
+----------+--------+-----+----+----+-----------+--------+----------+-------+--------+--------+-----------------+--------------------+------------------+
```

And a summary plot for your model's errors:                                                                                                                                  

<p align="center">
  <img src="https://github.com/danifranco/TIMISE/blob/main/examples/img/toy_summary.png" alt="summary_plot" width="500"/>
</p>

## Details
Two different workflows are implemented:
- When no more folders are found inside the input path, e.g. ``/home/user/model_xx_prediction_folder`` in this example, the file ``.h5`` or ``.tif`` inside that folder will be evaluated against gt file in ``/home/user/gt_folder``. Find an example in this notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danifranco/TIMISE/blob/main/examples/TIMISE_one_method_evaluation_example.ipynb)

- When more folders are found inside the input path e.g. ``/home/user/model_xx_prediction_folder`` in this example. In this case every folder will be processed as if it were a method to be evaluated, so each folder must contain its own ``.h5``/``.tif`` file. This option is usefull when multiple models' predictions need to be evaluated at once. Apart for individual plots this workflow also allows the creation of a general plot gathering the results of all methods. Find an example in this notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danifranco/TIMISE/blob/main/examples/TIMISE_multiple_method_evaluation_example.ipynb)

## Jupyter Notebook
Check out the jupyter notebooks in [examples folder](https://github.com/danifranco/TIMISE/blob/main/examples) for every type of plot than can be generated with TIMISE for the two workflows described above. Please note that some graphics do not display correctly on Github, so we suggest you open the Colab links above to see them correctly. 

## Understanding output folder files
These are the files that the toolbox will create:

```shell
/home/user/output_folder/model_xx_prediction_folder/
├─ gt_group_file.txt (file)
├─ gt_stats.csv (file)
├─ *.svg (files) 
└─ model_xx_prediction_folder/
   ├─ associations.csv (file)
   ├─ associations_stats.csv (file)
   ├─ gt_final.csv (file)
   ├─ map_map.txt (file)
   |─ map_match_fn.txt (file)
   ├─ map_match_p.txt (file)
   ├─ matching_metrics.csv (file)
   ├─ prediction_stats.csv (file)
   ├─ pred_gt_matching_info.csv (file)
   ├─ pred_group_file.txt (file)
   ├─ map_aux_files(folder)
   └─ plots (folder)
```
- ``gt_group_file.txt``: each GT instance category database. Currently done by skeleton length.
- ``gt_stats.csv``: statistics of GT instances. 
- ``*.svg``: optional .svg plots generated when multiple methods are available.
- ``associations.csv``: associations between predicted and gt instances.
- ``associations_stats.csv``: summary of the associations to print them easily.
- ``gt_final.csv``: gt statistics mixed with the association errors. Used to generate the final plots easily.
- ``map_map.txt``: mAP results.
- ``map_match_fn.txt``: rest of matchings between prediction and GT. False negatives are here.
- ``map_match_p.txt``: IoU matching between prediction and GT (per category). False positives are here too.
- ``matching_metrics.csv``: summary of the matching metrics to print them easily.
- ``pred_group_file.txt``: each prediction instance category database. Currently done by skeleton length.
- ``pred_gt_matching_info.csv``: IoU matching between prediction and GT as in map_match_p.txt but in csv format so it can be loaded easily into a dataframe. 
- ``prediction_stats.csv``: statistics of predicted instances.   
- ``map_aux_files``: directory where some auxiliary files of the metrics evaluation will be store so the calculation goes faster.
- ``plots``: folder where all the plots are stored.

## Citation
Under construction . . .
