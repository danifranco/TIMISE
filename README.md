# A **T**oolbox for **I**dentifying **M**itochondria **I**nstance **S**egmentation **E**rrors

## Installation
Create a conda environment with all the dependencies:
```shell
conda env create -f environment.yml
conda activate timise
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

# Plot errors (2D plot by default) and save in '/home/user/output/model_xx_prediction_folder'
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
|     0.701     |  0.887   |  0.844   |  0.294   |
+---------------+----------+----------+----------+

                                          Associations
+-------+------------+---------+-------------------+--------------------+--------------+-------+
|       | one-to-one | missing | over-segmentation | under-segmentation | many-to-many | Total |
+-------+------------+---------+-------------------+--------------------+--------------+-------+
| Count |    8215    |   168   |        320        |        158         |      17      |  8878 |
|   %   |   92.53    |   1.89  |        3.6        |        1.78        |     0.19     |  8878 |
+-------+------------+---------+-------------------+--------------------+--------------+-------+

                                                                     Matching metrics
+--------+--------+--------+-------+-----------+--------+----------+-------+--------+---------+-----------------+--------------------+------------------+
| thresh |   fp   |   tp   |   fn  | precision | recall | accuracy |   f1  | n_true |  n_pred | mean_true_score | mean_matched_score | panoptic_quality |
+--------+--------+--------+-------+-----------+--------+----------+-------+--------+---------+-----------------+--------------------+------------------+
|  0.3   | 1932.0 | 8538.0 | 340.0 |   0.815   | 0.962  |   0.79   | 0.883 | 8878.0 | 10470.0 |      0.848      |       0.881        |      0.778       |
|  0.5   | 2001.0 | 8469.0 | 409.0 |   0.809   | 0.954  |  0.778   | 0.875 | 8878.0 | 10470.0 |      0.844      |       0.885        |      0.775       |
|  0.75  | 2257.0 | 8213.0 | 665.0 |   0.784   | 0.925  |  0.738   | 0.849 | 8878.0 | 10470.0 |      0.826      |       0.893        |      0.758       |
+--------+--------+--------+-------+-----------+--------+----------+-------+--------+---------+-----------------+--------------------+------------------+
```

And a summary plot for your model's errors:

<p align="center">
  <img src="https://github.com/danifranco/TIMISE/blob/main/examples/img/plot_error_example.png" alt="summary_plot" width="500"/>
</p>

## Details
Two different workflows are implemented:
- When no more folders are found inside the input path, e.g. ``/home/user/model_xx_prediction_folder`` in this example, the file ``.h5`` or ``.tif`` inside that folder will be evaluated against gt file in ``/home/user/gt_folder``.
- When more folders are found inside the input path e.g. ``/home/user/model_xx_prediction_folder`` in this example. In this case every folder will be processed as if it were a method to be evaluated, so each folder must contain its own ``.h5``/``.tif`` file. This option is usefull when multiple models' predictions need to be evaluated at once. Apart for individual plots this workflow also allows the creation of a general plot gathering the results of all methods.

## Understanding output folder files
These are the files that the toolbox will create:

```shell
/home/user/output_folder/model_xx_prediction_folder/
├─ model_xx_prediction_folder/
│  ├─ associations.csv
│  ├─ associations_stats.csv
│  ├─ gt_final.csv
│  ├─ map.csv
│  ├─ map_map.txt
│  ├─ matching_metrics.csv
│  ├─ prediction_stats.csv
│  ├─ target_daughter_matching_file.csv
│  ├─ target_mother_matching_file.csv
```

- ``associations.csv``: associations between predicted and gt instances. Created when associations metrics are computed. 
- ``associations_stats.csv``: summary of the associations to print them easily. Created when associations metrics are computed. 
- ``gt_final.csv``: gt statistics mixed with the association errors. Used to generate the final plots easily. Created when associations metrics are computed. 
- ``map.csv``: matching stats between prediction and gt instances. Created when mAP metric is computed.  
- ``map_map.txt``: mAP auxiliary file. Created when mAP metric is computed. 
- ``matching_metrics.csv``: summary of the matching metrics to print them easily. Created when matching metrics are computed.  
- ``prediction_stats.csv``: statistics of predicted instances. Created when prediction statistics are computed.  
- ``target_daughter_matching_file.csv``: auxiliary matching file used for association metrics. Created when associations metrics are computed. 
- ``prediction_statarget_mother_matching_filets.csv``: auxiliary matching file used for association metrics. Created when associations metrics are computed. 

## Jupyter Notebook
Check out the jupyter notebooks in [examples folder](https://github.com/danifranco/TIMISE/blob/main/examples) for more details.

## Citation
Under construction . . .
