These are the instructions to start a local neuroglancer instance to visualize predictions or ground truth data. 

## Prepare environment

Set-up a development environment with all necessary dependencies based on the [.yml file](https://github.com/danifranco/TIMISE/blob/main/examples/neuroglancer/environment.yml) in this folder:

```
conda env create -f environment.yml
conda activate py3_torch
```
## Create neuroglancer scripts

You can easily prepare a python script to run neuroglancer using TIMISE toolbox:

```
from timise import TIMISE

timise = TIMISE()
timise.evaluate(..)

timise.create_neuroglancer_file("gt", categories=['large'])
```

## Run neuroglancer

Run the prepared script to visualize the selected method and instances of it (ground truth large instances in this example). 

Following the toy example of our [notebooks](https://github.com/danifranco/TIMISE/blob/main/examples) you can run the file like this:

```
(py3_torch)$ python -iu neuroglancer_gt_large.py
load im and gt segmentation
(45, 1504, 672)
Open in you browser:
http://localhost:9999/v/32c9b156d30dea420796c120c65707ba029b51ae/
>>> 
```

When you open it you should be able to see somethign similar to this:

<p align="center">
  <img src="https://github.com/danifranco/TIMISE/blob/main/examples/neuroglancer/toy_gt.png" alt="toy_gt" width="400"/>
</p>
