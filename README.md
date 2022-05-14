# A **T**oolbox for **I**dentifying **M**itochondria **I**nstance **S**egmentation **E**rrors

# Usage

```python
from timise import TIMISE

timise = TIMISE()
timise.evaluate("/home/user/model_xx_prediction","/home/user/gt", "/home/user/output", data_resolution=[30,8,8])
timise.summarize()  # Summarize the results as tables in the console
```

Example of the tables printed in the console:

```
Stats in /home/user/analysis/output/model_xx_prediction

             Average Precision (AP)
+---------------+----------+----------+----------+
| IoU=0.50:0.95 | IoU=0.50 | IoU=0.75 | IoU=0.90 |
+---------------+----------+----------+----------+
|     0.701     |  0.887   |  0.841   |  0.294   |
+---------------+----------+----------+----------+

                                         ASSOCIATIONS
+-------+------------+---------+-------------------+--------------------+--------------+-------+
|       | one-to-one | missing | over-segmentation | under-segmentation | many-to-many | Total |
+-------+------------+---------+-------------------+--------------------+--------------+-------+
| Count |    8262    |   171   |        174        |        255         |      16      |  8878 |
|   %   |   93.06    |   1.93  |        1.96       |        2.87        |     0.18     |       |
+-------+------------+---------+-------------------+--------------------+--------------+-------+
```


