# CWRU Bearing Dataset
## Introduction
This dataset is commonly used as benchmark for bearing fault classification algorithm 
[(Link)](https://csegroups.case.edu/bearingdatacenter/home), which contains vibration signal data of normal and fault bearings. 
Vibration signals of 4 classes of bearings were measured in the experiment, namely:
- normal bearing without fault (N)
- bearing with single point fault at the inner raceway (IR)
- bearing with single point fault at the outer raceway (OR)
- bearing with single point fault at the ball (B). 

The faults of different diameters (0.007 ~ 0.028 Inches) are manufactured to the bearings artificially.

In the experiment, vibration signal data was recorded using accelerometers at the drive 
end (DE) and fan end (FE) and the data is stored as Matlab files. The sampling rate is 12 kHz and each Matlab file contains 
between ~120k to ~240k sample points. For more information please refer to the 
[website](https://csegroups.case.edu/bearingdatacenter/home). 

This study focuses on the classification of the **Drive End (DE)** bearing defects using only the signal data at **DE**. 
It is a **multiclass classification** problem. The input is the vibration signal data at DE and the output is the type of defects, 
i.e. **Normal, IR, OR, B**.
