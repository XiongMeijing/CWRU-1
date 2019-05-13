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


## Overview
1D CNN has been sucessfully applied to fault classification based on signal data in some papers 
(e.g. [[1]](https://doi.org/10.1155/2017/8617315), 
[[2]](https://www.researchgate.net/publication/304550799_Real-Time_Motor_Fault_Detection_by_1D_Convolutional_Neural_Networks)). 
The main advantage of using a 1D CNN is that manual feature extraction like spectrum analysis, statistical features and so on is not
required. After normalization, the signal data can be directly feed into the 1D CNN for training.

In this repo, simple two layers and 3 layers 1D CNN are experimented and achieved similar performance 
(99%+ accuracy) as recent papers, as summarized by [[3]](https://arxiv.org/pdf/1901.08247.pdf).

Helper functions for data cleaning and preprocessing are written in the `helper.py` module, whereas helper functions for training using Pytorch Framework are written in the `train_helper.py` module.

The notebook `CWRU_Dataset.ipynb` shows the training process and the trained model is saved in the `./Model` folder.
