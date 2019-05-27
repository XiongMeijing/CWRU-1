import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from pathlib import Path
save_model_path = Path("./Model")
import scipy.io
import nn_model

import numpy as np
import pandas as pd

class _Model:
    """
    Base class for Model which stores read signal files and stores states.
    Inherit this class, extend the init method to load your serialized ml model 
    and implement the predict method.
    """
    def __init__(self):
        self.data = {}
        self.data['filenames'] = []
        self.data['filepaths'] = []
        self.data['signals'] = []
        self.data['prediction'] = []
        
        self._pred2class_map = {
            0:'N',
            1:'B',
            2:'IR',
            3:'OR',
        }

    def update_prediction(self, file_index):
        """
        Get prediction of the file_index coresponding signal and update the data
        prediction state of this signal.
        """
        prediction = self.predict(file_index)
        self.data['prediction'][file_index] = self._pred2class_map[prediction]
    
    def predict(self, file_index):
        """
        Implement this method in subclass to return the prediction by your ml
        model as Python object, e.g. int, str.
        """
        print("Not implemented")

    def read_files(self, filepaths):
        """
        Read the .mat signal files and update the states of data
        
        Parameter:
            filepaths: List/tuple of .mat signal files' paths
        """
        for filepath in filepaths:
            file_name = str(filepath).split('/')[-1]
            self.data['filenames'].append(file_name)
            self.data['filepaths'].append(Path(filepath))
            self.data['signals'].append(mat_to_ndarray(Path(filepath)))
            self.data['prediction'].append('None')
        return self.data['filenames']


class CNN_1D(_Model):
    def __init__(self):
        super().__init__()
        # Instantiate the CNN_1D_2L with input size=500 and load the trained
        # model parameters
        self.pred_model = nn_model.CNN_1D_2L(500)
        self.pred_model.load_state_dict(
            torch.load(save_model_path / 'model.pth', map_location='cpu')
            )
        self.pred_model.eval()

    def predict(self, file_index):
        """
        Split the signal data into segments of 500 points (same preprocessing
        during training). Then predict the class of all segments and return the
        most frequent class (mode) as output.
        """
        x = self.data['signals'][file_index]
        x = preprocess_signal(x, 500)
        x = torch.tensor(x, dtype=torch.float32)
        out = self.pred_model(x)
        pred = torch.argmax(out, dim=1)
        mode, _ = torch.mode(pred, dim=0)
        return mode.item()


def rename_matfile_keys(dic):
    '''
    Rename some keys so that they can be loaded into a DataFrame with 
    consistent column names
    '''
    # For each key-value pair, rename the following keys 
    for k2,_ in list(dic.items()):
        if 'DE_time' in k2:
            dic['DE_time'] = dic.pop(k2)
        elif 'BA_time' in k2:
            dic['BA_time'] = dic.pop(k2)
        elif 'FE_time' in k2:
            dic['FE_time'] = dic.pop(k2)
        elif 'RPM' in k2:
            dic['RPM'] = dic.pop(k2)

def mat_to_ndarray(matfile_path):
    """
    Read the .mat signal file and return the drive end signal data (DE_time).
    """
    matfile_dic = scipy.io.loadmat(matfile_path)
    rename_matfile_keys(matfile_dic)
    return matfile_dic['DE_time']

def preprocess_signal(array, segment_length):
    """
    Preprocess the raw signal data by normalizing and divide the signal into
    smaller segments with size of segment_length. Segment length must be the 
    same as the value used in training.
    
    Parameter:
        array: array
        segment_length: int
    """
    array = normalize_signal_array(array)
    return divide_signal_array(array, segment_length)

def normalize_signal_array(array):
    """ Normalize the signal array"""
    mean = np.mean(array)
    std = np.std(array)
    return (array - mean) / std

def divide_signal_array(array, segment_length):
    '''
    This function divide the signal into segments, each with a specific number of points as
    defined by segment_length. Each segment will be added as an example (a row) in the 
    returned DataFrame. Thus it increases the number of training examples. The remaining 
    points which are less than segment_length are discarded.
    
    Parameter:
        array: 
            Numpy array which contains the signal sample points
        segment_length: 
            Number of points per segment.
    Return:
        DataFrame with segmented signals and their corresponding filename
    '''
    dic = {}
    idx = 0
    n_sample_points = len(array)
    n_segments = n_sample_points // segment_length
    for segment in range(n_segments):
        dic[idx] = {
            'signal': array[segment_length * segment:segment_length * (segment+1)], 
        }
        idx += 1
    df_tmp = pd.DataFrame.from_dict(dic,orient='index')
    return np.hstack(df_tmp["signal"].values).T
