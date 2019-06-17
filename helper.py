# Helper functions to read and preprocess data files from Matlab format
# Data science libraries
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Others
from pathlib import Path
from tqdm.auto import tqdm
import requests

def matfile_to_dic(folder_path):
    '''
    Read all the matlab files of the CWRU Bearing Dataset and return a 
    dictionary. The key of each item is the filename and the value is the data 
    of one matlab file, which also has key value pairs.
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
    Return:
        output_dic: 
            Dictionary which contains data of all files in the folder_path.
    '''
    output_dic = {}
    for _, filepath in enumerate(folder_path.glob('*.mat')):
        # strip the folder path and get the filename only.
        key_name = str(filepath).split('\\')[-1]
        output_dic[key_name] = scipy.io.loadmat(filepath)
    return output_dic


def remove_dic_items(dic):
    '''
    Remove redundant data in the dictionary returned by matfile_to_dic inplace.
    '''
    # For each file in the dictionary, delete the redundant key-value pairs
    for _, values in dic.items():
        del values['__header__']
        del values['__version__']    
        del values['__globals__']


def rename_keys(dic):
    '''
    Rename some keys so that they can be loaded into a 
    DataFrame with consistent column names
    '''
    # For each file in the dictionary
    for _,v1 in dic.items():
        # For each key-value pair, rename the following keys 
        for k2,_ in list(v1.items()):
            if 'DE_time' in k2:
                v1['DE_time'] = v1.pop(k2)
            elif 'BA_time' in k2:
                v1['BA_time'] = v1.pop(k2)
            elif 'FE_time' in k2:
                v1['FE_time'] = v1.pop(k2)
            elif 'RPM' in k2:
                v1['RPM'] = v1.pop(k2)


def label(filename):
    '''
    Function to create label for each signal based on the filename. Apply this
    to the "filename" column of the DataFrame.
    Usage:
        df['label'] = df['filename'].apply(label)
    '''
    if 'B' in filename:
        return 'B'
    elif 'IR' in filename:
        return 'IR'
    elif 'OR' in filename:
        return 'OR'
    elif 'Normal' in filename:
        return 'N'


def matfile_to_df(folder_path):
    '''
    Read all the matlab files in the folder, preprocess, and return a DataFrame
    
    Parameter:
        folder_path: 
            Path (Path object) of the folder which contains the matlab files.
    Return:
        DataFrame with preprocessed data
    '''
    dic = matfile_to_dic(folder_path)
    remove_dic_items(dic)
    rename_keys(dic)
    df = pd.DataFrame.from_dict(dic).T
    df = df.reset_index().rename(mapper={'index':'filename'},axis=1)
    df['label'] = df['filename'].apply(label)
    return df.drop(['BA_time','FE_time', 'RPM', 'ans'], axis=1, errors='ignore')


def divide_signal(df, segment_length):
    '''
    This function divide the signal into segments, each with a specific number 
    of points as defined by segment_length. Each segment will be added as an 
    example (a row) in the returned DataFrame. Thus it increases the number of 
    training examples. The remaining points which are less than segment_length 
    are discarded.
    
    Parameter:
        df: 
            DataFrame returned by matfile_to_df()
        segment_length: 
            Number of points per segment.
    Return:
        DataFrame with segmented signals and their corresponding filename and 
        label
    '''
    dic = {}
    idx = 0
    for i in range(df.shape[0]):
        n_sample_points = len(df.iloc[i,1])
        n_segments = n_sample_points // segment_length
        for segment in range(n_segments):
            dic[idx] = {
                'signal': df.iloc[i,1][segment_length * segment:segment_length * (segment+1)], 
                'label': df.iloc[i,2],
                'filename' : df.iloc[i,0]
            }
            idx += 1
    df_tmp = pd.DataFrame.from_dict(dic,orient='index')
    df_output = pd.concat(
        [df_tmp[['label', 'filename']], 
         pd.DataFrame(np.hstack(df_tmp["signal"].values).T)
        ], 
        axis=1 )
    return df_output


def normalize_signal(df):
    '''
    Normalize the signals in the DataFrame returned by matfile_to_df() by subtracting
    the mean and dividing by the standard deviation.
    '''
    mean = df['DE_time'].apply(np.mean)
    std = df['DE_time'].apply(np.std)
    df['DE_time'] = (df['DE_time'] - mean) / std


def get_df_all(data_path, segment_length=512, normalize=False):
    '''
    Load, preprocess and return a DataFrame which contains all signals data and
    labels and is ready to be used for model training.
    
    Parameter:
        normal_path: 
            Path of the folder which contains matlab files of normal bearings
        DE_path: 
            Path of the folder which contains matlab files of DE faulty bearings
        segment_length: 
            Number of points per segment. See divide_signal() function
        normalize: 
            Boolean to perform normalization to the signal data
    Return:
        df_all: 
            DataFrame which is ready to be used for model training.
    '''
    df = matfile_to_df(data_path)

    if normalize:
        normalize_signal(df)
    df_processed = divide_signal(df, segment_length)

    map_label = {'N':0, 'B':1, 'IR':2, 'OR':3}
    df_processed['label'] = df_processed['label'].map(map_label)
    return df_processed

def download(url:str, dest_dir:Path, save_name:str, suffix=None) -> Path:
    assert isinstance(dest_dir, Path), "dest_dir must be a Path object"
    if not dest_dir.exists():
        dest_dir.mkdir()
    if save_name == None: filename = url.split('/')[-1]
    else: filename = save_name+suffix
    file_path = dest_dir / filename
    if not file_path.exists():
        print(f"Downloading {file_path}")
        with open(f'{file_path}', 'wb') as f:
            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-length'))
            with tqdm(total=total, unit='B', unit_scale=True, desc=filename) as pbar:
                for data in response.iter_content(chunk_size=1024*1024):
                    f.write(data)
                    pbar.update(1024*1024)
    else:
        return file_path
    return file_path