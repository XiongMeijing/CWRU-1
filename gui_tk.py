from gui_core.view import *

from pathlib import Path
import scipy.io
import numpy as np
import pandas as pd

import nn_model
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from pathlib import Path
save_model_path = Path("./Model")


def rename_matfile_keys(dic):
    '''
    Rename some keys so that they can be loaded into a 
    DataFrame with consistent column names
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
    matfile_dic = scipy.io.loadmat(matfile_path)
    rename_matfile_keys(matfile_dic)
    return matfile_dic['DE_time']

def preprocess_signal(array, segment_length):
    array = normalize_signal_array(array)
    return divide_signal_array(array, segment_length)

def normalize_signal_array(array):
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
    # return pd.concat([df_tmp[['label', 'filename']], pd.DataFrame(np.hstack(df_tmp["signal"].values).T)], axis=1 )    
        
class MainApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        
        self.frames = {}

        for F in [StartPage]:
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        startpage_controller = StartPageController(self.frames[StartPage])
        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()


class StartPageController:
    def __init__(self, page):
        self.page = page
        self.model = Model()
        self.page.button_open.config(command=self.select_files)
        self.page.button_del_all.config(command=self.delete_list_all)
        self.page.button_del_selected.config(command=self.delete_list_selected)
        self.page.button_pred.config(command=self.make_prediction)
        self.page.plotframe.draw_button.config(command=self.plot_something)

    def select_files(self):
        paths = filedialog.askopenfilenames()
        filenames = self.model.read_files(paths)
        self.page.labelframe.lb_filenames.delete(0,END)
        self.page.labelframe.lb_predictions.delete(0,END)
        for i, (filename, pred) in enumerate(zip(self.model.data['filenames'], self.model.data['prediction'])):
            self.page.labelframe.lb_filenames.insert(i, filename)
            self.page.labelframe.lb_predictions.insert(i, pred)

    def delete_list_all(self):
        for k,v in self.model.data.items():
            self.model.data[k] = []
        self.page.labelframe.lb_filenames.delete(0, END)
        self.page.labelframe.lb_predictions.delete(0, END)

    def delete_list_selected(self):
        # Each time listbox.delete method is called, the index of the listbox is
        # reset. Hence the index of the selected item needs to be corrected by
        # subtracting the count of deleted items.
        del_indices = self.get_list_selection()
        count = 0
        for del_idx in del_indices:
            for k,v in self.model.data.items():
                v.pop(del_idx - count)
            
            self.page.labelframe.lb_filenames.delete(del_idx - count)
            self.page.labelframe.lb_predictions.delete(del_idx - count)
            count += 1

    def get_list_selection(self):
        return self.page.labelframe.lb_filenames.curselection()

    def plot_something(self):
        try:
            idx = self.get_list_selection()
            idx = idx[0]
            arr = self.model.data['signals'][idx]
            self.page.plotframe.a.clear()
            self.page.plotframe.a.plot(arr)
            self.page.plotframe.canvas.draw()
        except Exception as e:
            print(e)

    def make_prediction(self):
        pred_indices = self.get_list_selection()
        for pred_index in pred_indices:
            self.model.predict(pred_index)
            self.page.labelframe.lb_predictions.insert(pred_index, self.model.data['prediction'][pred_index])
            self.page.labelframe.lb_predictions.delete(pred_index + 1)


class Model:
    def __init__(self):
        self.data = {}
        self.data['filenames'] = []
        self.data['filepaths'] = []
        self.data['signals'] = []
        self.data['prediction'] = []
        self.pred_model = nn_model.CNN_1D_2L(500)
        self.pred_model.load_state_dict(torch.load(save_model_path / 'model.pth'))
        self.pred_model.eval()

    def get_signal(self, file_index):
        return self.data['filenames'][file_index], self.data['signals'][file_index]

    def predict(self, file_index):
        x = self.data['signals'][file_index]
        x = preprocess_signal(x, 500)
        x = torch.tensor(x, dtype=torch.float32)
        out = self.pred_model(x)
        pred = torch.argmax(out, dim=1)
        mode, _ = torch.mode(pred, dim=0)
        self.data['prediction'][file_index] = mode.item()

    def read_files(self, filepaths):
        for filepath in filepaths:
            file_name = str(filepath).split('/')[-1]
            self.data['filenames'].append(file_name)
            self.data['filepaths'].append(Path(filepath))
            self.data['signals'].append(mat_to_ndarray(Path(filepath)))
            self.data['prediction'].append('None')
        return self.data['filenames']



if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
    # testarr = np.array(range(50)).reshape(-1,1)
    # print(preprocess_signal(testarr, 8))
