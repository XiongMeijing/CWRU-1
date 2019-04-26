import tkinter as tk
from tkinter import filedialog
from tkinter import *
import tkinter.ttk as ttk

from pathlib import Path
import scipy.io
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
style.use('ggplot')

LARGE_FONT= ("Verdana", 12)

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


class SeaofBTCapp(tk.Tk):

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
        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

    def update_listview(self, filepaths):
        filenames = self.model.read_files(filepaths)
        return filenames


class ListViewFrame(ttk.LabelFrame):
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        scrollbar = ttk.Scrollbar(self)
        self.listbox = tk.Listbox(
            self, 
            height=10, 
            width=50, 
            selectmode=EXTENDED,
            yscrollcommand = scrollbar.set,)
        self.listbox.grid(row=1, column=1)
        self.listbox2 = tk.Listbox(
            self, 
            height=10, 
            width=20, 
            selectmode=EXTENDED,
            yscrollcommand = scrollbar.set,)
        self.listbox2.grid(row=1, column=2)

        scrollbar.grid(row=1, column=3, rowspan=1, sticky=N+S+W)
        scrollbar.config( command = self.yview )

    def yview(self, *args):
        self.listbox.yview(*args)
        self.listbox2.yview(*args)



class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.model = Model()
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.grid(row=1, column=1, pady=10,padx=10)

        self.labelframe = ListViewFrame(self, text="labeled frame")
        self.labelframe.grid(row=2, column=1, pady=10,padx=10)

        button_frame = ttk.Frame(self)
        button_frame.grid(row=3, column=1, pady=10,padx=10)
        
        button3 = ttk.Button(button_frame, text="Open",
                            command=self.select_files)
        button3.grid(row=1, column=1, rowspan=1)
        button4 = ttk.Button(button_frame, text="Delete All",
                            command=self.delete_list_all)
        button4.grid(row=1, column=2, rowspan=1)
        button5 = ttk.Button(button_frame, text="Delete",
                            command=self.delete_list_selected)
        button5.grid(row=1, column=3, rowspan=1)

        self.plotframe = PlotFrame(self)
        self.plotframe.grid(row=1, column=2, rowspan=3)

    def select_files(self):
        paths = filedialog.askopenfilenames()
        filenames = self.model.read_files(paths)
        self.labelframe.listbox.delete(0,END)
        self.labelframe.listbox2.delete(0,END)
        for i, filename in enumerate(self.model.data['filenames']):
            self.labelframe.listbox.insert(i, filename)
            self.labelframe.listbox2.insert(i, filename)

    def delete_list_all(self):
        for k,v in self.model.data.items():
            self.model.data[k] = []
        self.labelframe.listbox.delete(0, END)
        self.labelframe.listbox2.delete(0, END)

    def delete_list_selected(self):
        # Each time listbox.delete method is called, the index of the listbox is
        # reset. Hence the index of the selected item needs to be corrected by
        # subtracting the count of deleted items.
        del_indices = self.get_list_selection()
        
        count = 0
        for del_idx in del_indices:
            for k,v in self.model.data.items():
                v.pop(del_idx - count)
            
            self.labelframe.listbox.delete(del_idx - count)
            self.labelframe.listbox2.delete(del_idx - count)
            count += 1

    def get_list_selection(self):
        return self.labelframe.listbox.curselection()

    def plot_something(self):
        try:
            idx = self.get_list_selection()
            idx = idx[0]
            
            arr = self.model.data['signals'][idx]
            
            self.plotframe.a.clear()
            self.plotframe.a.plot(arr)
            self.plotframe.canvas.draw()
        except Exception as e:
            print(e)


class PlotFrame(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self.f = Figure(figsize=(5,5), dpi=100)
        self.a = self.f.add_subplot(111)
        label = ttk.Label(self, text="Graph Page!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        draw_button = ttk.Button(self, text="Draw Something",
                             command=parent.plot_something)
        draw_button.pack()

    
        
class Model:
    def __init__(self):
        self.data = {}
        self.data['filenames'] = []
        self.data['filepaths'] = []
        self.data['signals'] = []
        self.data['prediction'] = []

    def get_signal(self, file_index):
        return self.data['filenames'][index], self.data['signals'][index]

    def predict(self, file_index):
        pass

    def read_files(self, filepaths):
        for filepath in filepaths:
            file_name = str(filepath).split('/')[-1]
            self.data['filenames'].append(file_name)
            self.data['filepaths'].append(Path(filepath))
            ## TODO mat_to_ndarray
            self.data['signals'].append(mat_to_ndarray(Path(filepath)))
            self.data['prediction'].append('placeholder')
        return self.data['filenames']



if __name__ == "__main__":
    app = SeaofBTCapp()
    app.mainloop()
    # testpath = Path(r'C:\Users\A800005316\Desktop\Study\02_Automation of QA with Machine Learning\Experiments\CWRU\Data\12k_DE\B007_0.mat')
    # print(testpath)
    # arr = mat_to_ndarray(testpath)
    # print(arr.shape)
    # model = Model()
    # model.predict(["a"])