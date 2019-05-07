import tkinter as tk
from tkinter import filedialog
from tkinter import *
import tkinter.ttk as ttk
LARGE_FONT= ("Verdana", 12)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
style.use('ggplot')


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.grid(row=1, column=1, pady=10,padx=10)

        self.labelframe = ListViewFrame(self, text="labeled frame")
        self.labelframe.grid(row=2, column=1, pady=10,padx=10, sticky=NSEW)

        self.button_frame = ttk.Frame(self)
        self.button_frame.grid(row=3, column=1, pady=10,padx=10)
        
        self.button_open = ttk.Button(self.button_frame, text="Open")
        self.button_open.grid(row=1, column=1, rowspan=1)
        self.button_del_all = ttk.Button(self.button_frame, text="Delete All")
        self.button_del_all.grid(row=1, column=2, rowspan=1)
        self.button_del_selected = ttk.Button(self.button_frame, text="Delete")
        self.button_del_selected.grid(row=1, column=3, rowspan=1)
        self.button_pred = ttk.Button(self.button_frame, text="Predict")
        self.button_pred.grid(row=1, column=4, rowspan=1)

        self.plotframe = PlotFrame(self)
        self.plotframe.grid(row=1, column=2, rowspan=3)


class ListViewFrame(ttk.LabelFrame):
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        scrollbar = ttk.Scrollbar(self)
        self.lb_filenames = tk.Listbox(
            self, 
            height=30, 
            width=50, 
            selectmode=EXTENDED,
            yscrollcommand = scrollbar.set,)
        self.lb_filenames.grid(row=1, column=1)
        self.lb_predictions = tk.Listbox(
            self, 
            height=30, 
            width=20, 
            selectmode=EXTENDED,
            yscrollcommand = scrollbar.set,)
        self.lb_predictions.grid(row=1, column=2)

        scrollbar.grid(row=1, column=3, rowspan=1, sticky=N+S+W)
        scrollbar.config( command = self.yview )

    def yview(self, *args):
        self.lb_filenames.yview(*args)
        self.lb_predictions.yview(*args)


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

        self.draw_button = ttk.Button(self, text="Draw Something")
        self.draw_button.pack()
