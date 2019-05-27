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
        label = tk.Label(self, text="", font=LARGE_FONT)
        label.grid(row=1, column=1, pady=10,padx=10)

        # Frame to contain the list views
        self.labelframe = ListViewFrame(self, text="Files")
        self.labelframe.grid(row=2, column=1, pady=10,padx=10, sticky=NSEW)

        # Frame to contain the buttons
        self.button_frame = ttk.Frame(self)
        self.button_frame.grid(row=3, column=1, pady=10,padx=10)
        
        self.button_open =         ttk.Button(self.button_frame, text="Open")
        self.button_del_selected = ttk.Button(self.button_frame, text="Delete")
        self.button_del_all =      ttk.Button(self.button_frame, text="Delete All")
        self.button_pred =         ttk.Button(self.button_frame, text="Predict")
        self.draw_button =         ttk.Button(self.button_frame, text="Draw")
        
        self.button_open.grid        (row=1, column=1, pady=3,padx=5, rowspan=1)
        self.button_del_selected.grid(row=2, column=1, pady=3, padx=5, rowspan=1)
        self.button_del_all.grid     (row=3, column=1, pady=3, padx=5, rowspan=1)
        self.button_pred.grid        (row=1, column=3, pady=3, padx=30, rowspan=1)
        self.draw_button.grid        (row=2, column=3, pady=3, padx=30, rowspan=1)

        # Frame to contain the PlotFrame
        self.plotframe = PlotFrame(self)
        self.plotframe.grid(row=1, column=2, rowspan=3)


class ListViewFrame(ttk.LabelFrame):
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        label_fn = tk.Label(self, text="Filenames")
        label_fn.grid(row=1, column=1, pady=5,padx=5)
        label_pred = tk.Label(self, text="Predictions")
        label_pred.grid(row=1, column=2, pady=5,padx=5)
        
        scrollbar = ttk.Scrollbar(self)
        self.lb_filenames = tk.Listbox(
            self, 
            height=30, 
            width=50, 
            selectmode=EXTENDED,
            yscrollcommand = scrollbar.set,)
        self.lb_filenames.grid(row=2, column=1)
        self.lb_predictions = tk.Listbox(
            self, 
            height=30, 
            width=20, 
            selectmode=EXTENDED,
            yscrollcommand = scrollbar.set,)
        self.lb_predictions.grid(row=2, column=2)
        self.lb_filenames.bind("<MouseWheel>", self.on_mouse_wheel)
        self.lb_predictions.bind("<MouseWheel>", self.on_mouse_wheel)
        scrollbar.grid(row=2, column=3, rowspan=1, sticky=N+S+W)
        scrollbar.config( command = self.yview )

    def yview(self, *args):
        self.lb_filenames.yview(*args)
        self.lb_predictions.yview(*args)

    def on_mouse_wheel(self, event):
        '''
        Source: https://stackoverflow.com/questions/17355902/python-tkinter-binding-mousewheel-to-scrollbar
        '''
        self.lb_filenames.yview("scroll", int(-1*(event.delta/60)),"units")
        self.lb_predictions.yview("scroll", int(-1*(event.delta/60)),"units")
        # this prevents default bindings from firing, which
        # would end up scrolling the widget twice
        return "break"


class PlotFrame(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self.f = Figure(figsize=(5,5), dpi=100)
        self.a = self.f.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        
