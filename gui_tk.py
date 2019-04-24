import tkinter as tk
from tkinter import filedialog
from tkinter import *

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style
style.use('ggplot')

LARGE_FONT= ("Verdana", 12)


class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

class ListViewFrame(tk.LabelFrame):
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        scrollbar = tk.Scrollbar(self)
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

    def select_files(self):
        filenames = filedialog.askopenfilenames()
        for i, filename in enumerate(filenames):
            self.listbox.insert(i, filename)
            self.listbox2.insert(i, filename)

    def delete_list_all(self):
        self.listbox.delete(0, END)
        self.listbox2.delete(0, END)

    def delete_list_selected(self):
        # Each time listbox.delete method is called, the index of the listbox is
        # reset. Hence the index of the selected item needs to be corrected by
        # subtracting the count of deleted items.
        count = 0
        for item in self.get_list_selection():
            self.listbox.delete(item - count)
            self.listbox2.delete(item - count)
            count += 1

    def get_list_selection(self):
        return self.listbox.curselection()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.grid(row=1, column=1, pady=10,padx=10)

        labelframe = ListViewFrame(self, text="labeled frame")
        labelframe.grid(row=2, column=1, pady=10,padx=10)

        button_frame = tk.Frame(self)
        button_frame.grid(row=3, column=1, pady=10,padx=10)
        
        button3 = tk.Button(button_frame, text="Open",
                            command=labelframe.select_files)
        button3.grid(row=1, column=1, rowspan=1)
        button4 = tk.Button(button_frame, text="Delete All",
                            command=labelframe.delete_list_all)
        button4.grid(row=1, column=2, rowspan=1)
        button5 = tk.Button(button_frame, text="Delete",
                            command=labelframe.delete_list_selected)
        button5.grid(row=1, column=3, rowspan=1)

        plotframe = PlotFrame(self)
        plotframe.grid(row=1, column=2, rowspan=3)


class PlotFrame(tk.Frame):

    def __init__(self, parent):
        super().__init__(parent)
        self.f = Figure(figsize=(5,5), dpi=100)
        self.a = self.f.add_subplot(111)
        label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        draw_button = Button(self, text="Draw Something",
                             command=self.plot_something)
        draw_button.pack()

    def plot_something(self):
        self.a.clear()
        self.a.plot(np.array([1,2,3,4,5,6,7,8]),np.random.randn(8))
        self.canvas.draw()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = tk.Button(self, text="Page Two",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page Two!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = tk.Button(self, text="Page One",
                            command=lambda: controller.show_frame(PageOne))
        button2.pack()
        


app = SeaofBTCapp()
app.mainloop()