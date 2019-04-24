import tkinter as tk
from tkinter import filedialog
from tkinter import *

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

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
        label.pack(pady=10,padx=10)

        labelframe = ListViewFrame(self, text="labeled frame")
        labelframe.pack()

        button_frame = tk.Frame(self)
        button_frame.pack()
        
        button3 = tk.Button(button_frame, text="Open",
                            command=labelframe.select_files)
        button3.grid(row=1, column=1, rowspan=1)
        button4 = tk.Button(button_frame, text="Delete All",
                            command=labelframe.delete_list_all)
        button4.grid(row=1, column=2, rowspan=1)
        button5 = tk.Button(button_frame, text="Delete",
                            command=labelframe.delete_list_selected)
        button5.grid(row=1, column=3, rowspan=1)




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