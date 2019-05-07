import gui_core.model as model
from tkinter import *

Model = model.CNN_1D

class StartPageController:
    def __init__(self, page):
        self.page = page
        self.model = Model()
        self.page.button_open.config(command=self.select_files)
        self.page.button_del_all.config(command=self.delete_list_all)
        self.page.button_del_selected.config(command=self.delete_list_selected)
        self.page.button_pred.config(command=self.get_predictions)
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

    def get_predictions(self):
        pred_indices = self.get_list_selection()
        for pred_index in pred_indices:
            self.model.update_prediction(pred_index)
            self.page.labelframe.lb_predictions.insert(pred_index, self.model.data['prediction'][pred_index])
            self.page.labelframe.lb_predictions.delete(pred_index + 1)
