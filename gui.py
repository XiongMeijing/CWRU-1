import os
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import ObjectProperty
from kivy.uix.stencilview import StencilView
from kivy.uix.popup import Popup
# from kivy.uix.filechooser import 
import pickle
from kivy.garden.filebrowser import FileBrowser
from desktop_file_dialogs import Desktop_FilesDialog, FileGroup


class Model:
    def __init__(self):
        pass

    def predict(self, x):
        print('Model predict method')

    def export(self):
        print('Model export method')


class Main(BoxLayout):

    painter = ObjectProperty(None)
    prediction_display = ObjectProperty(None)
    
    model = Model()

    def predict(self, obj):
        print('predict btn binding')
        self.model.predict(1)

    def export(self, obj):
        print('export btn binding')
        self.model.export()

    def save_png(self, obj):
        print('save btn binding')

    def get_file_path(self, path):
        self.file = path
        print(self.file)

    def show_load(self):
        Desktop_FilesDialog(
        title             = "Select File",
        initial_directory = "",
        on_accept         = self.get_file_path,
        on_cancel         = lambda: print(">>> NO FILE SELECTED"),
        file_groups = [
            FileGroup.All_FileTypes,
            # FileGroup(name="Image Files", extensions=["jpg", "jpeg", "png", "gif"]),
        ],
        ).show()
    
    def load(self, path, filename):
        # with open(os.path.join(path, filename[0])) as stream:
        #     self.text_input.text = stream.read()
        print('load!!')
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

class PredictApp(App):

    def build(self):
        self.parent = Main()
        return self.parent




if __name__ == '__main__':
    print(os.path.dirname(kivy.__file__))
    PredictApp().run()