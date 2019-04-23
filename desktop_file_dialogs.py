# From https://gist.github.com/Enteleform/c9049a589ef7c490c2c2a8e1e02860e5#file-test-py
import os
from typing    import List, Callable
from abc       import ABCMeta, abstractproperty
from tkinter   import Tk, filedialog as TkInter_FileDialog

Tk().withdraw() # disables TkInter GUI


########################################################################################################################################################################################################################################################################################################~{#
#//////|   FileGroup   |////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////g2#
########################################################################################################################################################################################################################################################################################################~}#

class _All_FileTypes:
  name       = "All Files"
  extensions = ["*"]
  file_types = ("All Files", "*.*")


class FileGroup:
  All_FileTypes = _All_FileTypes

  def __init__(self, name:str, extensions:List[str]):
    self.name = name
    self.extensions = extensions

  @property
  def file_types(self):
    extensions_string = " ".join([f"*.{x}" for x in self.extensions])
    return (self.name, extensions_string)


########################################################################################################################################################################################################################################################################################################~{#
#//////|   Dialog Base   |//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////g2#
########################################################################################################################################################################################################################################################################################################~}#

class Desktop_FileDialog_Base(metaclass=ABCMeta):

  @abstractproperty
  @property
  def args(self):
    return {}

  def __init__(self,
    show_dialog:       Callable,
    title:             str,
    initial_directory: str,
    on_accept:         Callable,
    on_cancel:         Callable,
    file_groups:       List[FileGroup],
  ):
    self.show_dialog       = show_dialog
    self.title             = title
    self.initial_directory = initial_directory
    self.on_accept         = on_accept
    self.on_cancel         = on_cancel
    self.file_groups       = file_groups
    self._validate_initial_directory()

  def show(self):
    path = self.show_dialog(**self.args)
    if(path):
      self.on_accept(path)
    else:
      self.on_cancel()

  def _validate_initial_directory(self):
    if not(self.initial_directory):
      self.initial_directory = os.path.abspath(os.sep)
    else:
      if not(os.path.isdir(self.initial_directory)):
        raise ValueError(f"\n\t_invalid Directory: '{self.initial_directory}'")


########################################################################################################################################################################################################################################################################################################~{#
#//////|   Dialogs   |//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////g2#
########################################################################################################################################################################################################################################################################################################~}#

class Desktop_FolderDialog(Desktop_FileDialog_Base):
  @property
  def args(self):
    return {
      "title":      self.title,
      "initialdir": self.initial_directory,
    }
  def __init__(self,
    title:             str,
    initial_directory: str,
    on_accept:         Callable,
    on_cancel:         Callable,
  ):
    super().__init__(
      show_dialog       = TkInter_FileDialog.askdirectory,
      title             = title,
      initial_directory = initial_directory,
      on_accept         = on_accept,
      on_cancel         = on_cancel,
      file_groups       = [],
    )


class Desktop_FileDialog(Desktop_FileDialog_Base):
  @property
  def args(self):
    return {
      "title":      self.title,
      "initialdir": self.initial_directory,
      "filetypes":  (x.file_types for x in self.file_groups),
    }
  def __init__(self,
    title:             str,
    initial_directory: str,
    on_accept:         Callable,
    on_cancel:         Callable,
    file_groups:       List[FileGroup],
  ):
    super().__init__(
      show_dialog       = TkInter_FileDialog.askopenfilename,
      title             = title,
      initial_directory = initial_directory,
      on_accept         = on_accept,
      on_cancel         = on_cancel,
      file_groups       = file_groups,
    )


class Desktop_FilesDialog(Desktop_FileDialog_Base):
  @property
  def args(self):
    return {
      "title":      self.title,
      "initialdir": self.initial_directory,
      "filetypes":  (x.file_types for x in self.file_groups),
    }
  def __init__(self,
    title:             str,
    initial_directory: str,
    on_accept:         Callable,
    on_cancel:         Callable,
    file_groups:       List[FileGroup],
  ):
    super().__init__(
      show_dialog       = TkInter_FileDialog.askopenfilenames,
      title             = title,
      initial_directory = initial_directory,
      on_accept         = on_accept,
      on_cancel         = on_cancel,
      file_groups       = file_groups,
    )


class Desktop_SaveFile_Dialog(Desktop_FileDialog_Base):
  @property
  def args(self):
    return {
      "title":            self.title,
      "initialdir":       self.initial_directory,
      "filetypes":        (x.file_types for x in self.file_groups),
      "defaultextension": "",
    }
  def __init__(self,
    title:             str,
    initial_directory: str,
    on_accept:         Callable,
    on_cancel:         Callable,
    file_groups:       List[FileGroup],
  ):
    super().__init__(
      show_dialog       = TkInter_FileDialog.asksaveasfilename,
      title             = title,
      initial_directory = initial_directory,
      on_accept         = on_accept,
      on_cancel         = on_cancel,
      file_groups       = file_groups,
    )
