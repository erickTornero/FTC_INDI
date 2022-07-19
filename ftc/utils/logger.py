import os
from typing import Optional
class Logger:
    def __init__(self, save_folder: os.path, namefile: Optional[str]=None):
        """
            Personal Logger allows to log on a file any process
        """
        self.save_folder    =   save_folder
        self.text           =   ''
        self.namefile       =   namefile
    def save(self, namefile: Optional[str]=None):
        namefile = self.namefile if namefile is None else self.namefile
        file_path           =   os.path.join(self.save_folder, namefile)
        if len(self.text) > 0:
            with open(file_path, 'w') as fc:
                fc.write(self.text)
    
    def log(self, newtext: str, newline=True):
        print(newtext)
        newtext = newtext + ('\n' if newline else '')
        self.text += newtext
        

    def logwarn(self, newtext: str, newline=True):
        print(newtext)
        newtext = '[WARNING]' + newtext +('\n' if newline else '')
        self.text +=  newtext
        
    
    def logerr(self, newtext: str, newline=True):
        print(newtext)
        newtext =   '[ERROR]' + newtext + ('\n' if newline else '')
        self.text += newtext
        

    def load(self, namefile: Optional[str]=None):
        namefile = self.namefile if namefile is None else self.namefile
        file_path           =   os.path.join(self.save_folder, namefile)
        if os.path.exists(file_path):
            with open(file_path, 'r') as fc:
                lines   =   fc.readlines()
                self.text   =   ''.join(lines)
    