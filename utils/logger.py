from datetime import datetime
import os
import pandas as pd

class Logger:
    def __init__(self,logger_name,path_to_save):

        self.logger_name = logger_name
        self.path_to_save = path_to_save

        assert os.path.exists(self.path_to_save)

        self.path_ = os.path.join(self.path_to_save,self.logger_name+'.txt')

    
    def get_path(self):
        return os.path.join(self.path_to_save,self.logger_name+'.txt')


    def write_log(self,log):

        with open(self.get_path(),'a+') as f:
            f.write(log+'\n')


    def read_log(self):
        with open(self.get_path(),'r+') as f:
            print(f.read())



if __name__ == '__main__':
    log = Logger("first",'/media/awen/D/dataset/rstnet/')
    print(log.get_path())
    log.write_log("zhangawen"+log.get_path())
    log.write_log("zhangawen"+log.get_path())
    log.write_log("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@zhangawen"+log.get_path())
    log.read_log()
    l = str(datetime.today().date())
    print(l)
    
    
