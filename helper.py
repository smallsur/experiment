from utils import del_cache

import argparse

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Experiment_Helper')

    parser.add_argument("--del_cache",type=bool,default=True)

    args = parser.parse_args()
    
    
    
    if args.del_cache:

        del_cache()