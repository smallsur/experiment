from utils import del_cache

import argparse

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Experiment_Helper')

    parser.add_argument("--del_cache",type=bool,default=True)

    args = parser.parse_args()
    
    # find ./ -type d -empty -exec touch {}/.gitignore \;
    if args.del_cache:

        del_cache()