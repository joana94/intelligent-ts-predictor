import os
from base.func_utils import unpickle_file, json_dir
# from data_functions import read_data
import matplotlib.pyplot as plt
from base.data_class import Data


def get_data_file():
    
    sub_dirs = unpickle_file('sub_dirs_list')
    data_dir = sub_dirs.get('Data')
    
    try:
        fname = os.listdir(data_dir)[0] # Assumes that the Data folder contains the time series data file
        filepath = os.path.join(data_dir, fname)
        dataset = Data.read_data(filepath)
        # print(dataset.head())
        return dataset

    except IndexError:
        print("No file was found!")


DATASET = get_data_file()



