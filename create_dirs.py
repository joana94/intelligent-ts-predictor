
"""
This script is responsible for creating the directory structure of the project
"""
import os
from base.func_utils import pickle_file, json_dir, create_project_dirs

if __name__ == "__main__":

    os.makedirs(json_dir, exist_ok=True) 

    project_name = input("\tPlease choose a name for the time series forecasting project: ")
    paths = create_project_dirs(project_name, json_dir)
    

    