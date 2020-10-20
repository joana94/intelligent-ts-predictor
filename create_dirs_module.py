"""
------------------------------------------------------------------------------
This module is responsible for creating the directory structure of the project
------------------------------------------------------------------------------
"""
import os
from utils_classes import FoldersUtils, json_dir


def main():
    """
    Functionalities to be executed by the module.
    """

    os.makedirs(json_dir, exist_ok=True)

    project_name = input(
        "> Please choose a name for the time series forecasting project: ")
    paths = FoldersUtils.create_project_dirs(project_name, json_dir)


if __name__ == "__main__":

    main()
