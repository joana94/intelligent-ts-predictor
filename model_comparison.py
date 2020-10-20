"""
------------------------------------------------------------------------------------------------
This module is responsible for comparing the predictive performence of two or more fitted models
------------------------------------------------------------------------------------------------
"""
import os
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils_classes import DataUtils, FoldersUtils, TimeSeriesPlots

sub_dirs = FoldersUtils.unpickle_file('sub_dirs_list')
data_folder = sub_dirs.get('Data')
models_folder = sub_dirs.get('Models')


def main():

    # Join the dataframes from all the fitted models
    models_df = DataUtils.generate_model_comparison_df(models_to_compare=config.MODELS, folder=models_folder)
    # Generate the a bar plot for the errors of each model
    TimeSeriesPlots.errors_bar_plot(models_df, folder=models_folder)
    print(f'> Generated predictive performance comparison files and charts to {models_folder}')
if __name__ == '__main__':

    main()