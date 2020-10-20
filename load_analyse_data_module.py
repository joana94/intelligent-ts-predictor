"""
------------------------------------------------------------------------
This module is responsible for loading, analysing and splitting the data
------------------------------------------------------------------------
"""

import os
import matplotlib.pyplot as plt
from utils_classes import FoldersUtils, DataUtils, TimeSeriesPlots
from reports_class import StatsReports
import config
import warnings
warnings.filterwarnings("ignore")

# Extract the generated paths by the create_dirs_module.py
sub_dirs = FoldersUtils.unpickle_file('sub_dirs_list')
data_folder = sub_dirs.get('Data')
models_folder = sub_dirs.get('Models')
graphics_folder = sub_dirs.get('Graphics')
reports_folder = sub_dirs.get('Reports')

def main():
    """
    Functionalities to be executed by the module.
    """
    # Loads dataset inside the "Data" folder
    DATASET = DataUtils.get_data_file(folder = data_folder)

    # Checks any missing dates or values and if finds any, fills
    # the dates and then the missing values through linear interpolation
    print("\n> Checking any missing dates or values...")
    DATASET = DataUtils.check_missing_dates_and_values(DATASET)

    print(
        f'\n> Generating a report with descriptive statistics of the time series in {reports_folder}')
    StatsReports.general_stats(DATASET, report_name='0. TS descriptive stats', out_folder=reports_folder)

    print(f'\n> Saving the time plot to {graphics_folder}')
    TimeSeriesPlots.time_plot(DATASET, out_folder=graphics_folder)

    print(f'> Saving the ACF and PACF plots to {graphics_folder}')
    TimeSeriesPlots.acf_plot(DATASET, 35, out_folder=graphics_folder)
    TimeSeriesPlots.pacf_plot(DATASET, 35, out_folder=graphics_folder)

    print(f'\n> Creating a KPSS test report in {reports_folder}')
    StatsReports.kpss_(DATASET, significance=0.05, report_name='1. KPSS report', out_folder=reports_folder)

    print('\n> Spliting the time series dataset in a training and a test set based on the split point provided in the config.py file')
    train, test = DataUtils.train_test_split(DATASET, split_point=config.SPLIT_POINT)

    DataUtils.train_test_to_csv(train, test, out_folder=data_folder) 
    print(f'> Train and test sets saved to {data_folder}')


if __name__ == '__main__':

    main()



