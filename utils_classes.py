import os
import pandas as pd
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler

current_dir = os.getcwd()
json_dir = os.path.join(current_dir, 'json')


class FoldersUtils(object):
    """
    Container class for all the methods related to folder creation.
    """
    @staticmethod
    def pickle_file(filename, object_name):
        with open(os.path.join(json_dir, filename), 'wb') as f:
            jfile = pickle.dump(object_name, f)
        return jfile

    @staticmethod
    def unpickle_file(filename):
        with open(os.path.join(json_dir, filename), 'rb') as f:
            jfile = pickle.load(f)
        return jfile

    @staticmethod
    def create_project_dirs(project_name, json_dir):
        """
        Creates by default the working directory in the user's "Documents" folder
        and all the nested directories, namely,"Data", "Analysis", "Models",
        where the outputs of the script will be dumped

        Parameters
        ----------
        name: str, optional
            The user must give a name to the project folder.

        Returns
        -------
        List of strings
            The list contains the paths of each subdirectory
        """
        main_folder = 'Intelligent TS Predictor'
        sub_dirs = ['Data', 'Models', 'Graphics', 'Reports']
        # will be used by each module to store its outputs in the correct directory
        sub_dirs_paths = {}

        root = os.path.expanduser("~/Documents")
        project_dir = os.path.join(root, main_folder, project_name)

        # exists_ok prevents the FileExistsError if the directory already exists
        os.makedirs(project_dir, exist_ok=True)

        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(project_dir, sub_dir), exist_ok=True)
            sub_dirs_paths[sub_dir] = os.path.join(project_dir, sub_dir)

        FoldersUtils.pickle_file('sub_dirs_list', sub_dirs_paths)
        FoldersUtils.pickle_file('project_name', project_name)

        print('> Project folder successfuly created!')
        print(f"> Folder name: {project_name}")

        return sub_dirs_paths


class DataUtils(object):
    """
    Container class for all the methods related to data ingestion and preprocessing.
    """

    @staticmethod
    def get_data_file(folder, name=None):
        """
        Grabs the time series file inside the data folder.
        """
        if name is None:
            try:
                # Assumes that the Data folder contains the time series data file
                name = os.listdir(folder)[0]
                filepath = os.path.join(folder, name)
                dataset = DataUtils.read_data(filepath)
                # print(dataset.head())
                print(f'> {name} file successfully loaded.')
                return dataset

            except IndexError:
                print("> ERROR: No file was found inside the Data folder!")
                sys.exit(
                    "> You have to insert a time series .csv file in the Data folder in order to proceed.")
        else:
            filepath = os.path.join(folder, name)
            dataset = DataUtils.read_data(filepath)
            return dataset

    @staticmethod
    def read_data(filename, date_col='infer'):
        """
        Utility function to read the time series dataset into a pandas DataFrame

        Parameters
        ----------
        filename: str
            Accepts the filename. The file format should be a comma-separated values (.csv)
            or a text file (.txt)
        date_col: str, default 'infer'
            Accepts the name of the column that contains the date strings
            If 'infer' the function will automatically find which column contains the datetime

        Returns
        -------
        pandas DataFrame
            A tabular data structure useful for futher processing of the data
        """
        if date_col == 'infer':
            data = pd.read_csv(filename)
            # https://stackoverflow.com/questions/33204500/pandas-automatically-detect-date-columns-at-run-time
            data = data.apply(lambda col: pd.to_datetime(col, errors='ignore')
                              if col.dtypes == object
                              else col,
                              axis=0)

            type_list = list(data.dtypes)

            if "datetime64[ns]" in type_list:
                index_datetime = type_list.index("datetime64[ns]")
                data.set_index(data.columns[index_datetime], inplace=True)
            else:
                print("> Couldn't find any datetime column")

        elif date_col is not None and date_col != 'infer':
            data = pd.read_csv(filename, index_col=date_col, parse_dates=True)

        else:
            data = pd.read_csv(filename)
            #print("> You should provide a date column name")

        return data

    @staticmethod
    def check_missing_dates_and_values(data):
        """
        Helper function to check if any dates are missing or if there are duplicated dates. If dates are missing,
        it automatically reindexes the data by including the missing dates and filling the corresponding new missing
        values through a linear interpolation.

        Arguments
        ---------
        data: pd.DataFrame or pd.Series
            Must have a valid datetime index. 

        Returns:
            New pd.DataFrame with all missing dates values filled. 
        """

        if pd.infer_freq(data.index) is not None:
            print('> No dates are missing')
            data.index.freq = pd.infer_freq(data.index)

        elif pd.infer_freq(data.index) is None:
            # determines date frequency based on the last seven timesteps
            freq_last_timesteps = pd.infer_freq(data.index[-7:])
            all_dates = pd.date_range(start=data.index.min(
            ), end=data.index.max(), freq=freq_last_timesteps)
            missing_dates = all_dates.difference(data.index)
            # # print(f'Missing dates: {missing_dates}')
            print(
                '> Found missing dates! Inserting dates and filling their values through linear interpolation')

            if True in data.index.duplicated():
                remove_duplicates = data[~data.index.duplicated()]
                data_new_idx = remove_duplicates.reindex(all_dates)
                interpolate = data_new_idx.interpolate('time')
                data.index.freq = pd.infer_freq(data.index)
                return interpolate

            else:
                data_new_idx = data.reindex(all_dates)
                interpolate = data_new_idx.interpolate('time')
                data.index.freq = pd.infer_freq(data.index)
                return interpolate

        if True in data.isnull().all(1):
            interpolate = data.interpolate('linear')
            return interpolate

        else:
            print('> No values are missing')
            return data

    @staticmethod
    def train_test_split(data, split_point):
        """
        Splits the dataset into a training and a testing set.

        Arguments:
        ----------
        data: pd.DataFrame or pd.Series
            Must have a valid datetime index. 
        split_point: int or float
            If a integer equal to or greater than one is passed, the test set will have the number of timesteps equal to 
            the number passed, i.e., if 12 is passed, the test will have the last 12 timesteps and the train will have the
            remaining previous steps.
            If a float between between zero and 1 is passed, the will be splitted as the equivalent percentage from the
            training set.

        Returns:
        --------
        Two pd.DataFrames, one containing the training data and the other containing the test data.
        """
        # split based on percentual value
        if split_point < 1 and split_point > 0:
            test_portion = int(split_point*len(data))
            train = data[:-test_portion]
            test = data[-test_portion:]

        # split based on integer value
        elif split_point >= 1:
            train = data[:-split_point]
            test = data[-split_point:]

        return train, test

    @staticmethod
    def train_test_to_csv(train_set, test_set, out_folder):
        train_set.to_csv(os.path.join(out_folder, 'train_set.csv'))
        test_set.to_csv(os.path.join(out_folder, 'test_set.csv'))

    @staticmethod
    def normalize_data(train_data, test_data=None):
        """
        Reduces the data points to values between the range (0,1).

        Arguments
        ---------
        train_data: pd.DataFrame or pd.Series
        test_data: optional, pd.DataFrame or pd.Series

        Returns
        -------
        A dictionary containing the scaler object and the normalized data.

        """
        # Scaler initialization
        scaler = MinMaxScaler()
        # Fit scaler to train data
        scaler = scaler.fit(train_data)
        # Normalize the training data
        norm_train = scaler.transform(train_data)

        if test_data is not None:
            # Normalize the test data
            norm_test = scaler.transform(test_data)
            scaler_dict = {
                'scaler': scaler, 'normalized_train': norm_train, 'normalized_test': norm_test}
        else:
            scaler_dict = {'scaler': scaler, 'normalized_train': norm_train}

        return scaler_dict

    @staticmethod
    def create_sequences_and_labels(data, seq_len):
        """
        Transforms the data to an appropriate format for use with data mining supervised algorithms,
        namely, neural networks. More specifically, creates windows or sequences of values and their 
        corresponding labels.

        Arguments
        ---------
        data: pd.DataFrame, pd.Series or np.array
        seq_len: int
            Defines the size of the windows or sequences that will be created.
        """
        # if a pandas DataFrame or Series is passed, it is first converted
        # to a numpy array
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.values

        Xs = []
        Ys = []

        for i in range(len(data)-seq_len):
            x = data[i: (i+seq_len)]  # creates the window or sequence
            # creates the corresponding label which is the very next data point
            y = data[i+seq_len]
            Xs.append(x)  # appends each sequence to a list
            Ys.append(y)  # appends each label to a list

        # both lists are converted to numpy arrays
        return np.array(Xs), np.array(Ys)

    @staticmethod
    def generate_model_comparison_df(models_to_compare, folder=None):
        """
        """
        model_dict = {}

        for model in models_to_compare:

            model_dict[model] = pd.read_csv(
                f'{os.path.join(folder, model, model)} predictive performance.csv', index_col=[0])

        values_list = []
        for k, v in model_dict.items():

            v.rename(columns={'Predictive Performance': k}, inplace=True)
            v = v
            values_list.append(v)

        concat_df = pd.concat(values_list, axis=1)

        if folder is not None:
            concat_df.to_csv(os.path.join(folder, 'Model comparison.csv'))
    
        with open(os.path.join(folder, 'Model comparison.txt'), 'w') as f:
            concat_df.to_string(f)

        return concat_df



class TimeSeriesPlots(object):

    @staticmethod
    def time_plot(dataset, out_folder=None, fig_name=None):
        """
        Plots the time series time plot

        Arguments
        --------
        dataset: pd.DataFrame
            The dataframe containing the time series data
        out_folder: str or filepath, optional, default: None
            Location to save the plot
        fig_name: str, optional, default: None
            The name of the saved figure
        """
        plt.style.use('fivethirtyeight')
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.plot(dataset, label=dataset.columns[0], lw=2)
        plt.legend(loc='upper left')
        plt.title("Full time series plot")

        if fig_name is not None:
            name = fig_name
        else:
            name = f'{dataset.columns[0]} time plot.png'

        if out_folder is not None:
            plt.savefig(os.path.join(out_folder, name))
        
        plt.close()


    @staticmethod
    def acf_plot(dataset, lags=50, out_folder=None, fig_name=None):
        """
        """
        plt.style.use('fivethirtyeight')
        plt.rcParams["figure.figsize"] = (12, 6)
        plot_acf(dataset, lags=lags)
        if fig_name is not None:
            name = fig_name
        else:
            name = f'{dataset.columns[0]} ACF plot.png'

        if out_folder is not None:
            plt.savefig(os.path.join(out_folder, name))
        
        plt.close()


    @staticmethod
    def pacf_plot(dataset, lags=50, out_folder=None, fig_name=None):
        """
        """
        plt.style.use('fivethirtyeight')
        plt.rcParams["figure.figsize"] = (12, 6)
        plot_pacf(dataset, lags=lags)

        if fig_name is not None:
            name = fig_name
        else:
            name = f'{dataset.columns[0]} PACF plot.png'

        if out_folder is not None:
            plt.savefig(os.path.join(out_folder, name))
        
        plt.close()

    @staticmethod
    def errors_bar_plot(df, folder=None):
        """
        """
        plt.style.use('fivethirtyeight')
        plt.rcParams["figure.figsize"] = (12, 12)
        df.plot.bar()
        plt.title('Errors for each model')
        plt.xticks(rotation=0)
        plt.savefig(os.path.join(folder, 'Model comparison.png'))
        plt.close()
