import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Utils(object):
    """
    Container class for all the methods related to data ingestion and preprocessing.
    """


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
                print("Couldn't find any datetime column")

        elif date_col is not None and date_col != 'infer':
            data = pd.read_csv(filename, index_col=date_col, parse_dates=True)

        else:
            data = pd.read_csv(filename)
            print("You should provide a date column name")

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
            print('No dates are missing')

        elif pd.infer_freq(data.index) is None:
            # determines date frequency based on the last seven timesteps
            freq_last_timesteps = pd.infer_freq(data.index[-7:])
            all_dates = pd.date_range(start=data.index.min(
            ), end=data.index.max(), freq=freq_last_timesteps)
            missing_dates = all_dates.difference(data.index)
            print(f'Missing dates: {missing_dates}')

            if True in data.index.duplicated():
                remove_duplicates = data[~data.index.duplicated()]
                data_new_idx = remove_duplicates.reindex(all_dates)
                interpolate = data_new_idx.interpolate('time')

                return interpolate

            else:
                data_new_idx = data.reindex(all_dates)
                interpolate = data_new_idx.interpolate('time')

                return interpolate

        if True in data.isnull().all(1):
            interpolate = data.interpolate('linear')
            return interpolate

        else:
            print('No values are missing')

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
    def train_test_to_csv(train_set, test_set, filepath):
        train_set.to_csv(os.path.join(filepath, 'train_set.csv'))
        test_set.to_csv(os.path.join(filepath, 'test_set.csv'))

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

        if test_data:
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
