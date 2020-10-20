import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rnn_classes import TradRNN, GRU, LSTM
import time
import os
import random
import pickle
from metrics_class import PredictionMetrics
from collections import namedtuple
from itertools import product

class SearchBestArchitecture(object):

    """
    Container class for the methods related to the Neural Network optimal hyperparameters search.
    """

    @staticmethod
    def search_method(method, params, max_iters = 5):
        """
        method: str,
            Accepts 'grid' or 'random' search.
        params: OrderedDict
            Data structure that contains the hyperparameters values.
        max_iters: int, default: 5
            Only necessary for the random search.
        """
        if method == 'grid':
            
            Run = namedtuple('Combination', params.keys())
        
            runs = []
            for v in product(*params.values()):
                runs.append(Run(*v))
            
            return runs
        
        if method == 'random':
                    
            Run = namedtuple('Combination', params.keys())

            runs = []
            random.seed(50)
            for i in range(max_iters):
                random_params= {k: random.sample(v, 1)[0] for k, v in params.items()}
                runs.append(Run(*random_params.values()))

            return runs

    @staticmethod
    def find_architecture(model_name, hyperparameters, search_method, data_x, data_y, epochs, seq_len, max_iters = 5, validation_split=0.1, print_every=1, gradient_clip=1, folder=None):
        """
        Trains a neural network on different hyperparameter combinations and chooses the best architecture based on the 
        global minimum validation across all the inputed hyperparameter combinations. The hyperparameter search can be done
        through two different algorithms: an exhaustive grid search or a random search.
        
        Arguments:
        ----------
        model_name: str, 
            The type of recurrent neural network to be used. It accepts 'rnn', 'gru' or 'lstm' 
        hyperparameters: OrderedDict
            The data structure that holds the possible values for each hyperparameter.
        search_method: str
            Accepts either 'grid' or 'random' as values. Grid search is recommended for a small hyperparameter search
            space whereas random search is recommend for bigger search spaces.
        data_x: numpy.ndarray 
            The input sequences for training the neural network. The array must have shape (num sequences, sequence length, num features)
        data_y: numpy.ndarray 
            The labels of each input sequence. The array must have shape (num sequences, num_features)
        epochs: int
            The number of epochs used to train the model
        max_iters: int, default: 5
            The maximum number of iterations if the choosen search method is 'random'.
        validation_split: float, default:0.1
            How much of the training data to save for validation. The validation data is essential for the hyperparameter search
        print_every: int, default:1
            The interval of information printing between epochs
        """
        if folder is None:
            folder = os.getcwd()

        f = open(os.path.join(
            folder, f'Best {model_name} model search report.txt'), 'w')

        # standard hyperparameter values that should not be changed in order to guarantee the correct functioning of the system
        input_size = 1
        output_size = 1
        n_layers = 2
        
        v_split = int(validation_split*len(data_x))

        train_x, val_x = data_x[:-v_split], data_x[-v_split:]
        train_y, val_y = data_y[:-v_split], data_y[-v_split:]

        training_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    

        validation_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
        
        train_hist = np.zeros(epochs+1)
        valid_hist = np.zeros(epochs+1)
        
        all_train_hist = []
        all_valid_hist = []
        
        # evaluation criterion
        criterion = torch.nn.MSELoss()
        
        # Count number of iterations with diferent hyperparameters
        iteration = 1
        
        best_model = None
        best_score = np.Inf
        
        for hyperparam in SearchBestArchitecture.search_method(method=search_method, params=hyperparameters, max_iters=max_iters):
            
            train_loader = DataLoader(training_data, shuffle= hyperparam.shuffle, 
                                                    batch_size=hyperparam.batch_size, drop_last=True)
            
            valid_loader = DataLoader(validation_data, shuffle=hyperparam.shuffle, 
                                    batch_size=hyperparam.batch_size, drop_last=True)

            
            if model_name.lower() == 'tradrnn':
                model = TradRNN(device='cpu', input_size=input_size, hidden_size=hyperparam.hidden_dim, 
                                output_size=output_size, seq_len=seq_len, n_layers=n_layers)
        
            if model_name.lower() =='gru':
                model = GRU(device='cpu', input_size = input_size, hidden_size=hyperparam.hidden_dim, 
                            output_size=output_size,seq_len = seq_len, n_layers=n_layers)

            if model_name.lower() =='lstm':
                model = LSTM(device='cpu',input_size=input_size, hidden_size=hyperparam.hidden_dim, 
                            output_size= output_size, seq_len=seq_len, n_layers=n_layers)
            
            print(f"\n\n> Run: {iteration}", file=f)
            print(f'> Model: {model_name.upper()}', file=f)
            print(f'> {hyperparam}', file=f)
            print('------------------------------------------------------------------------------', file=f)

            # Optimizer definition
            optimizer = torch.optim.Adam(model.parameters(), lr=hyperparam.learning_rate)
            
            valid_loss_min = np.Inf
            
            for epoch in range(1, epochs + 1):

                start_time = time.time()

                for input_seq, label in train_loader:

                    # get the output of the NN
                    out = model(input_seq.float())
                    # calculate the gradient based on the loss criterion
                    loss = criterion(out, label.float())

                    for valid_seq, valid_label in valid_loader:
                        with torch.no_grad():
                            valid_out = model(valid_seq.float())
                            valid_loss = criterion(valid_out, valid_label.float())

                    # store validation loss for plot
                    valid_hist[epoch] = valid_loss.item()
                
                # store the train loss for plot
                train_hist[epoch] = loss.item()
                epoch_current_time = time.time()

                if epoch % print_every == 0:
                        print('> Epoch {}/{} \n-> Train loss: {:.6f} \n-> Validation loss: {:.6f}'.format(epoch, epochs, loss.item(), valid_loss.item()), file=f)
                        print(f'\nEpoch duration: {epoch_current_time - start_time} seconds', file=f)

                # reset acumulated gradients
                optimizer.zero_grad()

                # backpropagate gradients
                loss.backward()

                # clip gradient to prevent exploding gradients problem
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                # update weights
                optimizer.step()
                
                if valid_loss <= valid_loss_min:
                    print('Minimum Validation loss decresead from {:.6f} --> {:.6f}\n'.format(valid_loss_min, valid_loss), file=f)
                    valid_loss_min = valid_loss
                    
            current_time = time.time()
            
            print('\nMinimum validation loss: {:.6f}'.format(valid_hist[1:].min()), file=f)

            
            if valid_loss_min <= best_score:
                print('Global validation loss decreased from {:.6f} --> {:.6f}'.format(best_score,valid_loss_min), file=f)
                
                best_score = valid_loss_min
                best_model = model
                best_model_train_hist = train_hist
                best_model_valid_hist = valid_hist
                best_params = {'model': best_model,
                            'batch_size': hyperparam.batch_size, 'shuffle': hyperparam.shuffle, 
                            'learning_rate': hyperparam.learning_rate, 'hidden_dim': hyperparam.hidden_dim, 
                            'seq_len': seq_len}

            print(f'\nIteration total time: {str(current_time - start_time)} seconds.', file=f) 
            iteration += 1
        
        print('\n--> GLOBAL MINIMUM VALIDATION LOSS: {} <--'.format(valid_loss_min), file=f)

        print('\n-----------------------------------------------------------------', file=f)
        print(f'BEST ARCHITECTURE FOR {model_name}', file=f)
        print('-----------------------------------------------------------------', file=f)
        for k, v in best_params.items():
            print("{}: {}".format(k, v), file=f)

        f.close()

        return best_model, best_model_train_hist[1:], best_model_valid_hist[1:], best_params
    
    @staticmethod
    def save_best_params(best_params, model_name):
        """
        Pickle best parameters to json file.
        """
        with open(os.path.join(os.getcwd(), f'{model_name} best parameters.pkl'), 'wb') as f:
            jfile = pickle.dump(best_params, f)

    @staticmethod
    def load_best_params(model_name):
        """
        Read back the best parameters
        """
        with open(os.path.join(os.getcwd(), f'{model_name} best parameters.pkl'), 'rb') as f:
            best_p  = pickle.load(f)
        return best_p



class NNForecast(object):

    """
    Container class for methods related to the use of the RNN classes for time series forecasting.
    Included methods: train, evaluate and forecast.
    """
    @staticmethod
    def train(model_name, data_x, data_y, batch_size, hidden_dim, epochs, learning_rate, seq_len, validation_split=None,
              shuffle=False, gradient_clip=None, print_every=1, folder=None):
        """
        Contains all the logic for training the RNNs defined in the RNN module.

        Arguments
        ---------
        model_name: str
            It accepts either 'RNN' for an Elman or traditional recurrent neural network, 'GRU' for a 
            Gated Recurrent Unit or'LSTM' for a Long Short-Term Memory unit.
        data_x: np.array 
            The windowed input sequences. Must have 3D shape (batch_size, seq_len, n_features)
        data_y: np.array
            The labels for each input sequence. Must have 2D shape (batch_size, n_features)
        batch_size: int
            The number of samples inputed simultaneously to the Neural Network for computing the gradient
        hidden_dim: int
            The number of hidden neurons in the Recurrent Neural Networks layers.
        epochs: int
            The number of times the entire data is passed into the network and the gradients backpropagated.
        learning_rate: float
            The step size at each the optimizer updates the gradients. The used optimizer is Adam.
            Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. 
            ArXiv.Org. https://arxiv.org/abs/1412.6980
        seq_len: int
            The length of each sequence passed in data_x.
        validation_split: float, default: None
            The percentage of the training data to save for validation of the neural network
        shuffle: bool, default: False
            To whether shuffle the training samples inside each batch. Theoretically, it is not recommended 
            to shuffle time series data, however, in practice it sometimes yields good results.
        gradient_clip: int or float, default: 1
            Deals with the exploding gradients problem, common in training any type of RNN. Proposed by
            Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training Recurrent Neural Networks.
            ArXiv:1211.5063 [Cs]. https://arxiv.org/abs/1211.5063
        print_every: int, default: 1
            Step at which information about the training is outputted.

        Returns
        -------
        The trained model, the training loss history, the validation loss history (if a validation 
            portion is set) and all the hyperparameters used to train the neural net.
        """
        if folder is None:
            folder = os.getcwd()

        f = open(os.path.join(
            folder, f'{model_name} model training report.txt'), 'w')

        # standard hyperparameter values
        input_size = 1
        output_size = 1
        n_layers = 2

        if model_name.lower() == 'rnn':
            model = TradRNN(device='cpu', input_size=input_size, hidden_size=hidden_dim,
                            output_size=output_size, seq_len=seq_len, n_layers=n_layers)

        if model_name.lower() == 'gru':
            model = GRU(device='cpu', input_size=input_size, hidden_size=hidden_dim,
                        output_size=output_size, seq_len=seq_len, n_layers=n_layers)

        if model_name.lower() == 'lstm':
            model = LSTM(device='cpu', input_size=input_size, hidden_size=hidden_dim,
                         output_size=output_size, seq_len=seq_len, n_layers=n_layers)

        # evaluation criterion
        criterion = nn.MSELoss()
        # optimizer definition
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if validation_split:
            v_split = int(validation_split*len(data_x))

            train_x, val_x = data_x[:-v_split], data_x[-v_split:]
            train_y, val_y = data_y[:-v_split], data_y[-v_split:]

            training_data = TensorDataset(
                torch.from_numpy(train_x), torch.from_numpy(train_y))
            train_loader = train_loader = DataLoader(
                training_data, shuffle=shuffle, batch_size=batch_size, drop_last=True)

            validation_data = TensorDataset(
                torch.from_numpy(val_x), torch.from_numpy(val_y))
            valid_loader = DataLoader(
                validation_data, shuffle=shuffle, batch_size=batch_size, drop_last=True)

            train_hist = np.zeros(epochs+1)
            valid_hist = np.zeros(epochs+1)
            print('----------------------------------------', file=f)
            print(f"Starting Training of {model_name} model", file=f)
            print('----------------------------------------', file=f)
            for epoch in range(1, epochs + 1):

                start_time = time.time()

                for input_seq, label in train_loader:

                    # get the output of the NN
                    out = model(input_seq.float())
                    # calculate the gradient based on the loss criterion
                    loss = criterion(out, label.float())

                    for valid_seq, valid_label in valid_loader:
                        with torch.no_grad():
                            valid_out = model(valid_seq.float())
                            valid_loss = criterion(
                                valid_out, valid_label.float())

                    # store validation loss for plot
                    valid_hist[epoch] = valid_loss.item()

                # store the train loss for plot
                train_hist[epoch] = loss.item()
                current_time = time.time()

                if epoch % print_every == 0:
                    print(
                        'Epoch {}/{} \nTrain loss: {:.6f} Validation loss: {:.6f}'.format(epoch,epochs,loss.item(),valid_loss.item()), file=f)

                # reset acumulated gradients
                optimizer.zero_grad()

                # backpropagate gradients
                loss.backward()

                # clip gradient to prevent exploding gradients problem
                if gradient_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                # update weights
                optimizer.step()

            current_time = time.time()
            print(
                f'\nTotal training time: {str(current_time - start_time)} seconds.')
            return model.train(), train_hist[1:], valid_hist[1:]

        else:
            training_data = TensorDataset(
                torch.from_numpy(data_x), torch.from_numpy(data_y))
            train_loader = DataLoader(
                training_data, shuffle=shuffle, batch_size=batch_size, drop_last=True)

            train_hist = np.zeros(epochs+1)

            for epoch in range(1, epochs + 1):

                start_time = time.time()

                for input_seq, label in train_loader:

                    # get the output of the NN
                    out = model(input_seq.float())
                    # calculate the gradient based on the loss criterion
                    loss = criterion(out, label.float())

                    # store the train loss for plot
                    train_hist[epoch] = loss.item()
                    epoch_current_time = time.time()

                if epoch % print_every == 0:
                    print('Epoch {}/{} Train loss: {:.6f}'.format(epoch, epochs, loss.item()), file=f)
                    print(
                        f'Epoch duration: {epoch_current_time - start_time} seconds', file=f)
                    # reset acumulated gradients
                    optimizer.zero_grad()

                    # backpropagate gradients
                    loss.backward()

                    # clip gradient to prevent exploding gradients problem
                    if gradient_clip is not None:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip)

                    # update weights
                    optimizer.step()

            current_time = time.time()
            print(
                f"\nTotal training time: {str(current_time-start_time)} seconds.", file=f)
            
            f.close()
            return model.train(), train_hist[1:]


    @staticmethod
    def test_predictions(model, train_x, scaler_object, test_data, evaluation_metrics):
        """
        Used to make predictions for the test period and to evaluate how far the point predicitons
        are from the the real values in the test set.

        Arguments
        ---------
        model: pytorch model
            A previously instanced and trained model.
        train_x: numpy.array
            The normalized windowed sequences that were used to train the model. 
            Must have 3D shape (batch_size, seq_len, num_features)
        scaler_object: MinMaxScaler
            Instance of the MinMaxScaler object used to normalize the training data.
        test_data: pd.DataFrame or pd.Series
            The original testing data to which the predicted values will be compared.
        evaluation_metrics: list of strs
            Accepts the following values: 'mse', 'rmse', 'mae', 'mdae'.

        Returns
        -------
        A pandas DataFrame with the real and predicted values. 
        A pandas DataFrame containing the values for each of the passed metrics.
        """

        with torch.no_grad():
            model.eval()

            predictions = []

            first_eval_batch = train_x[-1, :, :]
            current_batch = first_eval_batch.reshape(
                1, -1, first_eval_batch.shape[1])
            current_batch = torch.from_numpy(current_batch)

            for i in range(len(test_data)):

                # get prediction
                out = model(current_batch.float())
                # extract it from the tensor
                pred = torch.flatten(out).item()
                # store the prediction
                predictions.append(pred)

                # update the current batch to include the prediction
                current_batch = torch.from_numpy(
                    np.append(current_batch[:, 1:, :], [[[pred]]], axis=1))

            predicted_values = scaler_object.inverse_transform(
                np.expand_dims(predictions, axis=0)).flatten()

        true_and_predicted = test_data.copy()
        true_and_predicted['Predictions'] = predicted_values

        eval_dict = {}

        if 'mse' in evaluation_metrics:
            eval_dict['Mean Square Error'] = PredictionMetrics.mean_squared_error(
                true_and_predicted.iloc[:,0].values, true_and_predicted.iloc[:,1].values)
        if 'rmse' in evaluation_metrics:
            eval_dict['Root Mean Square Error'] = PredictionMetrics.root_mean_squared_error(
                true_and_predicted.iloc[:,0].values, true_and_predicted.iloc[:,1].values)
        if 'mae' in evaluation_metrics:
            eval_dict['Mean Absolute Error'] = PredictionMetrics.mean_absolute_error(
                true_and_predicted.iloc[:,0].values, true_and_predicted.iloc[:,1].values)
        if 'mdae' in evaluation_metrics:
            eval_dict['Median Absolute Error'] = PredictionMetrics.median_absolute_error(
                true_and_predicted.iloc[:,0].values, true_and_predicted.iloc[:,1].values)

        eval_df = pd.DataFrame.from_dict(eval_dict, orient='index', columns=[
                                         'Predictive Performance'])

        return true_and_predicted, eval_df

    @staticmethod
    def save_eval_metrics(model_name, metrics_df, folder):
        """
        """
        with open(os.path.join(folder, f'{model_name} predictive performance.txt'), 'w') as f:
            metrics_df.to_string(f)
        metrics_df.to_csv(os.path.join(folder, f'{model_name} predictive performance.csv'))

    @staticmethod
    def save_test_predictions(model_name, predictions, folder, train_data=None):
        """
        """
        # Save test prediction to a .txt file
        with open(os.path.join(folder, f'{model_name} predictions for {predictions.columns[0]} test set.txt'), 'w') as f:
            predictions.to_string(f)

        # Save test predictions to a .csv file
        predictions.to_csv(os.path.join(folder, f'{model_name} predictions for {predictions.columns[0]} test set.csv'))

        plt.style.use('fivethirtyeight')
        plt.rcParams["figure.figsize"] = (12, 6)

        if train_data is not None:

            plt.plot(train_data.index, train_data,
                     label='Train set (historical) values', lw=2)

            plt.plot(predictions.index, predictions.iloc[:,0],
                     label='Test set (real) values', lw=2)

            plt.plot(
                predictions.index,
                predictions.iloc[:, 1],
                label='Predictions', lw=2)
            plt.legend(loc='upper left')
            plt.title(f'Predictions for {predictions.columns[0]}')
            plt.xlabel('Dates')
            plt.ylabel('Values')
            plt.tight_layout()

        else:
            
            plt.plot(predictions.index, predictions.iloc[:,0],
                     label='Test set (real) values', lw=2)

            plt.plot(
                predictions.index,
                predictions.iloc[:, 1],
                label='Predictions', lw=2)
            plt.legend(loc='upper left')
            plt.title(f'Predictions for {predictions.columns[0]}')
            plt.xlabel('Dates')
            plt.ylabel('Values')
        
        plt.savefig(os.path.join(folder, f'{predictions.columns[0]} test predictions.png'))
        plt.close()

    @staticmethod
    def forecast(model, data_x, scaler_object, n_periods, confidence=0.95, num_sims=100, multiplier=4):
        """
        Produces the point forecasts and the accompanying prediction intervals based on the approach
        proposed by Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model 
        Uncertainty in Deep Learning". ArXiv:1506.02142 [Cs, Stat]. https://arxiv.org/abs/1506.02142

        Arguments
        ---------
        model: pytorch model
            A previously instantiated and trained model.
        data: numpy.array
            The array of normalized windowed sequences of the data to be forecast.
        scaler_object: MinMaxScaler
            The MinMaxScaler instance used to normalize the data.
        n_periods: int
            The number of future periods to be forecast.
        confidence: float or list of floats, default: 0.95
            The desired confidence level to calculate the prediction intervals of the forecast.
            Coverage probabilites are based on the Normal Distribution.
        num_sims: int, default: 100
            Number of times the forecasts are made in order to capture the variance of the model
            in order to produce prediction intervals.
        multiplier: int, default: 4
            The variance of the model that allows to compute the prediction intervals is due to
            the use of Dropout between the RNN layers. However, the used activation function between
            these layers (tanh) leads to very narrow intervals which do not reflect the real uncertainty
            of the forecasts. Hence, this multiplier is used to augment the coverage of the interval.

        Returns:
            A pandas DataFrame containing the point forecasts and the upper and lower bounds for each 
            of the confidence levels passed to the function. Optionally, it may return a csv file and 
            the plot of the forecast.
        """

        y_hat = []

        for sim in range(num_sims):

            with torch.no_grad():

                predictions = []

                last_data_batch = data_x[-1, :, :]
                last_batch = last_data_batch.reshape(
                    1, -1, last_data_batch.shape[1])
                last_batch = torch.from_numpy(last_batch)

                for i in range(n_periods):
                    model.train()

                    # get predicted value
                    out = model(last_batch.float())

                    # extract value from the tensor
                    pred = torch.flatten(out).item()

                    # store prediction
                    predictions.append(pred)

                    # update the batch to include the prediction
                    last_batch = torch.from_numpy(
                        np.append(last_batch[:, 1:, :], [[[pred]]], axis=1))

                predicted_values = scaler_object.inverse_transform(
                    np.expand_dims(predictions, axis=0)).flatten()

            y_hat.append(predicted_values)

        y_hat = np.reshape(y_hat, (num_sims, 1, n_periods))
        y_hat_mean = y_hat.mean(axis=0).reshape(-1, 1)
        y_hat_std = y_hat.std(axis=0).reshape(-1, 1)

        # Coverage probabilities for each confidence level (assuming a Normal Distribution)
        confidence_dict = {'0.8': 1.282, '0.85': 1.282,
                           '0.9': 1.645, '0.95': 1.960, '0.99': 2.576}

        upper_bound = {}
        lower_bound = {}

        if isinstance(confidence, list):
            # If a list of confidence levels is passed
            for conf_level in confidence:
                upper_bound[str(conf_level)] = (
                    y_hat_mean + confidence_dict[str(conf_level)]*y_hat_std*multiplier).squeeze()
                lower_bound[str(conf_level)] = (
                    y_hat_mean - confidence_dict[str(conf_level)]*y_hat_std*multiplier).squeeze()
        else:
            # If a single confidence level is passed
            upper_bound[str(confidence)] = (
                y_hat_mean + confidence_dict[str(confidence)]*y_hat_std*multiplier).squeeze()
            lower_bound[str(confidence)] = (
                y_hat_mean - confidence_dict[str(confidence)]*y_hat_std*multiplier).squeeze()

        return y_hat_mean, upper_bound, lower_bound

    @staticmethod
    def nn_forecast_to_df(point_forecasts, upper_bound, lower_bound, model_name, original_data=None, folder=None):
        """
        Converts the point forecasts array and upper and lower bounds dictionaries into a pd.DataFrame

        Arguments
        ----------
        point_forecasts: np.array
            The point forecasts outputed by the forecast() method
        upper_bound: dict
            The dictionary of the confidence levels and corresponding upper bound values
        lower_bound: dict
            The dictionary of the confidence levels and corresponding upper bound values
        original_data: Optional, pd.DataFrame with Datetime index, default: None
            Used to infer a datetime format for the index of the forecasts dataframe.
            If none is given the forecasts will have a numeric index.

        Returns
        -------
        A pandas DataFrame will the point forecasts all the accompanying prediction intervals for
        each given confidence level.
        """
        if original_data is not None:
            fc_dates = pd.date_range(original_data.index.max(), periods=len(
                point_forecasts), freq=pd.infer_freq(original_data.index))
            fc_df = pd.DataFrame(data=point_forecasts,
                                 index=fc_dates, columns=['Predictions'])
        else:
            fc_df = pd.DataFrame(data=point_forecasts, columns=['Predictions'])

        for i, j in zip(upper_bound.keys(), lower_bound.keys()):
            fc_df[f'Upper {int(float(i)*100)} % IC'] = upper_bound[i]
            fc_df[f'Lower {int(float(j)*100)} % IC'] = lower_bound[j]

        if folder is not None:
            fc_df.to_csv(os.path.join(folder, f'{model_name} forecasts.csv'))

            with open(os.path.join(folder, f'{model_name} forecasts.txt'), 'w') as f:
                fc_df.to_string(f)
        
        return fc_df

    @staticmethod
    def plot_nn_forecast(model_name, forecasts, original_data=None, folder=None):
        """
        Creates a plot for the neural network point forecasts and accompanying prediction intervals 
        """
        plt.style.use('fivethirtyeight')
        plt.rcParams["figure.figsize"] = (12, 8)

        if original_data is not None:

            plt.plot(original_data.index, original_data,
                     label='Historical values', lw=2)

            plt.plot(
                forecasts.index,
                forecasts.iloc[:, 0],
                label='Predictions', c='g', lw=2)

            for i in range(1, len(forecasts.columns), 2):
                plt.fill_between(forecasts.index,
                                 forecasts.iloc[:, i],
                                 forecasts.iloc[:, i+1],
                                 color='g', alpha=0.1, lw=0, label=forecasts.columns[i][6:])
            plt.legend(loc='upper left')
            plt.title(
                f'{original_data.columns[0]} Forecasts for future {len(forecasts)} periods'.capitalize())

        else:
            plt.plot(forecasts.index,
                     forecasts.iloc[:, 0], label='Predictions', c='g')

            for i in range(1, len(forecasts.columns), 2):
                plt.fill_between(forecasts.index,
                                 forecasts.iloc[:, i],
                                 forecasts.iloc[:, i+1],
                                 color='g', alpha=0.1, lw=0, label=forecasts.columns[i][6:])
            plt.legend(loc='upper left')
            plt.title(
                f'Forecasts for future {len(forecasts)} periods'.capitalize())

        plt.savefig(os.path.join(folder, f'{model_name} forecasts.png'))

        plt.close()

class NNPlots(object):

    """
    Container class for all the methods related to Neural Network plotting.
    """
    @staticmethod
    def plot_losses(model_name, train_history, valid_history=None, folder_to_save=None):
        """
        Plots the training and validation losses over each epoch.
        """
        plt.style.use('fivethirtyeight')
        plt.rcParams["figure.figsize"] = (12, 8)

        if valid_history is not None:

            plt.plot(train_history, label='Training Loss', lw=2)
            plt.plot(valid_history, label='Validation Loss', lw=2)
            plt.legend(loc='upper left')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.title('Loss per epoch')
            plt.savefig(os.path.join(folder_to_save, f'{model_name.upper()} train and validation losses.png'))
            plt.close()
        else:

            plt.plot(train_history, label='Training Loss')
            plt.legend(loc='upper left')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.title('Loss per epoch')
            plt.savefig(os.path.join(folder_to_save, f'{model_name.upper()} train loss.png'))
            plt.close()
        