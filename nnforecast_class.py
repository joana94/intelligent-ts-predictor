import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rnn_classes import TradRNN, GRU, LSTM
import time
from metrics_class import PredictionMetrics

class NNForecast(object):

    """
    Container class for methods related to the use of the RNN classes for time series forecasting.
    Included methods: train, evaluate and forecast.
    """
    @staticmethod
    def train(model_name, data_x, data_y, batch_size, hidden_dim, epochs, learning_rate, seq_len, validation_split=None,
              shuffle=False, gradient_clip=None, print_every=1):
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

            print(f"Starting Training of {model_name} model")

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
                        f'Epoch {epoch}/{epochs} \nTrain loss: {loss.item()} Validation loss: {valid_loss.item()}')

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
                    print(f'Epoch {epoch}/{epochs} Train loss: {loss.item()}')
                    print(
                        f'Epoch duration: {epoch_current_time - start_time} seconds')
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
                f"\nTotal training time: {str(current_time-start_time)} seconds.")
            return model.train(), train_hist[1:]

    @staticmethod
    def plot_losses(history):
        """
        Plots the training and validation losses over each epoch.
        """

        if len(history) == 2:

            train_hist = history[0]
            valid_hist = history[1]

            plt.plot(train_hist, label='Training Loss')
            plt.plot(valid_hist, label='Validation Loss')
            plt.legend()

            return

        else:

            plt.plot(history[0], label='Training Loss')
            plt.legend()

            return

    @staticmethod
    def test_prediction(model, train_x, scaler_object, test_data, evaluation_metrics):
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

        true_and_predicted_values = test_data.copy()
        true_and_predicted_values['Predictions'] = predicted_values
        
        eval_dict = {}

        if 'mse' in evaluation_metrics:
            eval_dict['Mean Square Error'] = PredictionMetrics.mean_squared_error(
                test_data.values, predicted_values.values)
        if 'rmse' in evaluation_metrics:
            eval_dict['Root Mean Square Error'] = PredictionMetrics.root_mean_squared_error(
                test_data.values, predicted_values.values)
        if 'mae' in evaluation_metrics:
            eval_dict['Mean Absolute Error'] = PredictionMetrics.mean_absolute_error(
                test_data.values, predicted_values.values)
        if 'mdae' in evaluation_metrics:
            eval_dict['Median Absolute Error'] = PredictionMetrics.median_absolute_error(
                test_data.values, predicted_values.values)

        eval_df = pd.DataFrame.from_dict(eval_dict, orient='index', columns=[
                                         'Predictive Performance']).T

        return true_and_predicted_values, eval_df

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

        upper_bound = []
        lower_bound = []

        if isinstance(confidence, list):
            # if a list of more than one confidence level is passed
            for conf_level in confidence:
                # Computes the upper and lower bounds for each of the passed confidence levels
                upper_bound.append(
                    (y_hat_mean + confidence_dict[str(conf_level)]*y_hat_std*multiplier).squeeze())
                lower_bound.append(
                    (y_hat_mean - confidence_dict[str(conf_level)]*y_hat_std*multiplier).squeeze())
        else:
            # If a single confidence level is passed
            upper_bound = (
                y_hat_mean + confidence_dict[str(confidence)]*y_hat_std*multiplier).squeeze()
            lower_bound = (
                y_hat_mean - confidence_dict[str(confidence)]*y_hat_std*multiplier).squeeze()

        return y_hat_mean, upper_bound, lower_bound, confidence
