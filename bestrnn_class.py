import time 
import numpy as np
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import namedtuple
from itertools import product
from rnn_classes import TradRNN, GRU, LSTM

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
    def find_architecture(model_name, hyperparameters, search_method, data_x, data_y, epochs, seq_len, max_iters = 5, validation_split=0.1, print_every=1, gradient_clip=1):
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

            
            if model_name.lower() == 'rnn':
                model = TradRNN(device='cpu', input_size=input_size, hidden_size=hyperparam.hidden_dim, 
                                output_size=output_size, seq_len=seq_len, n_layers=n_layers)
        
            if model_name.lower() =='gru':
                model = GRU(device='cpu', input_size = input_size, hidden_size=hyperparam.hidden_dim, 
                            output_size=output_size,seq_len = seq_len, n_layers=n_layers)

            if model_name.lower() =='lstm':
                model = LSTM(device='cpu',input_size=input_size, hidden_size=hyperparam.hidden_dim, 
                            output_size= output_size, seq_len=seq_len, n_layers=n_layers)
            
            print(f"\n\nRun: {iteration}")
            print(f'Model: {model_name.upper()}')
            print(f'{hyperparam}')
    
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
                        print(f'Epoch {epoch}/{epochs} \nTrain loss: {loss.item()} Validation loss: {valid_loss.item()}')
                        print(f'Epoch duration: {epoch_current_time - start_time} seconds')

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
                    print(f'Validation loss decresead from {valid_loss_min} ---> {valid_loss}')
                    valid_loss_min = valid_loss
                    
            current_time = time.time()
            print(f'\nIteration total time: {str(current_time - start_time)} seconds.')
            print(f'Minimum validation loss: {valid_hist[1:].min()}')

            
            if valid_loss_min <= best_score:
                print(f'Global valid loss decreased from {best_score} ---> {valid_loss_min}')
                best_score = valid_loss_min
                best_model = model
                best_model_train_hist = train_hist
                best_model_valid_hist = valid_hist
                best_params = {'model': best_model,
                            'batch_size': hyperparam.batch_size, 'shuffle': hyperparam.shuffle, 
                            'learning_rate': hyperparam.learning_rate, 'hidden_dim': hyperparam.hidden_dim}
                
            iteration += 1
            
        return best_model, best_model_train_hist[1:], best_model_valid_hist[1:], best_params