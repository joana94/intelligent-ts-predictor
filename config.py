"""
-----------------------
GENERAL CONFIGURATIONS
-----------------------
"""
# GLOBALS used by the system

SPLIT_POINT = 24 # -> How many timesteps to save for the test set.
                 # -> If an integer value, e.g. 24, it means the test set will have the 
                 # last 24 timesteps of the dataset, and the training set will have the remaining
                 # -> If a decimal value, e.g. 0.1, it means the test set will have 10% of the 
                 # (most recent) timesteps in the dataset, and the training set will have the remaining

N_PERIODS = 12   # -> How many periods into the future to produce forecasts for

"""
------------------------
CHOOSE FORECASTING MODEL
------------------------
"""
# There are two categories of models to choose from: Statistical and Neural Networks

# The available statistical models are:
# -> 'ARIMA' for non seasonal time series
# -> 'SARIMA' for seasonal time series

# The available neural network models are:
# -> 'TradRNN' 
# -> 'GRU'
# -> 'LSTM'

# The system automatically chooses the best set of hyperparameters for each selected model
# and it also generates evaluation metrics for the predictions made for the test set which
# allows to easily compare the performance of each model. Hence, a good approach might be
# to iteratively change the model type and at the end check which has performed better in
# the test set and choose that one two produce actual forecasts.

MODEL = 'ARIMA'


"""
----------------------------------------------
ARIMA AND SARIMA MODEL SPECIFIC CONFIGURATIONS
----------------------------------------------
"""

SEASONAL = False # -> Whether the time series has a seasonal pattern or not
                 # -> Possible values: True, False
                 # -> MUST BE True WHEN THE CHOOSEN MODEL IS SARIMA or it will
                 # automatically default to ARIMA    

M = 1  # -> The seasonal period of the time series. Only required if SEASONAL = True
       # -> It is an integer value. Common seasonal periods:
       # -> M = 7 for daily data
       # -> M = 12 for monthly data
       # -> M = 4 for quarterly data 
       # -> MUST BE GREATER THAN 1 WHEN THE CHOOSEN MODEL IS SARIMA or it will
       # automatically default to ARIMA

# PARAMETERS THAT DON'T NEED TO (BUT CAN) BE CHANGED:

metric = 'aic' # -> The metric to be minimized on which the best set of hyperparameters is chose
               # -> Possible values: 'aic' or 'aicc'
               # -> 'aicc' is preferred for small samples

max_p = 3      # -> The maximum p order of the non-seasonal AR component that will be tried
               # -> Values bigger than 3 are not recommended

d = 'auto'     # -> The order d of integration (equivalent to the number of first differences)
               # -> Possible values: 'auto', 1 or 2 (bigger values than 2 are not recommended)
               # -> When 'auto' it automatically finds the correct order through KPSS tests

max_q = 3      # -> The maximum q order of the non-seasonal MA component that will be tried
               # -> Values bigger than 3 are not recommended

max_P = 2      # -> The maximum p order of the seasonal AR component that will be tried
               # -> Values bigger than 2 are not recommended

D = 'auto'     # -> The order d of integration (equivalent to the number of first differences)
               # -> Possible values: 'auto', 1 or 2 (bigger values than 2 are not recommended)
               # -> When 'auto' it automatically finds the correct order through the seasonal strength index

max_Q = 2      # -> The maximum q order of the seasonal MA component that will be tried
               # -> Values bigger than 2 are not recommended


"""
---------------------------------------------------
TradRNN, GRU and LSTM MODEL SPECIFIC CONFIGURATIONS
---------------------------------------------------
"""
# Neural Networks are a supervised learning data mining algorithm. For that reason,
# time series data need to be previously pre-processed and converted into a format
# ingestible by the NNs, more concretely, they need to be transformed into windows 
# of sequences and their respective labels. That transformation is parameterized by
# the SEQ_LEN parameter.

SEQ_LEN = 12 # -> This value is data-specific, i.e., it depends on the time series 
             # -> to forecast. It basically means, how many previous timesteps are
             # -> used in order to predict the immediate next value. E.g., if
             # -> SEQ_LEN = 4 and a time series with 12 timesteps it will work as:
             # -> Time Series : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
             # -> First window and label:  [1, 2, 3, 4] -> [5]
             # -> Second window and label: [2, 3, 4, 5] -> [6]
             # -> Third window and label:  [3, 4, 5, 6] -> [7]
             # -> And it will proceed like this until it reaches the end of the ts.

VALID_SPLIT = 0.1 # -> The fraction of the train set to be saved for validation.
                  # -> It must be bigger than zero (and less than 1). It is an
                  # -> essential part for the system to find the best Neural Net
                  # -> architecture from the given set of hyperparameters

NUM_EPOCHS = 5 # -> The number of times that the windowed sequences and labels
                 # -> are inputed to the Neural Net in order to backpropagate the
                 # -> gradients and update the weights towards the minimization of
                 # -> loss function (the Mean Square Error in this particular case)

# The best Neural Net architecture will be found based on the following set of hyperparameters
# The values inside the [ ] are the ones that can be changed in order to test different
# architectures. There must be at least one value inside each []. If only one value is 
# passed to each set of [], it is equivalent to be directly choosing the NN architecture.

HYPERPARAMETERS = {
      'HIDDEN_DIM': [256, 512],
      'LEARNING_RATE': [0.01, 0.001],
      'BATCH_SIZE': [1, 8, 16],
      'SHUFFLE': [False, True]
}

SEARCH_METHOD = 'random' # -> Possible values: 'grid' or 'random'
                         # -> The 'grid' method will try every possible combination of the values in HYPERPARAMETERS
                         # -> The 'random' method will try random combinations of the values in HYPERPARAMETERS up to
                         # -> a fixed maximum number of iterations, defined by MAX_ITERS below.  When very long lists 
                         # -> of hyperparameters are passed, it is highly recommended to use the 'random' search method
                         # -> due to computational constraints, as using the 'grid' method will be very computationally
                         # -> demanding. 

MAX_ITERS = 1  # -> Only required for the 'random' search method. 
               # -> Note that the value of MAX_ITERS should be less than the total number of possible hyperparameter 
               # -> combinations. Otherwise, it will be less efficient/effective than the 'grid' search method.


"""
---------------------------------------------------
MODEL COMPARISON CONFIGURATIONS
---------------------------------------------------
"""

# If you have fitted more than one model, you can compare each 
# of them based on their predictive performance for the test set
# You just need to spceficiy the names of the models you fitted
# in the below global variable MODELS

MODELS = ['GRU', 'ARIMA'] # -> You can add any of the models: ARIMA, SARIMA, TradRNN, GRU or LSTM
                          # -> as long as you have fitted them to the data

