# Intelligent TS Predictor (iTSP)

### Description

Intelligent TS Predictor is an automated system for univariate time series prediction (forecasting).  Besides automatically finding the optimal set of hyperparameters, it also allows to easily compare the performance of different models. Furthermore, and very important, it is able to produce multiple prediction intervals for forecasts produced by the Neural Network models.

It comprises the following models:
- Statistical
  - ARIMA
  - SARIMA
- Neural Networks
  - Traditional Recurrent Neural Network (RNN)
  - Gated Recurrent Unit (GRU)
  - Long Short-Term Memory (LSTM)
 
  - The ARIMA and SARIMA models use the statespace SARIMAX class from the Statsmodels library as base. The search for the optimal hyperparameters is inspired by the Hyndman and Khandakar (2008) algorithm, that is, the number of required non-seasonal differences are found through the application of successive KPSS tests, whereas the required number of seasonal differences are found through the use of the seasonal strength measure proposed by Wang et al. (2006). 
- The neural network models are implemented with the Pytorch library which requires some degree of actual knowledge about the way NNs work but it is very flexible and has many useful tools that help the development process. 
- The NNs were implemented as being "autoregressive", i.e. they predict one time step into the future which is then fed back into the model in order to predict the following time steps.
- The approach to the generation of the prediction intervals is inspired by the work of Gal & Ghahramani (2016), i.e. dropout is used during training and also during inference. During inference, the NN generates at each timestep a high number of predictions/simulations (the default value in the system is 100) and then the mean value is used as point forecasts and the standard deviation is used to compute the prediction intervals (that use the Normal Distribution coverage probabilities). Any of these NNs use the hyperbolic tangent activation function which does not produce forecasts with high variance and, consequently, leads to fairly narrow prediction intervals. In order to overcome that "problem", the system uses a multiplier factor to augment the prediction intervals to a more reasonable width.
 
### Example of a forecast with a LSTM model using the Intelligent TS Predictor system
 
 ![NN_forecasts](https://user-images.githubusercontent.com/23248450/96530982-8ae04700-1280-11eb-8827-eeac63a1ec18.png)

 
### How to use
 
The system is modular and it is executed through the command line. The correct way of using it is as follows:
 
 - First, run the 'create_dirs.py' throgh the command line. It will ask for a name project name and will create directory structure as follows:
:file_folder: (/User/Documents/)
   :file_folder: (../Intelligent TS Predictor/)
     :file_folder: (../"Project Name"/)
         :file_folder: (../Analysis)
              :file_folder: (../Graphics)
      :file_folder: (../Models)
      :file_folder: (../Reports)

  
