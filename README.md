# Intelligent TS Predictor (iTSP)

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
 
## Example of a forecast with a LSTM model using the Intelligent TS Predictor system


![NN_forecasts2](https://user-images.githubusercontent.com/23248450/96621622-c28ed380-1300-11eb-8178-45ec6170d9eb.png)


## Example of output from the search process of an ARIMA model using the Intelligent TS Predictor system

![boxjenkins](https://user-images.githubusercontent.com/23248450/96622483-e6064e00-1301-11eb-9feb-f766bc2e2be5.png)

 
## How to use
 
The system is modular and it is executed through the command line. The correct way of using it is as follows:
 
- First, run the 'create_dirs.py' throgh the command line. It will ask for a project name and will create directory structure as follows:

- :file_folder: / User / Documents/
  - :file_folder: .. / Intelligent TS Predictor /
     - :file_folder: .. / "Project Name" /
         - :file_folder: .. / Data
         - :file_folder: .. / Graphics
         - :file_folder: .. / Models
         - :file_folder: .. / Reports

- Second, insert the time series csv for which want to forecast (or fit models and compare) inside the "Project Name"/Data folder. It must have a **datetime** column.
- Before proceeding, you can check the ``config.py`` which is where you can configure the most important parts of the system to suit your dataset.
- Next you can execute the ``load_analyse_data_module.py`` which will automatically load the data, create a time plot, as well as ACF and PACF plots in the "Project Name"/Graphics folder and will split the dataset into a train and test dataset (based on the parameter SPLIT_POINT in the ``config.py' file``) and store them in the "Project Name"/Data folder.
- The next step is to run the 'model_fitting_module.py' which will find the best set of hyperparameters for the model you chose in the ``config.py``.
- You can use the ``model_fitting_module.py`` to find the hyperparameters for all the models available in the system (ARIMA, SARIMA, Traditional RNN, GRU and LSTM) and compare all of them as the system automatically produces reports inside the "Project Name"/Models/"Model Name" folder with the predictive performance of each in the test set based on the followig measures: Mean Square Error (MSE), Root Mean Square Error (RMSE), Mean Absolute Error (MAE) and Median Absolute Error (MedAE). 
- For faster comparison, you can run the ``model_comparison_module.py`` which will produce a bar plot and a report in "Project Name"/Models comparing the performance of all of them.
- Finally, you can run the ``forecast_module.py`` which will give actual forecasts into the future and their respective plots and .csv files that can be used for further analysis.

This is just a brief overview of how the system works. It has a lot of useful features such as checking missing dates and values and automatically filling them with the correct dates and the values through linear interpolation, automatically performing all the data pre-processing necessary to feed time series into Neural Networks, and so on... 

**The good part is the fact that it abstracts the end user of the system from all the inherent complexity of the time series forecasting process.**


## Future goals
- Refactor the code.
- Add more complex NN models capable of dealing with exogenous features.
