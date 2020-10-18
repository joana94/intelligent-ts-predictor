from diffs_class import Diffs
import itertools
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


class BoxJenkins(object):
    """
    Class that contains all the logic for identifying, estimating, diagnosting and forecasting time-series
    by using the Box-Jenkins models, namely, ARIMA, SARIMA and (S)ARIMA with exogenous predictors.
    """
    
    models = ('ARIMA', 'SARIMA')
    
    def __init__(self, model, data, target, m=1, exogenous = None):
        """
        Arguments
        ---------
        model: str
            Name of the model to use: ARIMA or SARIMA
        data : pandas DataFrame 
            The passed DataFrame must have a datetime index
        target: str
            The name (contained in the passed DataFrame) of the variable to be predicted
        m: int, default: 1
            Seasonal period of the time series. For monthly data it is usually 12, for quarterly data it is 4, etc.
        exogenous: str or list of strs, optional, default: None
            The name of any useful exogenous variables (contained in the passed DataFrame) to predict the target time series
        """

        self._model = model
        self.data = data
        self.target = target
        self.m = m
        if self.m == 1:
            self._model = BoxJenkins.models[0]
            print("> Model set to ARIMA, because the passed seasonal period, m, is equal to 1.")
            print("> In order to use SARIMA, the seasonal period, m, must be greater than 1.\n")
        else:
            self._model = model
        self.exogenous = exogenous
    
    # @property
    # def model_name(self):
    #     return self._model
        
    # @model_name.setter
    # def model_name(self, model_name):
    #     if model_name not in BoxJenkins.models:
    #         print(f"ERROR: That's not a valid model! You can either choose {BoxJenkins.models[0]} or {BoxJenkins.models[1]}.")
    #     self._model = model_name
    

    def _search_best_model(self, metric='aic', max_p = 3, d='auto', max_q = 3, 
                          seasonal = False, m=1, max_P = 2, D = 'auto', max_Q = 2, 
                          exogenous = None, trace=True):
        
        if self._model == BoxJenkins.models[0] or self.m ==1:
            p = range(max_p+1)
            q = range(max_q+1)
            
            if d =='auto':
                d = [Diffs.num_diffs(data)]
            else:
                d = [d]
            
            nonseasonal_params = list(itertools.product(p,d,q))
            
            counter = 0
            models = {}
            
            for parameter in nonseasonal_params:
                try:
                    counter += 1
                    model = sm.tsa.statespace.SARIMAX(data,
                                                    order=parameter,
                                                    seasonal_order=None,
                                                    exog = self.exogenous,
                                                    enforce_stationarity=True,
                                                    enforce_invertibility=True)
                    results = model.fit(disp=0)
                    
                    models[counter] = [parameter, results.aic, results.aicc]
                    
                    if trace is True:
                        print(f'ARIMA{parameter} - AIC:{results.aic} | AICc:{results.aicc}')
                    else:
                        pass
                except:
                    continue
                
                df_models = pd.DataFrame.from_dict(models, orient='index')
                
            if metric == 'aic':
                best_model = df_models[df_models.iloc[:,2] == df_models.iloc[:,2].min()].values
                print(f'-----------------------------------------------------')
                print(f'Best model: ARIMA{best_model[0][0]} - AIC: {best_model[0][2]}')

            if metric == 'aicc':
                print(f'-----------------------------------------------------')
                best_model = df_models[df_models.iloc[:,3] == df_models.iloc[:,3].min()].values
                print(f'Best model: ARIMA{best_model[0][0]} - AICc: {best_model[0][3]}')
            
            return best_model
        
        if self._model == BoxJenkins.models[1]:
            
            p = range(max_p+1)
            q = range(max_q+1)
    
            if d == 'auto':
                d = [Diffs.num_diffs(data, seasonal=True, m=self.m)]
            else:
                d = [d]

            nonseasonal_params = list(itertools.product(p,d,q))

            # Seasonal parameters SARIMA(p, d , q) (P, D, Q)m
            P = range(max_P + 1)
            Q = range(max_Q + 1)
            if D == 'auto': #range(seasonal_d+1)
                D = [Diffs.num_sdiffs(data, self.m)]
            else:
                D = [D]

            seasonal_params = [(x[0], x[1], x[2], self.m) for x in list(itertools.product(P,D,Q))]

            counter = 0
            models = {}

            for parameter in nonseasonal_params:
                for seasonal_parameter in seasonal_params:
                    try:
                        counter += 1
                        model = sm.tsa.statespace.SARIMAX(data,
                                                        order=parameter,
                                                        seasonal_order=seasonal_parameter,
                                                        exog = self.exogenous,
                                                        enforce_stationarity=True,
                                                        enforce_invertibility=True)
                        results = model.fit(disp=0)
                        models[counter] = [parameter, seasonal_parameter, results.aic, results.aicc]

                        if trace is True:
                            print(f'SARIMA{parameter}x{seasonal_parameter} - AIC:{results.aic} | AICc:{results.aicc}')
                        else:
                            pass
                        
                    except:
                        continue

                    df_models = pd.DataFrame.from_dict(models, orient='index')

            if metric == 'aic':
                best_model = df_models[df_models.iloc[:,2] == df_models.iloc[:,2].min()].values
                print(f'-----------------------------------------------------------------')
                print(f'Best model: SARIMA{best_model[0][0]}x{best_model[0][1]} - AIC: {best_model[0][2]}')
                print(f'-----------------------------------------------------------------')
                
            if metric == 'aicc':
                best_model = df_models[df_models.iloc[:,3] == df_models.iloc[:,3].min()].values
                print(f'-----------------------------------------------------------------')
                print(f'Best model: SARIMA{best_model[0][0]}x{best_model[0][1]} - AICc: {best_model[0][3]}')
                print(f'-----------------------------------------------------------------')
                
            return best_model
     
    def fit(self, metric='aic', max_p = 3, d='auto', max_q = 3, seasonal = False, m=1, max_P = 2, D = 'auto', max_Q = 2, 
            exogenous = None, trace=True):
        
        model = BoxJenkins(self._model, self.data, self.target, self.m, self.exogenous)
        best_model = model._search_best_model(metric = metric, max_p=max_p, d=d, max_q=max_q, seasonal = seasonal, m=m, max_P=max_P,
                                            D = D, max_Q = max_Q, exogenous = None, trace=trace)

        if self._model == BoxJenkins.models[0] or self.m ==1:
            fitted_model = sm.tsa.statespace.SARIMAX(self.data,
                                              order = best_model[0][0],
                                              seasonal_order=None,
                                              exog = self.exogenous,
                                              enforce_stationarity=True,
                                              enforce_invertibility=True)
            results = fitted_model.fit(disp=0)

        else:
            fitted_model = sm.tsa.statespace.SARIMAX(self.data,
                                              order = best_model[0][0],
                                              seasonal_order=best_model[0][1],
                                              exog = self.exogenous,
                                              enforce_stationarity=True,
                                              enforce_invertibility=True)
            results = fitted_model.fit(disp=0)
        
        return results

    def predict(self, n_periods, confidence):

        pass
                
    def __repr__(self):
        return f"Model: {self._model} \nTarget variable: {self.target} \nSeasonal period: {self.m} \nExogenous variables: {self.exogenous}"
        
data = pd.read_csv(r"C:\Users\joana\Documents\TESE 2020\Datasets/airline-passengers.csv", index_col=[0], parse_dates=True)


model = BoxJenkins(model='SARIMA', data=data, m=1, target='Target variable')

print(model)

#results = model.fit()

# print(results.summary())

