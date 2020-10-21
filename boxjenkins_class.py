import os
import itertools
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle
import time
from metrics_class import PredictionMetrics
from diffs_class import Diffs
from utils_classes import TimeSeriesPlots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class BoxJenkins(object):
    """
    Class that contains all the logic for identifying, estimating, diagnosting and forecasting time-series
    by using the Box-Jenkins models, namely, ARIMA, SARIMA and (S)ARIMA with exogenous predictors.
    """

    models = ('ARIMA', 'SARIMA')

    def __init__(self, model, data, target, m=1, exogenous=None):
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
        self.target_var = self.data[[self.target]]
        self.m = m
        # If chosen model is ARIMA, then the seasonal period, m, is automatically set to 1
        if self._model == BoxJenkins.models[0]:
            self.m = 1
        # If the chosen model is SARIMA, but the seasonal period, m, is left as 1, then the model is set to ARIMA
        # Because SARIMA needs a seasonal period m > 1 in order to work proprerly
        if self.m == 1 and self._model == BoxJenkins.models[1]:
            self._model = BoxJenkins.models[0]
            print(
                "> Model set to ARIMA, because the passed seasonal period, m, is equal to 1.")
            print(
                "> In order to use SARIMA, the seasonal period, m, must be greater than 1.\n")
        else:
            self._model = model

        # Logic for incorporating one or more exogenous variables
        self.exogenous = exogenous
        # If a list of exogenous variables is passed
        if self.exogenous is not None and isinstance(self.exogenous, list):
            self.exog_vars = self.data[exogenous]
        elif self.exogenous is not None:  # If a single exogenous variable is passed
            self.exog_vars = self.data[[exogenous]]
        else:
            self.exog_vars = None

    def _search_best_model(self, metric, max_p, d, max_q, seasonal, m, max_P, D, max_Q, exogenous, trace, folder):
        """
        Private method that performs a grid search in order to find the optimal model.
        Used trough the .fit() method which also defines the default arguments.

        Arguments:
        ----------
        metric: str
            Possible values are 'aic' or 'aicc'.
        max_p: int, default: 3
            The max order of p of the non-seasonal AR component to search.
        d: str or int, default:'auto'
            If 'auto', the order of d of the non-seasonal I component is found out by applying successive KPSS tests.
            Otherwise it is defined as the integer value passed by the user.
        max_q: int, default: 3
            The max order of q of the non-seasonal MA component to search.
        seasonal : bool, default = False
            If False, a non-seasonal ARIMA is fitted to the date.
            If True, a seasonal ARIMA is fitted to the data.
        m: int, default:1
            The seasonal period. It must be greater than 1 if a SARIMA is fitted to the data.
        max_P: int, default: 2
            The max order of P of the seasonal AR component to search.
        D: str or int, default: 'auto'
            If 'auto', the order of d of the non-seasonal I component is found out by using the seasonal strength index.
            Otherwise it is defined as the integer value passed by the user.
        max_Q: int, default: 2
            The max order of P of the seasonal MA component to search
        exogenous: str, list of strs or None, default = None
            Exogenous variables to include in the model, if any exists. Must be contained in the DataFrame passed to the model.
        trace: bool, default: True
            If True, outputs in the command line, the names, aic and aicc of the models that are being searched.

        Returns
        -------
        An array containing the parameters of the best model that has been found.
        """
        if folder is None:
            folder = os.getcwd()

        f = open(os.path.join(
            folder, f'Best {self._model} search report.txt'), 'w')

        if self._model == BoxJenkins.models[0] or self.m == 1:
            start_time = time.time()
            p = range(max_p+1)
            q = range(max_q+1)

            if d == 'auto':
                d = [Diffs.num_diffs(self.target_var)]
            else:
                d = [d]

            nonseasonal_params = list(itertools.product(p, d, q))

            counter = 0
            models = {}

            if trace:
                print(f'-------------------------------------------------', file=f)
                print(f'FITTED MODELS', file=f)
                print(f'-------------------------------------------------', file=f)

            for parameter in nonseasonal_params:
                try:
                    counter += 1
                    model = sm.tsa.statespace.SARIMAX(self.target_var,
                                                      order=parameter,
                                                      seasonal_order=None,
                                                      exog=self.exog_vars,
                                                      enforce_stationarity=True,
                                                      enforce_invertibility=True)
                    results = model.fit(disp=0)

                    models[counter] = [parameter, None,
                                       results.aic, results.aicc]

                    if trace is True:
                        print('ARIMA{} - AIC:{:.5f} | AICc:{:.5f}'.format(parameter,
                                                                          results.aic, results.aicc), file=f)
                    else:
                        pass
                except:
                    continue

                df_models = pd.DataFrame.from_dict(models, orient='index')

            if metric == 'aic':
                best_model = df_models[df_models.iloc[:, 2]
                                       == df_models.iloc[:, 2].min()].values
                print(f'-------------------------------------------------', file=f)
                print(
                    'Best model: ARIMA{} -> AIC: {:.5f}'.format(best_model[0][0], best_model[0][2]), file=f)
                print(f'-------------------------------------------------', file=f)
                end_time = time.time()
                print(
                    f'\nSearch time: {end_time - start_time} seconds', file=f)

            if metric == 'aicc':
                print(f'-------------------------------------------------', file=f)
                best_model = df_models[df_models.iloc[:, 3]
                                       == df_models.iloc[:, 3].min()].values
                print('Best model: ARIMA{} -> AICc: {:.5f}'.format(
                    best_model[0][0], best_model[0][3]), file=f)
                print(f'-------------------------------------------------', file=f)
                end_time = time.time()
                print(
                    f'\nSearch time: {end_time - start_time} seconds', file=f)

            best_params = {'model':'ARIMA',
                'p': best_model[0][0][0], 'd': best_model[0][0][1], 'q': best_model[0][0][2], 'm': self.m}
            with open(f"{self._model} best parameters.pkl", 'wb') as f:
                pickle.dump(best_params, f)

            return best_model

        if self._model == BoxJenkins.models[1]:
            start_time = time.time()

            p = range(max_p+1)
            q = range(max_q+1)

            if d == 'auto':
                d = [Diffs.num_diffs(self.target_var, seasonal=True, m=self.m)]
            else:
                d = [d]

            nonseasonal_params = list(itertools.product(p, d, q))

            # Seasonal parameters SARIMA(p, d , q) (P, D, Q)m
            P = range(max_P + 1)
            Q = range(max_Q + 1)
            if D == 'auto':  # range(seasonal_d+1)
                D = [Diffs.num_sdiffs(self.target_var, self.m)]
            else:
                D = [D]

            seasonal_params = [(x[0], x[1], x[2], self.m)
                               for x in list(itertools.product(P, D, Q))]

            counter = 0
            models = {}

            if trace:
                print(
                    f'---------------------------------------------------------------------------------', file=f)
                print(f'FITTED MODELS', file=f)
                print(
                    f'---------------------------------------------------------------------------------', file=f)

            for parameter in nonseasonal_params:
                for seasonal_parameter in seasonal_params:
                    try:
                        counter += 1
                        model = sm.tsa.statespace.SARIMAX(self.target_var,
                                                          order=parameter,
                                                          seasonal_order=seasonal_parameter,
                                                          exog=self.exog_vars,
                                                          enforce_stationarity=True,
                                                          enforce_invertibility=True)
                        results = model.fit(disp=0)
                        models[counter] = [parameter,
                                           seasonal_parameter, results.aic, results.aicc]

                        if trace is True:
                            print('SARIMA{}x{} - AIC:{:.5f} | AICc:{:.5f}'.format(parameter,
                                                                                  seasonal_parameter, results.aic, results.aicc), file=f)
                        else:
                            pass

                    except:
                        continue

                    df_models = pd.DataFrame.from_dict(models, orient='index')

            if metric == 'aic':
                best_model = df_models[df_models.iloc[:, 2]
                                       == df_models.iloc[:, 2].min()].values
                print(
                    f'---------------------------------------------------------------------------------', file=f)
                print('Best model: SARIMA{}x{} -> AIC: {:.5f}'.format(
                    best_model[0][0], best_model[0][1], best_model[0][2]), file=f)
                print(
                    f'---------------------------------------------------------------------------------', file=f)
                end_time = time.time()
                print(
                    f'\nSearch time: {end_time - start_time} seconds', file=f)

            if metric == 'aicc':
                best_model = df_models[df_models.iloc[:, 3]
                                       == df_models.iloc[:, 3].min()].values
                print(
                    f'---------------------------------------------------------------------------------', file=f)
                print('Best model: SARIMA{}x{} -> AICc: {:.5f}'.format(
                    best_model[0][0], best_model[0][1], best_model[0][3]), file=f)
                print(
                    f'---------------------------------------------------------------------------------', file=f)
                end_time = time.time()
                print(
                    f'\nSearch time: {end_time - start_time} seconds', file=f)

            best_params = {'model': 'SARIMA','p': best_model[0][0][0], 'd': best_model[0][0][1], 'q': best_model[0][0][2],
                           'P': best_model[0][1][0], 'D': best_model[0][1][1], 'Q': best_model[0][1][2], 'm':self.m }

            with open(f"{self._model} best parameters.pkl", 'wb') as f:
                pickle.dump(best_params, f)
            
            f.close()
            return best_model

    def fit(self, metric='aic', max_p=3, d='auto', max_q=3, seasonal=False, m=1, max_P=2, D='auto', max_Q=2,
            exogenous=None, trace=True, enforce_stationarity=True, enforce_invertibility=True, folder=None):
        """
        Wraps the _search_best_model() method to find the best hyperparameter combination.
        """

        model = BoxJenkins(self._model, self.data,
                           self.target, self.m, self.exogenous)
        best_m = model._search_best_model(metric=metric, max_p=max_p, d=d, max_q=max_q, seasonal=seasonal, m=m, max_P=max_P,
                                          D=D, max_Q=max_Q, exogenous=None, trace=trace, folder=folder)

        if self._model == BoxJenkins.models[0] or self.m == 1:
            start_time_fit = time.time()
            fitted_model = sm.tsa.statespace.SARIMAX(self.target_var,
                                                     order=best_m[0][0],
                                                     seasonal_order=None,
                                                     exog=self.exog_vars,
                                                     enforce_stationarity=True,
                                                     enforce_invertibility=True)
            results = fitted_model.fit(disp=0, low_memory=True)
            end_time_fit = time.time()
            print(f'\n> Fitting time: {end_time_fit - start_time_fit} seconds')

        else:
            start_time_fit = time.time()
            fitted_model = sm.tsa.statespace.SARIMAX(self.target_var,
                                                     order=best_m[0][0],
                                                     seasonal_order=best_m[0][1],
                                                     exog=self.exog_vars,
                                                     enforce_stationarity=True,
                                                     enforce_invertibility=True)
            results = fitted_model.fit(disp=0, low_memory=True)
            end_time_fit = time.time()
            print(f'\n>Fitting time: {end_time_fit - start_time_fit} seconds')

        with open("fitted_model.pkl", 'wb') as f:
            pickle.dump(results, f)

        return results


    def fit_to_entire_dataset(self, p, d, q, seasonal=False, m=1, P=None, D=None, Q=None, folder=None):
        """
        """
        if self._model == BoxJenkins.models[0] or self.m == 1:
            start_time_fit = time.time()
            fitted_model = sm.tsa.statespace.SARIMAX(self.target_var,
                                                     order=(p, d, q),
                                                     seasonal_order=None,
                                                     exog=None,
                                                     enforce_stationarity=True,
                                                     enforce_invertibility=True)
            results = fitted_model.fit(disp=0, low_memory=True)
            end_time_fit = time.time()
            print(f'\n> Fitting time: {end_time_fit - start_time_fit} seconds')

        else:
            start_time_fit = time.time()
            fitted_model = sm.tsa.statespace.SARIMAX(self.target_var,
                                                     order=(p, d, q),
                                                     seasonal_order=(
                                                         P, D, Q, m),
                                                     exog=None,
                                                     enforce_stationarity=True,
                                                     enforce_invertibility=True)
            results = fitted_model.fit(disp=0, low_memory=True)
            end_time_fit = time.time()
            print(f'\n>Fitting time: {end_time_fit - start_time_fit} seconds')
      
        if folder is not None:
            TimeSeriesPlots.box_jenkins_diagnostics(model_name=self._model, results=results, folder=folder)
            BoxJenkins.box_jenkins_method_report(model_name=self._model, results=results, metric=None, folder=folder)
            print(f'> Box-Jenkins method report and diagnostics plots saved to {folder}')
        
        return results


    @staticmethod
    def evaluate(test_data, evaluation_metrics, exog=None):
        """
        Arguments:
        ----------
        test_data: pandas DataFrame
            Portion of data reserved for testing the model
        evaluation_metrics: list of strs
            Accepted values: mse, rmse, mae, mdae
        """

        with open("fitted_model.pkl", 'rb') as f:
            best_model = pickle.load(f)

        eval_dict = {}

        start = test_data.index[0]
        end = test_data.index[-1]

        in_sample_preds = best_model.get_prediction(start=start, end=end)
        test_and_preds = test_data.copy()
        test_and_preds.columns = ['Test set values']
        test_and_preds['Predictions'] = in_sample_preds.predicted_mean

        # pred_values = in_sample_preds.predicted_mean.to_frame(
        #     name='Predictions for test set')

        if 'mse' in evaluation_metrics:
            eval_dict['Mean Square Error'] = PredictionMetrics.mean_squared_error(
                test_and_preds.iloc[:, 0].values, test_and_preds.iloc[:, 1].values)
        if 'rmse' in evaluation_metrics:
            eval_dict['Root Mean Square Error'] = PredictionMetrics.root_mean_squared_error(
                test_and_preds.iloc[:, 0].values, test_and_preds.iloc[:, 1].values)
        if 'mae' in evaluation_metrics:
            eval_dict['Mean Absolute Error'] = PredictionMetrics.mean_absolute_error(
                test_and_preds.iloc[:, 0].values, test_and_preds.iloc[:, 1].values)
        if 'mdae' in evaluation_metrics:
            eval_dict['Median Absolute Error'] = PredictionMetrics.median_absolute_error(
                test_and_preds.iloc[:, 0].values, test_and_preds.iloc[:, 1].values)

        eval_df = pd.DataFrame.from_dict(eval_dict, orient='index', columns=[
                                         'Predictive Performance'])

        return test_and_preds, eval_df

    @staticmethod
    def save_eval_metrics(model_name, metrics_df, folder):
        """
        """
        with open(os.path.join(folder, f'{model_name} predictive performance.txt'), 'w') as f:
            metrics_df.to_string(f)
        metrics_df.to_csv(os.path.join(
            folder, f'{model_name} predictive performance.csv'))

    @staticmethod
    def save_test_predictions(model_name, predictions, folder, train_data=None):
        """
        """
        # Save test prediction to a .txt file
        with open(os.path.join(folder, f'{model_name} predictions for {predictions.columns[0]} test set.txt'), 'w') as f:
            predictions.to_string(f)

        # Save test predictions to a .csv file
        predictions.to_csv(os.path.join(
            folder, f'{model_name} predictions for {predictions.columns[0]} test set.csv'))

        plt.style.use('fivethirtyeight')
        plt.rcParams["figure.figsize"] = (12, 8)

        if train_data is not None:

            plt.plot(train_data.index, train_data,
                     label='Train set (historical) values', lw=2)

            plt.plot(predictions.index, predictions.iloc[:, 0],
                     label='Test set (real) values', lw=2)

            plt.plot(
                predictions.index,
                predictions.iloc[:, 1],
                label='Predictions', lw=2)
            plt.legend(loc='upper left')

        else:

            plt.plot(predictions.index, predictions.iloc[:, 0],
                     label='Test set (real) values', lw=2)

            plt.plot(
                predictions.index,
                predictions.iloc[:, 1],
                label='Predictions', lw=2)
            plt.legend(loc='upper left')

        plt.savefig(os.path.join(
            folder, f'{predictions.columns[0]} test predictions'))
        plt.close()

    @staticmethod
    def forecast(model, n_periods=1, confidence=0.95):

        preds = model.get_forecast(n_periods)
        mean_pred = preds.predicted_mean.to_frame(name='Predictions')
        conf_ic = preds.conf_int(alpha=(1-confidence))
        conf_ic = conf_ic.rename(columns={
                                 conf_ic.columns[0]: f'Lower {confidence}% PI', conf_ic.columns[1]: f'Upper {confidence}% PI'})
        pred_df = mean_pred.join(conf_ic)

        return pred_df

    @staticmethod
    def save_forecast_plot(forecast_df, folder, model_name, original_data=None):
        """
        """
        plt.style.use('fivethirtyeight')
        plt.rcParams["figure.figsize"] = (12, 6)

        if original_data is not None:

            plt.plot(original_data.index, original_data,
                     label='Historical values', lw=2)

            plt.plot(
                forecast_df.index,
                forecast_df.iloc[:, 0],
                label='Forecasted values', lw=2)

            plt.fill_between(forecast_df.index,
                             forecast_df.iloc[:, 1],
                             forecast_df.iloc[:, 2],
                             alpha=0.1, lw=0, label=forecast_df.columns[1][6:])

            plt.legend(loc='upper left')
            plt.title(
                f'{original_data.columns[0]} Forecasts for future {len(forecast_df)} periods'.capitalize())

        else:
            plt.plot(
                forecast_df.index,
                forecast_df.iloc[:, 0],
                label='Forecasted values', lw=2)

            plt.fill_between(forecast_df.index,
                             forecast_df.iloc[:, 1],
                             forecast_df.iloc[:, 2],
                             alpha=0.2, lw=0)

            plt.legend(loc='upper left')
            plt.title(
                f'{original_data.columns[0]} Forecasts for future {len(forecast_df)} periods'.capitalize())

        plt.savefig(os.path.join(folder, f'{model_name} forecasts.png'))

        plt.close()

    @staticmethod
    def save_forecast_files(predictions, folder, model_name):
        """
        Save predictions dataframe to csv file.
        """
        predictions.to_csv(os.path.join(folder, f'{model_name} forecasts.csv'))
        with open(os.path.join(folder, f'{model_name} forecasts.txt'), 'w') as f:
            predictions.to_string(f)

    @staticmethod
    def load_best_params(model_name):
        """
        Read back the best parameters
        """
        with open(os.path.join(os.getcwd(), f'{model_name} best parameters.pkl'), 'rb') as f:
            best_p  = pickle.load(f)
        return best_p

    @staticmethod
    def ljung_box(results, significance=0.05, period=None):
        """
        Determines if the residuals of a model are independently distribuded.
        H0: The residuals are not serially correlated.
        H1: The residuals are autocorrelated.

        Returns:
        --------
        0 if the null hypothesis cannot be rejected
        1 if the null hypothesis is rejected for the given significance level
        """
        deg_free = results.df_model
        result = sm.stats.diagnostic.acorr_ljungbox(results.resid, return_df=True, period=period).iloc[deg_free-1,1]
        if result > significance:
            return result, 0
        return result, 1

    @staticmethod
    def box_jenkins_method_report(model_name, results, metric= None, folder=None):
        """
        """
        if folder is None:
            folder = os.getcwd()

        f = open(os.path.join(folder, 'Box Jenkins Method Report.txt'), 'w')
        order = results.specification['order']
        seas_order = results.specification['seasonal_order']
        sig = 0.05
        
        print('------------------------------------------------------------------------------', file=f)
        print("                        1. IDENTIFICATION", file=f)
        print('------------------------------------------------------------------------------', file=f)

        if model_name == 'ARIMA':
            print(f'\n Model: {model_name} {order}',file=f)

            print(f'\n Description:', file=f)
            print('\n The model uses:', file=f)
            print(f' -> p: {order[0]} lags for the non-seasonal AR component: AR({order[0]})',file=f)
            print(f' -> d: {order[1]} non-seasonal difference(s): I({order[1]})', file=f)
            print(f' -> q: {order[2]} lags for the non-seasonal MA component: MA({order[2]})', file=f)

            lb_test = BoxJenkins.ljung_box(results, significance=sig)

        elif model_name == 'SARIMA':
            print(f'\n Model: {model_name} {order}x{seas_order}', file=f)
            print(f'\n Description:', file=f)
            print('\n The model uses:', file=f)
            print(f' -> p: {order[0]} lags for the non-seasonal AR component: AR({order[0]})', file=f)
            print(f' -> d: {order[1]} non-seasonal difference(s): I({order[1]})', file=f)
            print(f' -> q: {order[2]} lags for the non-seasonal MA component: AR({order[2]})', file=f)
            print(f' -> P: {seas_order[0]} lags for the non-seasonal AR component: AR({seas_order[0]})', file=f)
            print(f' -> D: {seas_order[1]} non-seasonal difference(s): I({seas_order[0]})', file=f)
            print(f' -> Q: {seas_order[2]} lags for the non-seasonal MA component: MA({seas_order[2]})', file=f)
            print(f' -> m: {seas_order[3]} as the seasonal period', file=f)

            lb_test = BoxJenkins.ljung_box(results, significance=sig, period=seas_order[3])

        if metric is not None:
            print(f'\nMetric: {metric.upper()} -> {results.metric}', file=f)

        print('\n------------------------------------------------------------------------------', file=f)
        print("                        2. ESTIMATION", file=f)
        print('------------------------------------------------------------------------------', file=f)

        estimation = results.summary().tables[1]
        print()
        print(estimation, file=f)

        print('\n------------------------------------------------------------------------------', file=f)
        print("                        3. DIAGNOSTICS", file=f)
        print('------------------------------------------------------------------------------', file=f)
        
        print(f'\n Ljung-Box test pvalue: {lb_test[0]}', file=f)
        if lb_test[1] == 0:
            print(f' -> Cannot reject the null hypothesis at the {sig*100}% level', file=f)
            print(f' -> The residuals of the model are uncorrelated.', file=f)
        if lb_test[1] == 1:
            print(f' -> Reject the null hypothesis at the {int(sig*100)}% signficance level', file=f)
            print(f' -> The residuals of the model are correlated.', file=f)
        print(f"\n Please check the accompanying '{model_name} diagnostic plots.png'", file=f)


    def __repr__(self):

        return f"Model: {self._model} \nTarget variable: {self.target} \nSeasonal period: {self.m}"
