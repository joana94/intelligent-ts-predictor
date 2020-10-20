import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

class Diffs(object):
    """
    Container for all methods related to time series differencing
    """

    @staticmethod
    def kpss_bool_logic(x, significance=0.05):
        """
        Auxiliary function to determine the need of differencing.

        Arguments.
        x: a time series 
        significance: default=0.05

        Returns:
        1: if the p-value is below or equal to the significance level meaning that the data is not trend stationary
        0: if the p-value is above the significance level meaning that the time series is trend stationary
        """
        result = kpss(x)
        p_value = result[1]
        
        if p_value <= significance:
            return 1
        else:
            return 0

    @staticmethod
    def num_diffs(x, max_d=2, seasonal=False, m=1):
        """
        Function that computes the number of first differences to make the data stationary.
        Uses Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test to determine the number of differences.
        Can be used as input for the "d" parameter of the ARIMA model.
        
        Arguments:
        x: Series or DataFrame containing the data
        threshold: the significance level to compare the KPSS value
        
        Returns:
        d: number of first differences needed to stationarize the data
        """
        if seasonal:
            s_diffs = Diffs.num_sdiffs(x, m=m)
            df = diff(x, k_diff=0, k_seasonal_diff=s_diffs, seasonal_periods=m) 
            result = Diffs.kpss_bool_logic(df)
            d = 0 
            while (result ==1 and d <= max_d):
                d +=1
                x = diff(df, k_diff=d)
                if Diffs.kpss_bool_logic(df) == 0:
                    #print(f'Number of first differences to make data stationary: {d}')
                    return d
                result = Diffs.kpss_bool_logic(df) 
        
        else:
            result = Diffs.kpss_bool_logic(x)
            d = 0 
            while (result ==1 and d <= max_d):
                d +=1
                x = diff(x, k_diff=d)
                if Diffs.kpss_bool_logic(x) == 0:
                    #print(f'Number of first differences to make data stationary: {d}')
                    return d
                result = Diffs.kpss_bool_logic(x) 
            
        #print(f'Number of first differences to make data stationary: {d}')
        return d

   
    @staticmethod
    def trend_strength_stationarity(x, threshold=0.64):
        """
        Computes the seasonal strength. Measure introduced by Wang, X., Smith, K. A., & Hyndman, R. J. (2006). In:
        Characteristic-based clustering for time series data. Data Mining and Knowledge Discovery, 13(3), 335–364.
        If seasonal strength is greater than 0.64, is not seasonally stationary and needs seasonal differencing.

        Arguments:
        x: time series for which the user wants to compute the trend strength
        threshold: the value above which the data is not seasonally stationary. Default is 0.64 as advised by Wang, Smith and Hyndman (2006)
        """
        result = STL(x).fit()
        trend_strength = max(0, min(1, 1- np.var(result.resid.values)/np.var(result.resid.values + result.trend.values)))
        #print(f"Trend strength index: {trend_strength}")
        
        if trend_strength > threshold:
            return 1
        else:
            return 0


    @staticmethod
    def seasonal_strength_stationarity(x, threshold=0.64):
        """
        Computes the seasonal strength. Measure introduced by Wang, X., Smith, K. A., & Hyndman, R. J. (2006). In:
        Characteristic-based clustering for time series data. Data Mining and Knowledge Discovery, 13(3), 335–364.
        If seasonal strength is greater than 0.64, is not seasonally stationary and needs seasonal differencing.

        Arguments:
        x: time series for which the user wants to compute the seasonal strength
        threshold: the value above which the data is not seasonally stationary. Default is 0.64 as advised by Wang, Smith and Hyndman (2006)
        """
        result = STL(x).fit()
        seas_strength = max(0, min(1, 1- np.var(result.resid.values)/np.var(result.resid.values + result.seasonal.values)))
        #print(f"Seasonal strength index: {seas_strength}")
        
        if seas_strength > threshold:
            return 1
        else:
            return 0

    @staticmethod
    def num_sdiffs(x, m, max_D = 2):
        """
        Computes the number of differences necessary to make the data seasonally stationary.

        Arguments: 
        x: pandas series containing a time series
        m: seasonal period
        max_D: maximum number of seasonal differences allowed (default=2 because usually no more than two differences are necessary)

        Returns:
        D: number of seasonal differences required to stationarize the data
        """
        D = 0
        s_strength = Diffs.seasonal_strength_stationarity(x)
        while(s_strength == 1 and D <= max_D):
            D += 1
            df = diff(x, k_diff=0, k_seasonal_diff=1, seasonal_periods=m) 
            if Diffs.seasonal_strength_stationarity(df) == 0:
                # print(f'Number of seasonal differences to make data stationary: {D}')
                return D
            s_strength = Diffs.seasonal_strength_stationarity(x)
        # print(f'Number of seasonal differences to make data stationary: {D}')
        return D