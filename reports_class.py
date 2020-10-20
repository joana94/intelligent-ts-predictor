import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import os
from statsmodels.tsa.stattools import kpss
import warnings
warnings.filterwarnings('ignore')
from boxjenkins_class import BoxJenkins


class StatsReports(object):
    """
    Container for statistical report
    """

    @staticmethod
    def kpss_(df, significance = 0.05, report_name='KPSS Report', out_folder=None):
        
        """
        Function that outputs a detailed report for the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
        
        Arguments:
        df: the functions accepts a dataframe as input
        significance (optional): if no significance is specified, the default value used is 0.05
        
        Returns:
        
        """
        if out_folder is None:
            out_folder = os.getcwd()

        f = open(os.path.join(out_folder, f'{report_name}.txt'), 'w')

        print('Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test', file = f)
        print('Null hypothesis: The series is STATIONARY', file=f)
        print('---------------------------------------------', file=f)

        result = kpss(df, nlags='auto',)
        
        labels = ['KPSS test statistic', 'p-value', 'number of lags used']
        out = pd.Series(result[0:3], index=labels)
        
        for key, value in result[3].items():
            out[f'critical value ({key})'] = value
        
        print(out.to_string(), end='\n', file=f)
        print('---------------------------------------------', file=f)
        print('Results:', end='\n', file=f)
        

        if result[1] <= significance:
            print('> STRONG evidence against the null hypothesis', file=f)
            print(f'> Reject the null hypothesis at {significance} significance level', file=f)
            print('> The time series is NOT STATIONARY', file=f)
        else:
            print('> WEAK evidence against the null hypothesis', file=f)
            print(f'> Fail to reject the null hypothesis at {significance} significance level', file=f)
            print('> The time series is STATIONARY', file=f)
        
        f.close()

    @staticmethod
    def general_stats(dataset, report_name='TS overview', out_folder=None):
        """
        Generates a report with descriptive statistics of the dataset.
        """

        if out_folder is None:
            out_folder = os.getcwd()

        f = open(os.path.join(out_folder, f'{report_name}.txt'), 'w')
    
        print("--------------------------------------------", file=f)
        print("DATASET OVERVIEW", file=f)
        print("--------------------------------------------", file=f)
        print("Number of total observations: {:14.1f}".format(len(dataset)), file=f)
        print("Mean: {:38.1f}".format(dataset.values.mean()), file=f)
        print("Standard deviation: {:24.1f}".format(dataset.values.std()), file=f)
        print("Median: {:36.1f}".format(np.median(dataset.values)), file=f)
        print("Minimum value:{:30.1f}".format(dataset.values.min()), file=f)
        print("Maximum value:{:30.1f}".format(dataset.values.max()), file=f)
        
        try:
            result = STL(dataset).fit()
            trend_strength = max(0, min(1, 1- np.var(result.resid.values)/np.var(result.resid.values + result.trend.values)))
            seas_strength = max(0, min(1, 1- np.var(result.resid.values)/np.var(result.resid.values + result.seasonal.values)))
            print('\nTrend strength index: {:22.4f}'.format(trend_strength), end=' -> ', file=f)
            if trend_strength > 0.64:
                print('Suggests the presence of a non-stationary trend pattern', file=f)
            else:
                print('Suggests a non-existent or weak trend pattern', file=f)
            print('Seasonal strength index: {:19.4f}'.format(seas_strength), end=' -> ', file=f)
            if seas_strength > 0.64:
                print('Suggests the presence of a non-stationary seasonal pattern', file=f)
            else:
                print('Suggests a non-existent or weak seasonal pattern', file=f)
        except:
            pass

   

