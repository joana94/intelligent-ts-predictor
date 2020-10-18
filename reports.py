import pandas as pd
import os
from statsmodels.tsa.stattools import kpss
from func_utils import unpickle_file, json_dir
import warnings
warnings.filterwarnings('ignore')
from load_data import DATASET

class StatsReports(object):
    """
    Container for statistical report
    """
    sub_dirs = unpickle_file('sub_dirs_list')
    reports_dir = sub_dirs.get('Reports')

    @staticmethod
    def kpss_(df, significance = 0.05):
        
        """
        Function that outputs a detailed report for the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
        
        Arguments:
        df: the functions accepts a dataframe as input
        significance (optional): if no significance is specified, the default value used is 0.05
        
        Returns:
        
        """
        with open(os.path.join(StatsReports.reports_dir,'kpss_report.txt'), 'w') as f:
        
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

    
# StatsReports.kpss_(DATASET)

