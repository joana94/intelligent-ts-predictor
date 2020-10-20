import numpy as np

class PredictionMetrics(object):
    """
    Container class for the Prediction Metrics most commonly used.
    > MSE, RMSE, MAE, MdAE
    """
    
    @staticmethod
    def mean_squared_error(actuals, predictions):
        errors = predictions - actuals
        return np.square(errors).mean()
    
    @staticmethod
    def root_mean_squared_error(actuals, predictions):
        errors = predictions - actuals
        return np.sqrt(np.square(errors).mean())
    @staticmethod
    def mean_absolute_error(actuals, predictions):
        errors = predictions - actuals
        return np.abs(errors).mean()

    @staticmethod
    def median_absolute_error(actuals, predictions):
        errors = predictions - actuals
        return np.median(np.abs(errors))