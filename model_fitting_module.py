"""
----------------------------------------------------------------------------------------------------------------------
This module is responsible for fitting, estimating and choosing the best set of hyperparameters for the selected model
----------------------------------------------------------------------------------------------------------------------
"""
import os
import config
from collections import OrderedDict
from boxjenkins_class import BoxJenkins
from utils_classes import FoldersUtils, DataUtils
from nnforecast_class import SearchBestArchitecture, NNForecast, NNPlots


def main():
    sub_dirs = FoldersUtils.unpickle_file('sub_dirs_list')
    data_folder = sub_dirs.get('Data')
    models_folder = sub_dirs.get('Models')

    # Create model folder
    os.makedirs(os.path.join(models_folder, config.MODEL), exist_ok=True)
    model_folder = os.path.join(models_folder, config.MODEL)

    # Read train and test sets
    train = DataUtils.get_data_file(data_folder, 'train_set.csv')
    test = DataUtils.get_data_file(data_folder, 'test_set.csv')
    # Set the target variable to predict
    target = train.columns[0]

    if config.MODEL == "ARIMA" or config.MODEL == "SARIMA":

        # Initialize model
        model = BoxJenkins(model=config.MODEL, data=train,
                        target=target, m=config.M)

        # Search the best model and estimate its parameters
        print('> Searching and fitting the best model...')
        print('> This process may take a while.')
        fitted_model = model.fit(metric=config.metric, max_p=config.max_p, d=config.d,
                                max_q=config.max_q, seasonal=config.SEASONAL, m=config.M,
                                max_P=config.max_P, D=config.D, max_Q=config.max_Q, folder=model_folder)

        print('\n> Best model found and fitted to the data!')
        print(f'> Saving reports, graphics and files to {model_folder}')

        test_predictions, eval_metrics = BoxJenkins.evaluate(
            test_data=test, evaluation_metrics=['mse', 'rmse', 'mae', 'mdae'])

        BoxJenkins.save_eval_metrics(
            model_name=config.MODEL, metrics_df=eval_metrics, folder=model_folder)

        BoxJenkins.save_test_predictions(
            model_name=config.MODEL, predictions=test_predictions, folder=model_folder, train_data=train)

    if config.MODEL == 'TradRNN' or config.MODEL == 'GRU' or config.MODEL == 'LSTM':

        # Pre-process time series

        # 1st step: Normalize data through Min-Max scaling
        print('> Normalizing train and test data...')
        scaler_object = DataUtils.normalize_data(train_data=train, test_data=test)

        normalized_train = scaler_object['normalized_train']
        normalized_test = scaler_object['normalized_test']

        # 2nd step: Create windows of sequences and the respective labels
        print('\n> Creating windows of sequences and their respective labels')

        train_x, train_y = DataUtils.create_sequences_and_labels(
            data=normalized_train, seq_len=config.SEQ_LEN)

        test_x, test_y = DataUtils.create_sequences_and_labels(
            data=normalized_test, seq_len=config.SEQ_LEN)

        # Grab the hyperparameters defined in the config.py file
        hyperparameters = OrderedDict(hidden_dim=config.HYPERPARAMETERS['HIDDEN_DIM'],
                                    learning_rate=config.HYPERPARAMETERS['LEARNING_RATE'],
                                    batch_size=config.HYPERPARAMETERS['BATCH_SIZE'],
                                    shuffle=config.HYPERPARAMETERS['SHUFFLE'])

        # Search for the best architecture
        print('\n> Searching for the best architecture based on the hyperparameters...')
        print('> This process may take a long time.')
        best_model, train_hist, valid_hist, best_model_hp = SearchBestArchitecture.find_architecture(
            model_name=config.MODEL, hyperparameters=hyperparameters, search_method=config.SEARCH_METHOD, data_x=train_x,
            data_y=train_y, epochs=config.NUM_EPOCHS, seq_len=config.SEQ_LEN, max_iters=config.MAX_ITERS,
            validation_split=config.VALID_SPLIT, folder=model_folder)
        print('\n> BEST ARCHITECTURE FOUND!')

        NNPlots.plot_losses(model_name=config.MODEL, train_history=train_hist,
                            valid_history=valid_hist, folder_to_save=model_folder)

        print(
            f'> Saving search details and train and validation losses graphics to {model_folder}')

        SearchBestArchitecture.save_best_params(best_model_hp, model_name=config.MODEL)
        # Create predictions for the test set with the found best architecture

        test_predictions, eval_metrics = NNForecast.test_predictions(
            model=best_model, train_x=train_x, scaler_object=scaler_object['scaler'],
            test_data=test, evaluation_metrics=['mse', 'rmse', 'mae', 'mdae'])

        NNForecast.save_eval_metrics(
            model_name=config.MODEL, metrics_df=eval_metrics, folder=model_folder)
        NNForecast.save_test_predictions(
            model_name=config.MODEL, predictions=test_predictions, folder=model_folder, train_data=train)


if __name__ == '__main__':
    
    main()
