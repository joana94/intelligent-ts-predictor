import config
import os
from boxjenkins_class import BoxJenkins
from nnforecast_class import SearchBestArchitecture, NNForecast, NNPlots
from utils_classes import FoldersUtils, DataUtils

def main():
    sub_dirs = FoldersUtils.unpickle_file('sub_dirs_list')
    data_folder = sub_dirs.get('Data')
    models_folder = sub_dirs.get('Models')

    model_folder = os.path.join(models_folder, config.MODEL)
    os.makedirs(os.path.join(model_folder, 'Forecasts'), exist_ok=True)

    forecasts_folder = os.path.join(model_folder, 'Forecasts')

    train = DataUtils.get_data_file(data_folder, 'train_set.csv')
    test = DataUtils.get_data_file(data_folder, 'test_set.csv')

    original_data = train.append(test)



    if config.MODEL == 'ARIMA' or config.MODEL == 'SARIMA':

        # It is required to load the found best hyperparameters in order to fit them to the entire dataset
        best_params_dict = BoxJenkins.load_best_params(model_name=config.MODEL)


        model = BoxJenkins(model=config.MODEL, data=original_data,
                        target=original_data.columns[0], m=best_params_dict["m"])

        if model._model == 'ARIMA':
            fitted_model = model.fit_to_entire_dataset(
                p=best_params_dict['p'], d=best_params_dict['d'], q=best_params_dict['q'], m=1, folder=forecasts_folder)
        elif model._model == 'SARIMA':
            fitted_model = model.fit_to_entire_dataset(
                p=best_params_dict['p'], d=best_params_dict['d'], q=best_params_dict['q'], seasonal=True, m=best_params_dict['m'],
                P=best_params_dict['P'], D=best_params_dict['D'], Q=best_params_dict['Q'], folder=forecasts_folder)

        # Computing the forecasts from the best model
        fc_df = BoxJenkins.forecast(model= fitted_model, n_periods=config.N_PERIODS, confidence=0.95)

        # Save forecasts csv to model folder
        BoxJenkins.save_forecast_files(
            fc_df, folder=forecasts_folder, model_name=config.MODEL)

        # Save forecasts plot to model folder
        BoxJenkins.save_forecast_plot(
            fc_df, folder=forecasts_folder, model_name=config.MODEL, original_data=original_data)
        print(f'> Forecast files and plot saved to {forecasts_folder}')


    if config.MODEL == 'TradRNN' or config.MODEL == 'GRU' or config.MODEL == 'LSTM':

        best_params_dict = SearchBestArchitecture.load_best_params(
            model_name=config.MODEL)

        # The model has to be trained on the entire dataset in order to produce actual forecasts into future
        # This time it will be trained with the found best hyperparameters

        # 1st normalize the data
        scaler_object = DataUtils.normalize_data(
            train_data=original_data, test_data=None)

        normalized_data = scaler_object['normalized_train']

        # 2nd step: Create windows of sequences and the respective labels
        print('\n> Creating windows of sequences and their respective labels')

        data_x, data_y = DataUtils.create_sequences_and_labels(
            data=normalized_data, seq_len=best_params_dict["seq_len"])

        # Training
        print('\n> Starting training with optimal architecture in entire dataset...')
        model, train_hist = NNForecast.train(model_name=config.MODEL, data_x=data_x, data_y=data_y,
                                            batch_size=best_params_dict["batch_size"], hidden_dim=best_params_dict['hidden_dim'],
                                            epochs=config.NUM_EPOCHS, learning_rate=best_params_dict[
                                                'learning_rate'],
                                            seq_len=config.SEQ_LEN, shuffle=best_params_dict["shuffle"], folder=forecasts_folder)

        # Save training loss history plot
        NNPlots.plot_losses(model_name=config.MODEL,
                            train_history=train_hist, folder_to_save=forecasts_folder)
        print(
            f'\n> The training has ended! Training loss history plot saved to {forecasts_folder}')

        # Produce the forecasts

        mean_preds, upper_bounds, lower_bounds = NNForecast.forecast(
            model=model, data_x=data_x, scaler_object=scaler_object['scaler'], n_periods=config.N_PERIODS, confidence=[0.85, 0.9, 0.95])

        # Save the forecasts to ModelName/Forecasts
        print(f'\n> Forecasts files and plot saved to {forecasts_folder}')
        fc_df = NNForecast.nn_forecast_to_df(mean_preds, upper_bounds, lower_bounds,
                                            model_name=config.MODEL, original_data=original_data, folder=forecasts_folder)

        # Save the plot of the forecasts to ModelName/Forecasts
        NNForecast.plot_nn_forecast(model_name=config.MODEL, forecasts=fc_df,
                                    original_data=original_data, folder=forecasts_folder)

if __name__ == '__main__':

    main()