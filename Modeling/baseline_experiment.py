import numpy as np
import pandas as pd
import argparse
import optuna
import os
from functools import partial

from Models.PyTorch.Individual_Models.Single_Input.RNNs import LSTM, GRU, BiLSTM, BiGRU, HPT_LSTM, HPT_GRU, HPT_BiLSTM, HPT_BiGRU
from Models.PyTorch.Individual_Models.Single_Input.Others import TCN, HPT_TCN
from Models.PyTorch.Individual_Models.Multi_Input.RNNs import MI_LSTM, MI_GRU, MI_BiLSTM, MI_BiGRU, HPT_MI_LSTM, HPT_MI_GRU, HPT_MI_BiLSTM, HPT_MI_BiGRU
from Models.PyTorch.Individual_Models.Multi_Input.Others import MI_TCN, HPT_MI_TCN
from Models.PyTorch.Individual_Models.Multi_Modal.RNNs import MM_LSTM, MM_GRU, MM_BiLSTM, MM_BiGRU, HPT_MM_LSTM, HPT_MM_GRU, HPT_MM_BiLSTM, HPT_MM_BiGRU
from Models.PyTorch.Individual_Models.Multi_Modal.Others import MM_TCN, HPT_MM_TCN

from Models.PyTorch.Hybrid_Models.Single_Input.MSRCNN_LSTM import MSRCNN_LSTM, HPT_MSRCNN_LSTM
from Models.PyTorch.Hybrid_Models.Single_Input.MSRCNN_GRU import MSRCNN_GRU, HPT_MSRCNN_GRU
from Models.PyTorch.Hybrid_Models.Multi_Input.MSRCNN_LSTM import MI_MSRCNN_LSTM, HPT_MI_MSRCNN_LSTM
from Models.PyTorch.Hybrid_Models.Multi_Input.MSRCNN_GRU import MI_MSRCNN_GRU, HPT_MI_MSRCNN_GRU
from Models.PyTorch.Hybrid_Models.Multi_Modal.MSRCNN_LSTM import MM_MSRCNN_LSTM, HPT_MM_MSRCNN_LSTM
from Models.PyTorch.Hybrid_Models.Multi_Modal.MSRCNN_GRU import MM_MSRCNN_GRU, HPT_MM_MSRCNN_GRU

from Util.CryptoDataset import CryptoDataset
from Util import PT_Training_and_Prediction
from Util import Save_Results_Functions #plot_training


def optuna_objective_function(trial,
                              selected_crypto,
                              selected_forecast_interval,
                              selected_model,
                              selected_model_framework,
                              selected_experimental_setting,):
    
    if selected_experimental_setting != 'Original':
        vmd_params = {'K':trial.suggest_int('VMD_K', 2, 15),
                      'alpha':trial.suggest_int('VMD_alpha', 200, 8000, step=100),
                      'tau':trial.suggest_categorical('VMD_tau', [0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
                      'tol':trial.suggest_float("VMD_tol", 1e-8, 1e-2, log=True),
                      'DC':trial.suggest_categorical("VMD_DC", [False, True]),
                      'init':trial.suggest_categorical('VMD_init', ['uniform', 'zero', 'random'])}

    training_params = {'n_epochs':trial.suggest_int('TRAIN_n_epochs', 20, 200, step=2),
                       'batch_size':trial.suggest_int('TRAIN_batch_size', 16, 256, step=8),
                       'optimizer':trial.suggest_categorical("TRAIN_optimizer", ['Adam', 'AdamW', 'Adadelta', 'Adagrad', 'Adamax', 'NAdam', 'RMSprop', 'SGD']),
                       'learning_rate':trial.suggest_float("TRAIN_learning_rate", 1e-5, 1e-2, log=True),
                       'weight_decay':trial.suggest_categorical("TRAIN_weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
                       'momentum':trial.suggest_float("TRAIN_momentum", 0.8, 0.99)}
    
    if selected_model in ['LSTM', 'MI_LSTM', 'GRU', 'MI_GRU', 'BiLSTM', 'MI_BiLSTM', 'BiGRU', 'MI_BiGRU']:
        model_params = {'RNN_n_layers':trial.suggest_int('RNN_n_layers', 1, 3),
                        'RNN_dim_l1':trial.suggest_int('RNN_dim_l1', 16, 512, step=16),
                        'RNN_dim_l2':trial.suggest_int('RNN_dim_l2', 16, 512, step=16),
                        'RNN_dim_l3':trial.suggest_int('RNN_dim_l3', 16, 512, step=16),
                        'RNN_activation':trial.suggest_categorical("RNN_activation", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
                        
                        'Dense_n_layers':trial.suggest_int('RNN_n_layers', 1, 3),
                        'Dense_dim_l1':trial.suggest_int("Dense_dim_l1", 16, 256, step=16),
                        'Dense_dim_l2':trial.suggest_int("Dense_dim_l2", 16, 256, step=16),
                        'Dense_dim_l3':trial.suggest_int("Dense_dim_l3", 16, 256, step=16),
                        
                        'Dense_activation_l1':trial.suggest_categorical("Dense_activation_l1", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
                        'Dense_activation_l2':trial.suggest_categorical("Dense_activation_l2", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
                        'Dense_activation_l3':trial.suggest_categorical("Dense_activation_l3", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh'])}
            
    elif selected_model in ['TCN', 'MI_TCN']:
        model_params = {'Conv_n_layers':trial.suggest_int('Conv_n_layers', 1, 8),
                        'Conv_out_channels':trial.suggest_int("Conv_out_channels", 16, 128, step=8),
                        'Conv_kernel_size':trial.suggest_int('Conv_kernel_size', 1, 7),
                        'Conv_dilation':trial.suggest_int('Conv_dilation', 1, 7),

                        'Conv_activation':trial.suggest_categorical("Conv_activation", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
                        'Conv_dropout_rate':trial.suggest_float("Conv_dropout_rate", 0, 0.5),
                        
                        'Dense_n_layers':trial.suggest_int('RNN_n_layers', 1, 3),
                        'Dense_dim_l1':trial.suggest_int("Dense_dim_l1", 16, 256, step=16),
                        'Dense_dim_l2':trial.suggest_int("Dense_dim_l2", 16, 256, step=16),
                        'Dense_dim_l3':trial.suggest_int("Dense_dim_l3", 16, 256, step=16),
                        
                        'Dense_activation_l1':trial.suggest_categorical("Dense_activation_l1", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
                        'Dense_activation_l2':trial.suggest_categorical("Dense_activation_l2", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
                        'Dense_activation_l3':trial.suggest_categorical("Dense_activation_l3", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh'])}
        
    elif selected_model in ['MSRCNN_LSTM', 'MI_MSRCNN_LSTM','MSRCNN_GRU', 'MI_MSRCNN_GRU']:
        model_params = {'Input_Conv1d_activation':trial.suggest_categorical("Input_Conv1d_activation", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),

                        'MSR_Block_Conv1d_out_channels':trial.suggest_int("MSR_Block_Conv1d_out_channels", 8, 64, step=4),
                        'MSR_Block_Conv1d_activation':trial.suggest_categorical("MSR_Conv1d_activation", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
                        'MSR_Block_cross_channel_fusion':trial.suggest_categorical("MSR_Block_cross_channel_fusion", [True, False]),
                        'MSR_Block_Conv2d_out_channels':trial.suggest_int("MSR_Block_Conv2d_out_channels", 16, 128, step=4),
                        'MSR_Block_activation':trial.suggest_categorical("MSR_Block_activation", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),

                        'RNN_n_layers':trial.suggest_int('RNN_n_layers', 1, 3),
                        'RNN_dim_l1':trial.suggest_int('RNN_dim_l1', 16, 512, step=16),
                        'RNN_dim_l2':trial.suggest_int('RNN_dim_l2', 16, 512, step=16),
                        'RNN_dim_l3':trial.suggest_int('RNN_dim_l3', 16, 512, step=16),
                        'RNN_activation':trial.suggest_categorical("RNN_activation", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
                        
                        'Dense_n_layers':trial.suggest_int('Dense_n_layers', 1, 3),
                        'Dense_dim_l1':trial.suggest_int("Dense_dim_l1", 16, 256, step=16),
                        'Dense_dim_l2':trial.suggest_int("Dense_dim_l2", 16, 256, step=16),
                        'Dense_dim_l3':trial.suggest_int("Dense_dim_l3", 16, 256, step=16),
                        
                        'Dense_activation_l1':trial.suggest_categorical("Dense_activation_l1", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
                        'Dense_activation_l2':trial.suggest_categorical("Dense_activation_l2", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
                        'Dense_activation_l3':trial.suggest_categorical("Dense_activation_l3", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh'])}
        
    else:
        print("")

    if selected_experimental_setting != 'Original':
        training_results = baseline_experiment(selected_crypto=selected_crypto,
                                               selected_forecast_interval=selected_forecast_interval,
                                               selected_model=selected_model,
                                               selected_model_framework=selected_model_framework,
                                               selected_experimental_setting=selected_experimental_setting,
                                                
                                               vmd_params=vmd_params,
                                               training_params=training_params,
                                               model_params=model_params,

                                               n_reps=1,
                                               save_results=False,
                                               optimization_mode=True,
                                               verbosity=0)
        
    else:
        training_results = baseline_experiment(selected_crypto=selected_crypto,
                                               selected_forecast_interval=selected_forecast_interval,
                                               selected_model=selected_model,
                                               selected_model_framework=selected_model_framework,
                                               selected_experimental_setting=selected_experimental_setting,
                                               
                                               training_params=training_params,
                                               model_params=model_params,
                                               
                                               n_reps=1,
                                               save_results=False,
                                               optimization_mode=True,
                                               verbosity=0)

    print(training_results)

    trial.set_user_attr('mean_train_mse', training_results['mean_train_mse'])
    trial.set_user_attr('median_train_mse', training_results['median_train_mse'])
    trial.set_user_attr('mean_train_mae', training_results['mean_train_mae'])
    trial.set_user_attr('median_train_mae', training_results['median_train_mae'])

    trial.set_user_attr('mean_val_mse', training_results['mean_val_mse'])
    trial.set_user_attr('median_val_mse', training_results['median_val_mse'])
    trial.set_user_attr('mean_val_mae', training_results['mean_val_mae'])
    trial.set_user_attr('median_val_mae', training_results['median_val_mae'])

    return training_results['mean_val_mse']

def optuna_hyperparameter_tuning(selected_crypto,
                                 selected_forecast_interval,
                                 selected_model,
                                 selected_model_framework,
                                 selected_experimental_setting,):
    
    study = optuna.create_study(direction='minimize')

    study.optimize(partial(optuna_objective_function, 
                           selected_crypto=selected_crypto,
                           selected_forecast_interval=selected_forecast_interval,
                           selected_model=selected_model,
                           selected_model_framework=selected_model_framework,
                           selected_experimental_setting=selected_experimental_setting), 
                   n_trials=10)

    hyperparameter_tuning_results_df = study.trials_dataframe().drop(['number',
                                                                      'value',
                                                                      'datetime_start',
                                                                      'datetime_complete',
                                                                      'duration',
                                                                      'duration',
                                                                      'state'], 
                                                                     axis=1)

    hyperparameter_tuning_results_df['Mean_Train_MSE'] = [t.user_attrs["mean_train_mse"] for t in study.trials]
    hyperparameter_tuning_results_df['Median_Train_MSE'] = [t.user_attrs["median_train_mse"] for t in study.trials]
    hyperparameter_tuning_results_df['Mean_Train_MAE'] = [t.user_attrs["mean_train_mae"] for t in study.trials]
    hyperparameter_tuning_results_df['Median_Train_MAE'] = [t.user_attrs["median_train_mae"] for t in study.trials]

    hyperparameter_tuning_results_df['Mean_Val_MSE'] = [t.user_attrs["mean_val_mse"] for t in study.trials]
    hyperparameter_tuning_results_df['Median_Val_MSE'] = [t.user_attrs["median_val_mse"] for t in study.trials]
    hyperparameter_tuning_results_df['Mean_Val_MAE'] = [t.user_attrs["mean_val_mae"] for t in study.trials]
    hyperparameter_tuning_results_df['Median_Val_MAE'] = [t.user_attrs["median_val_mae"] for t in study.trials]

    hyperparameter_tuning_results_df = hyperparameter_tuning_results_df.sort_values(by='Mean_Val_MSE',
                                                                                    ignore_index=True)

    res_map = f'Modeling/Results/Baseline_Experiment/{selected_experimental_setting}_Setting/{selected_model}/{selected_model_framework}/{selected_crypto}/{selected_forecast_interval}/Hyperparameter_Tuning/'

    if not os.path.exists(res_map):
        os.makedirs(res_map)

    hyperparameter_tuning_results_df.to_csv(f'{res_map}Optuna_Results.csv')

    print('Best hyperparameters: ', study.best_params)
    print('Best performance: ', study.best_value)


def baseline_experiment(selected_crypto='BTC',
                        selected_forecast_interval='1_Day',
                        selected_model='GRU',
                        selected_experimental_setting='Original',

                        vmd_params={},
                        training_params={'n_epochs':300,
                                         'batch_size':32,
                                         'optimizer':'Adam',
                                         'learning_rate':1e-4,
                                         'weight_decay':1e-3,
                                         'momentum':0.9},
                        model_params={},

                        save_results=False,
                        n_reps=1,
                        optimization_mode=False,
                        verbosity=1):
    
    months = ['June_2021', 'July_2021', 'August_2021', 'September_2021', 'October_2021', 'November_2021', 'December_2021', 'January_2022', 'February_2022', 'March_2022', 'April_2022', 'May_2022']
    
    train_splits_offsets = {'June_2021':0, 
                            'July_2021':30, 
                            'August_2021':31, 
                            'September_2021':31, 
                            'October_2021':30, 
                            'November_2021':31, 
                            'December_2021':30, 
                            'January_2022':31, 
                            'February_2022':31, 
                            'March_2022':28, 
                            'April_2022':31, 
                            'May_2022':30}

    prediction_portion_sizes = {'June_2021':30, 
                                'July_2021':31, 
                                'August_2021':31, 
                                'September_2021':30, 
                                'October_2021':31, 
                                'November_2021':30, 
                                'December_2021':31, 
                                'January_2022':31, 
                                'February_2022':28, 
                                'March_2022':31, 
                                'April_2022':30, 
                                'May_2022':31}

    rep_results = {}

    best_train_mses = []
    best_train_maes = []
    best_val_mses = []
    best_val_maes = []

    for rep in range(1, n_reps+1):
        current_train_split_offset = 0
        rep_results[rep] = {}
        rep_results[rep]['nMSE'], rep_results[rep]['nRMSE'], rep_results[rep]['nMAE'], rep_results[rep]['nMAPE'] = {}, {}, {}, {}
        rep_results[rep]['MSE'], rep_results[rep]['RMSE'], rep_results[rep]['MAE'], rep_results[rep]['MAPE'], rep_results[rep]['R2'] = {}, {}, {}, {}, {}
        rep_results[rep]['Expected'], rep_results[rep]['Predicted'], rep_results[rep]['Expected_Normalized'], rep_results[rep]['Predicted_Normalized'] = {}, {}, {}, {}

        print(f"\n\t\t\t- Current Rep:{rep}")

        for m in months:
            current_train_split_offset += train_splits_offsets[m]
            current_prediction_portion_size = prediction_portion_sizes[m]

            print(f"\n\t\t\t\t- Current Rep:{rep} and Current Month: {m}, with a Split Offset of {current_train_split_offset} and a Test Prediction Portion Size of {current_prediction_portion_size}")

            if selected_experimental_setting == 'Original':
                current_dataset = CryptoDataset(crypto=selected_crypto,
                                                interval=selected_forecast_interval,

                                                input_c=True,
                                                apply_vmd=False,
                                                apply_differencing=True,

                                                split_offset=current_train_split_offset)
                
            elif selected_experimental_setting in ['VMD_IMFs', 'VMD_MI_IMFs', 'VMD_MM_IMFs']:
                current_dataset = CryptoDataset(crypto=selected_crypto,
                                                interval=selected_forecast_interval,
                                                
                                                vmd_n_IMFs=vmd_params['K'],
                                                vmd_alpha=vmd_params['alpha'],
                                                vmd_tau=vmd_params['tau'],
                                                vmd_tol=vmd_params['tol'],
                                                vmd_DC=vmd_params['DC'],
                                                vmd_init=vmd_params['init'],

                                                target_IMFs=True,
                                                split_offset=current_train_split_offset)
                
            elif selected_experimental_setting in ['VMD_C', 'VMD_MI_C']:
                current_dataset = CryptoDataset(crypto=selected_crypto,
                                                interval=selected_forecast_interval,

                                                vmd_n_IMFs=vmd_params['K'],
                                                vmd_alpha=vmd_params['alpha'],
                                                vmd_tau=vmd_params['tau'],
                                                vmd_tol=vmd_params['tol'],
                                                vmd_DC=vmd_params['DC'],
                                                vmd_init=vmd_params['init'],

                                                split_offset=current_train_split_offset)
            else:
                print("ERROR: Invalid Baseline Experimental Setting")
                return

            current_dataset.process_dataset()
            current_dataset.summary(tabs="\t\t\t\t\t")

            IMF_models = []
            IMF_models_training_metrics = []

            if selected_model == 'LSTM':
                # if optimization_mode:
                #     if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                #         model = HPT_MI_LSTM(dataset=current_dataset,
                                        
                #                             lstm_n_layers=model_params['RNN_n_layers'],
                #                             lstm_dims=[model_params['RNN_dim_l1'], model_params['RNN_dim_l2'], model_params['RNN_dim_l3']],
                #                             lstm_activation=model_params['RNN_activation'],

                #                             dense_n_layers=model_params['Dense_n_layers'],
                #                             dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                             dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                #     else:
                #         model = HPT_LSTM(dataset=current_dataset,
                                        
                #                          lstm_n_layers=model_params['RNN_n_layers'],
                #                          lstm_dims=[model_params['RNN_dim_l1'], model_params['RNN_dim_l2'], model_params['RNN_dim_l3']],
                #                          lstm_activation=model_params['RNN_activation'],

                #                          dense_n_layers=model_params['Dense_n_layers'],
                #                          dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                          dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                # else:
                if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                    model = MI_LSTM(dataset=current_dataset)
                elif selected_experimental_setting == 'VMD_MM_IMFs':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(MM_LSTM(dataset=current_dataset,
                                                  selected_IMF_idx=i))
                else:
                    model = LSTM(dataset=current_dataset)

            elif selected_model == 'GRU':
                # if optimization_mode:
                #     if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                #         model = HPT_MI_GRU(dataset=current_dataset,
                                        
                #                            gru_n_layers=model_params['RNN_n_layers'],
                #                            gru_dims=[model_params['RNN_dim_l1'], model_params['RNN_dim_l2'], model_params['RNN_dim_l3']],
                #                            gru_activation=model_params['RNN_activation'],

                #                            dense_n_layers=model_params['Dense_n_layers'],
                #                            dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                            dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                #     else:
                #         model = HPT_GRU(dataset=current_dataset,
                                        
                #                         gru_n_layers=model_params['RNN_n_layers'],
                #                         gru_dims=[model_params['RNN_dim_l1'], model_params['RNN_dim_l2'], model_params['RNN_dim_l3']],
                #                         gru_activation=model_params['RNN_activation'],

                #                         dense_n_layers=model_params['Dense_n_layers'],
                #                         dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                         dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                # else:
                if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                    model = MI_GRU(dataset=current_dataset)
                elif selected_experimental_setting == 'VMD_MM_IMFs':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(MM_GRU(dataset=current_dataset,
                                                 selected_IMF_idx=i))
                else:
                    model = GRU(dataset=current_dataset)

            elif selected_model == 'BiLSTM':
                # if optimization_mode:
                #     if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                #         model = HPT_MI_BiLSTM(dataset=current_dataset,
                                        
                #                               bilstm_n_layers=model_params['RNN_n_layers'],
                #                               bilstm_dims=[model_params['RNN_dim_l1'], model_params['RNN_dim_l2'], model_params['RNN_dim_l3']],
                #                               bilstm_activation=model_params['RNN_activation'],

                #                               dense_n_layers=model_params['Dense_n_layers'],
                #                               dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                               dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                #     else:
                #         model = HPT_BiLSTM(dataset=current_dataset,
                                        
                #                            bilstm_n_layers=model_params['RNN_n_layers'],
                #                            bilstm_dims=[model_params['RNN_dim_l1'], model_params['RNN_dim_l2'], model_params['RNN_dim_l3']],
                #                            bilstm_activation=model_params['RNN_activation'],

                #                            dense_n_layers=model_params['Dense_n_layers'],
                #                            dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                            dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                # else:
                if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                    model = MI_BiLSTM(dataset=current_dataset)
                elif selected_experimental_setting == 'VMD_MM_IMFs':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(MM_BiLSTM(dataset=current_dataset,
                                                    selected_IMF_idx=i))
                else:
                    model = BiLSTM(dataset=current_dataset)

            elif selected_model == 'BiGRU':
                # if optimization_mode:
                #     if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                #         model = HPT_MI_BiGRU(dataset=current_dataset,
                                        
                #                              bigru_n_layers=model_params['RNN_n_layers'],
                #                              bigru_dims=[model_params['RNN_dim_l1'], model_params['RNN_dim_l2'], model_params['RNN_dim_l3']],
                #                              bigru_activation=model_params['RNN_activation'],

                #                              dense_n_layers=model_params['Dense_n_layers'],
                #                              dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                              dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                #     else:
                #         model = HPT_BiGRU(dataset=current_dataset,
                                        
                #                           bigru_n_layers=model_params['RNN_n_layers'],
                #                           bigru_dims=[model_params['RNN_dim_l1'], model_params['RNN_dim_l2'], model_params['RNN_dim_l3']],
                #                           bigru_activation=model_params['RNN_activation'],

                #                           dense_n_layers=model_params['Dense_n_layers'],
                #                           dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                           dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                # else:
                if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                    model = MI_BiGRU(dataset=current_dataset)
                elif selected_experimental_setting == 'VMD_MM_IMFs':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(MM_BiGRU(dataset=current_dataset,
                                                   selected_IMF_idx=i))
                else:
                    model = BiGRU(dataset=current_dataset)

            elif selected_model == 'TCN':
                # if optimization_mode:
                #     if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                #         model = HPT_MI_TCN(current_dataset,
                                        
                #                            conv_n_layers=model_params['Conv_n_layers'],
                #                            conv_out_channels=model_params['Conv_out_channels'],
                #                            conv_kernel_size=model_params['Conv_kernel_size'],
                #                            conv_dilation=model_params['Conv_dilation'],
                #                            conv_activation=model_params['Conv_activation'],
                #                            conv_dropout_rate=model_params['Conv_dropout_rate'],
                                        
                #                            dense_n_layers=model_params['Dense_n_layers'],
                #                            dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                            dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                #     else:
                #         model = HPT_TCN(current_dataset,
                                        
                #                         conv_n_layers=model_params['Conv_n_layers'],
                #                         conv_out_channels=model_params['Conv_out_channels'],
                #                         conv_kernel_size=model_params['Conv_kernel_size'],
                #                         conv_dilation=model_params['Conv_dilation'],
                #                         conv_activation=model_params['Conv_activation'],
                #                         conv_dropout_rate=model_params['Conv_dropout_rate'],
                                        
                #                         dense_n_layers=model_params['Dense_n_layers'],
                #                         dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                         dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                    
                # else:
                if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                    model = MI_TCN(current_dataset)
                elif selected_experimental_setting == 'VMD_MM_IMFs':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(MM_TCN(dataset=current_dataset,
                                                 selected_IMF_idx=i))
                else:
                    model = TCN(current_dataset)

            elif selected_model == 'MSRCNN_LSTM':
                # if optimization_mode:
                #     if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                #         model = HPT_MI_MSRCNN_LSTM(dataset=current_dataset,
                                        
                #                                    input_conv_activation=model_params['Conv_input_activation'],
                                            
                #                                     msr_block_conv1d_out_channels=model_params['MSR_Block_Conv1d_out_channels'],
                #                                     msr_block_conv1d_activation=model_params['MSR_Block_Conv1d_activation'],
                #                                     msr_block_cross_channel_fusion=model_params['MSR_Block_cross_channel_fusion'],
                #                                     msr_block_conv2d_out_channels=model_params['MSR_Block_Conv2d_out_channels'],
                #                                     msr_block_activation=model_params['MSR_Block_activation'],
                                        
                #                                     lstm_n_layers=model_params['RNN_n_layers'],
                #                                     lstm_dims=[model_params['RNN_dim_l1'], model_params['RNN_dim_l2'], model_params['RNN_dim_l3']],
                #                                     lstm_activation=model_params['RNN_activation'],

                #                                     dense_n_layers=model_params['Dense_n_layers'],
                #                                     dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                                     dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                #     else:
                #         model = HPT_MSRCNN_LSTM(dataset=current_dataset,
                                        
                #                                 input_conv_activation=model_params['Conv_input_activation'],
                                            
                #                                 msr_block_conv1d_out_channels=model_params['MSR_Block_Conv1d_out_channels'],
                #                                 msr_block_conv1d_activation=model_params['MSR_Block_Conv1d_activation'],
                #                                 msr_block_cross_channel_fusion=model_params['MSR_Block_cross_channel_fusion'],
                #                                 msr_block_conv2d_out_channels=model_params['MSR_Block_Conv2d_out_channels'],
                #                                 msr_block_activation=model_params['MSR_Block_activation'],
                                        
                #                                 lstm_n_layers=model_params['RNN_n_layers'],
                #                                 lstm_dims=[model_params['RNN_dim_l1'], model_params['RNN_dim_l2'], model_params['RNN_dim_l3']],
                #                                 lstm_activation=model_params['RNN_activation'],

                #                                 dense_n_layers=model_params['Dense_n_layers'],
                #                                 dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                                 dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                # else:
                if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                    model = MI_MSRCNN_LSTM(dataset=current_dataset)
                elif selected_experimental_setting == 'VMD_MM_IMFs':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(MM_MSRCNN_LSTM(dataset=current_dataset,
                                                         selected_IMF_idx=i))
                else:
                    model = MSRCNN_LSTM(dataset=current_dataset)

            elif selected_model == 'MSRCNN_GRU':
                # if optimization_mode:
                #     if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                #         model = HPT_MI_MSRCNN_GRU(dataset=current_dataset,
                                            
                #                                   input_conv_activation=model_params['Input_Conv1d_activation'],
                                                
                #                                   msr_block_scales=model_params['MSR_Block_scales'],
                #                                   msr_block_conv1d_out_channels=model_params['MSR_Block_Conv1d_out_channels'],
                #                                   msr_block_conv1d_activation=model_params['MSR_Block_Conv1d_activation'],
                #                                   msr_block_cross_channel_fusion=model_params['MSR_Block_cross_channel_fusion'],
                #                                   msr_block_conv2d_out_channels=model_params['MSR_Block_Conv2d_out_channels'],
                #                                   msr_block_activation=model_params['MSR_Block_activation'],
                                            
                #                                   gru_n_layers=model_params['RNN_n_layers'],
                #                                   gru_dims=[model_params['RNN_dim_l1'], model_params['RNN_dim_l2'], model_params['RNN_dim_l3']],
                #                                   gru_activation=model_params['RNN_activation'],

                #                                   dense_n_layers=model_params['Dense_n_layers'],
                #                                   dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                                   dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                #     else:
                #             model = HPT_MSRCNN_GRU(dataset=current_dataset,
                                            
                #                                    input_conv_activation=model_params['Input_Conv1d_activation'],
                                                
                #                                    msr_block_scales=model_params['MSR_Block_scales'],
                #                                    msr_block_conv1d_out_channels=model_params['MSR_Block_Conv1d_out_channels'],
                #                                    msr_block_conv1d_activation=model_params['MSR_Block_Conv1d_activation'],
                #                                    msr_block_cross_channel_fusion=model_params['MSR_Block_cross_channel_fusion'],
                #                                    msr_block_conv2d_out_channels=model_params['MSR_Block_Conv2d_out_channels'],
                #                                    msr_block_activation=model_params['MSR_Block_activation'],
                                            
                #                                    gru_n_layers=model_params['RNN_n_layers'],
                #                                    gru_dims=[model_params['RNN_dim_l1'], model_params['RNN_dim_l2'], model_params['RNN_dim_l3']],
                #                                    gru_activation=model_params['RNN_activation'],

                #                                    dense_n_layers=model_params['Dense_n_layers'],
                #                                    dense_dims=[model_params['Dense_dim_l1'], model_params['Dense_dim_l2'], model_params['Dense_dim_l3']],
                #                                    dense_activations=[model_params['Dense_activation_l1'], model_params['Dense_activation_l2'], model_params['Dense_activation_l3']])
                # else:
                if selected_experimental_setting in ['VMD_MI_C', 'VMD_MI_IMFs']:
                    MI_MSRCNN_GRU(dataset=current_dataset)
                elif selected_experimental_setting == 'VMD_MM_IMFs':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(MM_MSRCNN_GRU(dataset=current_dataset,
                                                        selected_IMF_idx=i))
                else:
                    MSRCNN_GRU(dataset=current_dataset)
                    
            else:
                print("ERROR: No such Model available!")
                return

            if selected_experimental_setting == 'VMD_MM_IMFs':
                for i, model in enumerate(IMF_models):
                    print(f"\n\t\t\t\t\t- Training IMF {i+1} Model:")

                    model, training_metrics = PT_Training_and_Prediction.training(model=model,
                                                                                  dataset=current_dataset,
                                                                                    
                                                                                  n_epochs=training_params['n_epochs'],
                                                                                  batch_size=training_params['batch_size'],
                                                                                  optimizer=training_params['optimizer'],
                                                                                  learning_rate=training_params['learning_rate'],
                                                                                  weight_decay=training_params['weight_decay'],
                                                                                  momentum=training_params['momentum'],
                                                                                  
                                                                                  early_stopping_model_base_save_path=f"Modeling/Model_Checkpoints/{selected_experimental_setting}/{selected_model}/", 

                                                                                  targeted_IMF_idx=i,
                                                                                  multi_modal=True,
                                                                            
                                                                                  optimization_mode=optimization_mode,
                                                                                  verbose=verbosity)
                    IMF_models[i] = model
                    IMF_models_training_metrics.append(training_metrics)
            else:
                model, training_metrics = PT_Training_and_Prediction.training(model=model,
                                                                              dataset=current_dataset,
                                                                            
                                                                              n_epochs=training_params['n_epochs'],
                                                                              batch_size=training_params['batch_size'],
                                                                              optimizer=training_params['optimizer'],
                                                                              learning_rate=training_params['learning_rate'],
                                                                              weight_decay=training_params['weight_decay'],
                                                                              momentum=training_params['momentum'],

                                                                              early_stopping_model_base_save_path=f"Modeling/Model_Checkpoints/{selected_experimental_setting}/{selected_model}/", 
                                                                            
                                                                              optimization_mode=optimization_mode,
                                                                              verbose=verbosity)
            

            if not optimization_mode:
                price_data_X_test_portion_to_predict = current_dataset.price_data_X_test[:current_prediction_portion_size]

                expected_close_prices = current_dataset.y_test_close_price[current_dataset.horizon-1:(len(price_data_X_test_portion_to_predict)+current_dataset.horizon-1)]
                expected_normalized_close_prices = current_dataset.y_test_normalized_close_price[current_dataset.horizon-1:(len(price_data_X_test_portion_to_predict)+current_dataset.horizon-1)]

                if current_dataset.target_IMFs:
                    expected_IMFs = current_dataset.y_test_IMFs[current_dataset.horizon-1:(len(price_data_X_test_portion_to_predict)+current_dataset.horizon-1)]
                    expected_normalized_IMFs = current_dataset.y_test_normalized_IMFs[current_dataset.horizon-1:(len(price_data_X_test_portion_to_predict)+current_dataset.horizon-1)]

                if selected_experimental_setting == 'VMD_MM_IMFs':
                    all_IMFs_preds = []

                    for i, model in enumerate(IMF_models):
                        current_preds = PT_Training_and_Prediction.predict(model=model,
                                                                           price_data_X=price_data_X_test_portion_to_predict)

                        current_inverse_transformed_preds = current_dataset.inverse_transform_specific_IMF_predictions(preds=current_preds,
                                                                                                                       selected_IMF_idx=i)
                        
                        current_IMF_preds = current_inverse_transformed_preds
                        current_IMF_normalized_preds = current_preds

                        Save_Results_Functions.record_rep_results(rep_results=rep_results,
                                                                  rep=rep,
                                                                  expected_vals=expected_IMFs[:,i].reshape(-1,1),
                                                                  normalized_expected_vals=expected_normalized_IMFs[:,i].reshape(-1,1),
                                                                  predicted_vals=current_IMF_preds,
                                                                  normalized_predicted_vals=current_IMF_normalized_preds,
                                                                  var_name=f'Close_Price_IMF_{i+1}')
                        
                        all_IMFs_preds.append(current_IMF_preds)

                    IMFs_preds = np.concatenate(all_IMFs_preds, 
                                                axis=1)
                    close_price_preds = IMFs_preds.sum(axis=1).reshape(-1,1)
                    normalized_close_price_preds = current_dataset.normalize_price_predictions(preds=close_price_preds) 
                else:
                    preds = PT_Training_and_Prediction.predict(model=model,
                                                               price_data_X=price_data_X_test_portion_to_predict)

                    inverse_transformed_preds = current_dataset.inverse_transform_predictions(preds=preds,
                                                                                              inversion_differencing_offset_length=(len(current_dataset.y_test)))

                    if current_dataset.target_IMFs:
                        IMFs_preds = inverse_transformed_preds
                        normalized_IMFs_preds = preds
                        
                        close_price_preds = IMFs_preds.sum(axis=1).reshape(-1,1)
                        normalized_close_price_preds = current_dataset.normalize_price_predictions(preds=close_price_preds) 

                        for i, _ in enumerate(current_dataset.target_columns):
                            Save_Results_Functions.record_rep_results(rep_results=rep_results,
                                                                    rep=rep,
                                                                    expected_vals=expected_IMFs[:,i].reshape(-1,1),
                                                                    normalized_expected_vals=expected_normalized_IMFs[:,i].reshape(-1,1),
                                                                    predicted_vals=IMFs_preds[:,i].reshape(-1,1),
                                                                    normalized_predicted_vals=normalized_IMFs_preds[:,i].reshape(-1,1),
                                                                    var_name=f'Close_Price_IMF_{i+1}')
                    else:
                        close_price_preds = inverse_transformed_preds
                        normalized_close_price_preds = current_dataset.normalize_price_predictions(preds=close_price_preds)

                rep_results = Save_Results_Functions.record_rep_results(rep_results=rep_results,
                                                                        rep=rep,
                                                                        expected_vals=expected_close_prices,
                                                                        normalized_expected_vals=expected_normalized_close_prices,
                                                                        predicted_vals=close_price_preds,
                                                                        normalized_predicted_vals=normalized_close_price_preds,
                                                                        var_name='Close_Price')

                # best_train_mses.append(min(training_metrics['train_mse']))
                # best_train_maes.append(min(training_metrics['train_mae']))
                # best_val_mses.append(min(training_metrics['val_mse']))
                # best_val_maes.append(min(training_metrics['val_mae']))
                # best_train_mses.append(training_metrics['train_mse'][-1])
                # best_train_maes.append(training_metrics['train_mae'][-1])
                # best_val_mses.append(training_metrics['val_mse'][-1])
                # best_val_maes.append(training_metrics['val_mae'][-1])

    if save_results:
        base_res_map = f'Modeling/Results/Baseline_Experiment/{selected_experimental_setting}_Setting/{selected_model}/{selected_crypto}/{selected_forecast_interval}/'

        Save_Results_Functions.save_metrics(base_res_map=base_res_map,
                                            rep_results=rep_results,)

        Save_Results_Functions.save_outputs(base_res_map=base_res_map,
                                            rep_results=rep_results,)

        Save_Results_Functions.save_average_metrics(base_res_map=base_res_map,
                                                    rep_results=rep_results)    

        Save_Results_Functions.save_average_outputs(base_res_map=base_res_map,
                                                    rep_results=rep_results)

    # if optimization_mode:
    #     return {'mean_train_mse':np.mean(best_train_mses),
    #             'median_train_mse':np.median(best_train_mses),
    #             'mean_train_mae':np.mean(best_train_maes),
    #             'median_train_mae':np.median(best_train_maes),

    #             'mean_val_mse':np.mean(best_val_mses),
    #             'median_val_mse':np.median(best_val_mses),
    #             'mean_val_mae':np.mean(best_val_maes),
    #             'median_val_mae':np.median(best_val_maes),}     
                

def main():
    parser = argparse.ArgumentParser(description="Baseline Experiment")

    parser.add_argument('--crypto', 
                        type=str, 
                        default='BTC', 
                        choices=['BTC', 'ETH', 'LTC', 'XMR', 'XRP'], 
                        help="The Cryptocurrency Forecasted in the Baseline Experiment")
    parser.add_argument('--interval', 
                        type=str, 
                        default='1_Day', 
                        choices=['1_Hour', '2_Hour', '4_Hour', '6_Hour', '8_Hour', '12_Hour', '1_Day'], 
                        help="The Interval at which the Cryptocurrency is Forecasted in the Baseline Experiment")
    parser.add_argument('--model', 
                        type=str, 
                        default='GRU', 
                        choices=['LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'TCN', 'MSRCNN_LSTM', 'MSRCNN_GRU'], 
                        help="The Model utilized in executing the Baseline Experiment")
    parser.add_argument('--setting', 
                        type=str, 
                        default='Proposed_C', 
                        choices=['Original', 'VMD_C', 'VMD_MI_C', 'VMD_IMFs', 'VMD_MI_IMFs', 'VMD_MM_IMFs'], 
                        help="The Baseline Experimental Setting")
    parser.add_argument('--hyperparameter_tuning', 
                        action='store_true')

    args = parser.parse_args()

    if args.hyperparameter_tuning:
        print(f"\nStarting Hyperparameter Search of Baseline Experiment:")
    else:
        print(f"\nStarting Baseline Experiment:")

    print(f"\t- Crypto to be Forecasted: {args.crypto}")
    print(f"\t- Forecasting Interval: {args.interval}")
    print(f"\t- Model to be Utilized: {args.model}")
    print(f"\t- Experimental Setting: {args.setting}\n")

    if args.crypto == 'BTC':
        optimal_vmd_params = {'K':6,
                              'alpha':250,
                              'tau':.15,
                              'tol':1e-5,
                              'DC':False,
                              'init':'uniform'}
    elif args.crypto == 'ETH':
        optimal_vmd_params = {'K':14,
                              'alpha':1250,
                              'tau':.25,
                              'tol':1e-5,
                              'DC':False,
                              'init':'uniform'}
    elif args.crypto == 'LTC':
        optimal_vmd_params = {'K':6,
                              'alpha':250,
                              'tau':.3,
                              'tol':1e-5,
                              'DC':False,
                              'init':'uniform'}
    elif args.crypto == 'XMR':
        optimal_vmd_params = {'K':6,
                              'alpha':250,
                              'tau':.3,
                              'tol':1e-5,
                              'DC':True,
                              'init':'random'}
    else:
        optimal_vmd_params = {'K':10,
                              'alpha':1000,
                              'tau':.05,
                              'tol':1e-5,
                              'DC':True,
                              'init':'uniform'}

    if args.hyperparameter_tuning:
        optuna_hyperparameter_tuning(selected_crypto=args.crypto,
                                     selected_forecast_interval=args.interval,
                                     selected_model=args.model,
                                     selected_experimental_setting=args.setting,)
    else:
        baseline_experiment(selected_experimental_setting=args.setting,
                            selected_model=args.model,
                            vmd_params=optimal_vmd_params,
                            save_results=True,)


if __name__ == "__main__":
    main()
