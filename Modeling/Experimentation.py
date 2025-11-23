import numpy as np
import argparse
import optuna
import os
import time
from functools import partial

from Models.PyTorch.Individual_Models.Single_Input.RNNs import SI_LSTM, SI_GRU, SI_BiLSTM, SI_BiGRU#, HPT_LSTM, HPT_GRU, HPT_BiLSTM, HPT_BiGRU
from Models.PyTorch.Individual_Models.Single_Input.Others import SI_TCN, SIMH_TCN #, HPT_TCN

from Models.PyTorch.Individual_Models.Multi_Input.RNNs import MIC_LSTM, MIC_GRU, MIC_BiLSTM, MIC_BiGRU#, HPT_MI_LSTM, HPT_MI_GRU, HPT_MI_BiLSTM, HPT_MI_BiGRU
from Models.PyTorch.Individual_Models.Multi_Input.Others import MIC_TCN, MIS_TCN #, HPT_MI_TCN

from Models.PyTorch.Individual_Models.Multi_Modal.RNNs import MM_LSTM, MM_GRU, MM_BiLSTM, MM_BiGRU#, HPT_MM_LSTM, HPT_MM_GRU, HPT_MM_BiLSTM, HPT_MM_BiGRU
from Models.PyTorch.Individual_Models.Multi_Modal.Others import MM_TCN #, HPT_MM_TCN

from Models.PyTorch.Hybrid_Models.Single_Input.SI_MSRCNN_LSTM import SI_MSRCNN_LSTM #, HPT_MSRCNN_LSTM
from Models.PyTorch.Hybrid_Models.Single_Input.SI_MSRCNN_GRU import SI_MSRCNN_GRU #, HPT_MSRCNN_GRU
from Models.PyTorch.Hybrid_Models.Single_Input.SI_ATCN import SI_ATCN

from Models.PyTorch.Hybrid_Models.Multi_Input.MI_MSRCNN_LSTM import MIC_MSRCNN_LSTM#, HPT_MI_MSRCNN_LSTM
from Models.PyTorch.Hybrid_Models.Multi_Input.MI_MSRCNN_GRU import MIC_MSRCNN_GRU#, HPT_MI_MSRCNN_GRU
from Models.PyTorch.Hybrid_Models.Multi_Input.MI_ATCN import MIC_ATCN

from Models.PyTorch.Hybrid_Models.Multi_Modal.MM_MSRCNN_LSTM import MM_MSRCNN_LSTM#, HPT_MM_MSRCNN_LSTM
from Models.PyTorch.Hybrid_Models.Multi_Modal.MM_MSRCNN_GRU import MM_MSRCNN_GRU#, HPT_MM_MSRCNN_GRU
from Models.PyTorch.Hybrid_Models.Multi_Modal.MM_ATCN import MM_ATCN

from Util.CryptoDataset import CryptoDataset
from Util import PT_Training_and_Prediction
from Modeling.Results_Util import Save_Results_Functions #plot_training


def optuna_objective_function(
        trial,
        
        selected_crypto,
        selected_forecast_interval,
        selected_model,
        
        selected_input_setting,
        selected_model_setting,
        selected_target_setting,
        selected_experiment_name,
):
    
    if selected_input_setting in ['VMD', 'VMD_n_Res', 'VMD_n_CEEMDAN']:
        vmd_params = {
            'K':trial.suggest_int('VMD_K', 2, 15),
            'alpha':trial.suggest_int('VMD_alpha', 250, 8000, step=250),
            'tau':trial.suggest_categorical('VMD_tau', [0.0, 0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
        }
    else:
        vmd_params = {}

    # training_params = {'n_epochs':trial.suggest_int('TRAIN_n_epochs', 20, 200, step=2),
    #                    'batch_size':trial.suggest_int('TRAIN_batch_size', 16, 256, step=8),
    #                    'optimizer':trial.suggest_categorical("TRAIN_optimizer", ['Adam', 'AdamW', 'Adadelta', 'Adagrad', 'Adamax', 'NAdam', 'RMSprop', 'SGD']),
    #                    'learning_rate':trial.suggest_float("TRAIN_learning_rate", 1e-5, 1e-2, log=True),
    #                    'weight_decay':trial.suggest_categorical("TRAIN_weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
    #                    'momentum':trial.suggest_float("TRAIN_momentum", 0.8, 0.99)}
    
    # if selected_model in ['LSTM', 'MI_LSTM', 'GRU', 'MI_GRU', 'BiLSTM', 'MI_BiLSTM', 'BiGRU', 'MI_BiGRU']:
    #     model_params = {'RNN_n_layers':trial.suggest_int('RNN_n_layers', 1, 3),
    #                     'RNN_dim_l1':trial.suggest_int('RNN_dim_l1', 16, 512, step=16),
    #                     'RNN_dim_l2':trial.suggest_int('RNN_dim_l2', 16, 512, step=16),
    #                     'RNN_dim_l3':trial.suggest_int('RNN_dim_l3', 16, 512, step=16),
    #                     'RNN_activation':trial.suggest_categorical("RNN_activation", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
                        
    #                     'Dense_n_layers':trial.suggest_int('RNN_n_layers', 1, 3),
    #                     'Dense_dim_l1':trial.suggest_int("Dense_dim_l1", 16, 256, step=16),
    #                     'Dense_dim_l2':trial.suggest_int("Dense_dim_l2", 16, 256, step=16),
    #                     'Dense_dim_l3':trial.suggest_int("Dense_dim_l3", 16, 256, step=16),
                        
    #                     'Dense_activation_l1':trial.suggest_categorical("Dense_activation_l1", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
    #                     'Dense_activation_l2':trial.suggest_categorical("Dense_activation_l2", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
    #                     'Dense_activation_l3':trial.suggest_categorical("Dense_activation_l3", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh'])}
            
    # elif selected_model in ['TCN', 'MI_TCN']:
    #     model_params = {'Conv_n_layers':trial.suggest_int('Conv_n_layers', 1, 8),
    #                     'Conv_out_channels':trial.suggest_int("Conv_out_channels", 16, 128, step=8),
    #                     'Conv_kernel_size':trial.suggest_int('Conv_kernel_size', 1, 7),
    #                     'Conv_dilation':trial.suggest_int('Conv_dilation', 1, 7),

    #                     'Conv_activation':trial.suggest_categorical("Conv_activation", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
    #                     'Conv_dropout_rate':trial.suggest_float("Conv_dropout_rate", 0, 0.5),
                        
    #                     'Dense_n_layers':trial.suggest_int('RNN_n_layers', 1, 3),
    #                     'Dense_dim_l1':trial.suggest_int("Dense_dim_l1", 16, 256, step=16),
    #                     'Dense_dim_l2':trial.suggest_int("Dense_dim_l2", 16, 256, step=16),
    #                     'Dense_dim_l3':trial.suggest_int("Dense_dim_l3", 16, 256, step=16),
                        
    #                     'Dense_activation_l1':trial.suggest_categorical("Dense_activation_l1", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
    #                     'Dense_activation_l2':trial.suggest_categorical("Dense_activation_l2", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
    #                     'Dense_activation_l3':trial.suggest_categorical("Dense_activation_l3", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh'])}
        
    # elif selected_model in ['MSRCNN_LSTM', 'MI_MSRCNN_LSTM','MSRCNN_GRU', 'MI_MSRCNN_GRU']:
    #     model_params = {'Input_Conv1d_activation':trial.suggest_categorical("Input_Conv1d_activation", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),

    #                     'MSR_Block_Conv1d_out_channels':trial.suggest_int("MSR_Block_Conv1d_out_channels", 8, 64, step=4),
    #                     'MSR_Block_Conv1d_activation':trial.suggest_categorical("MSR_Conv1d_activation", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
    #                     'MSR_Block_cross_channel_fusion':trial.suggest_categorical("MSR_Block_cross_channel_fusion", [True, False]),
    #                     'MSR_Block_Conv2d_out_channels':trial.suggest_int("MSR_Block_Conv2d_out_channels", 16, 128, step=4),
    #                     'MSR_Block_activation':trial.suggest_categorical("MSR_Block_activation", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),

    #                     'RNN_n_layers':trial.suggest_int('RNN_n_layers', 1, 3),
    #                     'RNN_dim_l1':trial.suggest_int('RNN_dim_l1', 16, 512, step=16),
    #                     'RNN_dim_l2':trial.suggest_int('RNN_dim_l2', 16, 512, step=16),
    #                     'RNN_dim_l3':trial.suggest_int('RNN_dim_l3', 16, 512, step=16),
    #                     'RNN_activation':trial.suggest_categorical("RNN_activation", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
                        
    #                     'Dense_n_layers':trial.suggest_int('Dense_n_layers', 1, 3),
    #                     'Dense_dim_l1':trial.suggest_int("Dense_dim_l1", 16, 256, step=16),
    #                     'Dense_dim_l2':trial.suggest_int("Dense_dim_l2", 16, 256, step=16),
    #                     'Dense_dim_l3':trial.suggest_int("Dense_dim_l3", 16, 256, step=16),
                        
    #                     'Dense_activation_l1':trial.suggest_categorical("Dense_activation_l1", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
    #                     'Dense_activation_l2':trial.suggest_categorical("Dense_activation_l2", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh']),
    #                     'Dense_activation_l3':trial.suggest_categorical("Dense_activation_l3", ['elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'selu', 'sigmoid', 'tanh'])}
        
    # else:
    #     print("")

    validation_results = run_experiment(
        selected_crypto=selected_crypto,
        selected_forecast_interval=selected_forecast_interval,
        selected_model=selected_model,
        
        selected_input_setting=selected_input_setting,
        selected_model_setting=selected_model_setting,
        selected_target_setting=selected_target_setting,
        selected_experiment_name=selected_experiment_name,
        
        VMD_params=vmd_params,
        #    training_params=training_params,
        #    model_params=model_params,
        
        n_reps=5,
        save_results=False,
        optimization_mode=True,
        verbosity=0,
    )

    trial.set_user_attr('mean_val_nmse', validation_results['mean_val_nmse'])
    trial.set_user_attr('mean_val_nrmse', validation_results['mean_val_nrmse'])
    trial.set_user_attr('mean_val_nmae', validation_results['mean_val_nmae'])
    trial.set_user_attr('mean_val_nmape', validation_results['mean_val_nmape'])

    trial.set_user_attr('mean_val_mse', validation_results['mean_val_mse'])
    trial.set_user_attr('mean_val_rmse', validation_results['mean_val_rmse'])
    trial.set_user_attr('mean_val_mae', validation_results['mean_val_mae'])
    trial.set_user_attr('mean_val_mape', validation_results['mean_val_mape'])
    trial.set_user_attr('mean_val_r2', validation_results['mean_val_r2'])

    return validation_results['mean_val_rmse']


def optuna_hyperparameter_tuning(
        selected_crypto,
        selected_forecast_interval,
        selected_model,
        
        selected_input_setting,
        selected_model_setting,
        selected_target_setting,
        selected_experiment_name,
):
    
    study = optuna.create_study(direction='minimize')

    study.optimize(
        partial(
            optuna_objective_function, 

            selected_crypto=selected_crypto,
            selected_forecast_interval=selected_forecast_interval,
            selected_model=selected_model,

            selected_input_setting=selected_input_setting,
            selected_model_setting=selected_model_setting,
            selected_target_setting=selected_target_setting,
            selected_experiment_name=selected_experiment_name,
        ), 
        n_trials=30,
    )

    hyperparameter_tuning_results_df = study.trials_dataframe().drop(
        columns=[
            'number',
            'value',
            'datetime_start',
            'datetime_complete',
            'duration',
            'state'
        ], 
    )

    hyperparameter_tuning_results_df['Mean_Val_nMSE'] = [t.user_attrs["mean_val_nmse"] for t in study.trials]
    hyperparameter_tuning_results_df['Mean_Val_nRMSE'] = [t.user_attrs["mean_val_nrmse"] for t in study.trials]
    hyperparameter_tuning_results_df['Mean_Val_nMAE'] = [t.user_attrs["mean_val_nmae"] for t in study.trials]
    hyperparameter_tuning_results_df['Mean_Val_nMAPE'] = [t.user_attrs["mean_val_nmape"] for t in study.trials]

    hyperparameter_tuning_results_df['Mean_Val_MSE'] = [t.user_attrs["mean_val_mse"] for t in study.trials]
    hyperparameter_tuning_results_df['Mean_Val_RMSE'] = [t.user_attrs["mean_val_rmse"] for t in study.trials]
    hyperparameter_tuning_results_df['Mean_Val_MAE'] = [t.user_attrs["mean_val_mae"] for t in study.trials]
    hyperparameter_tuning_results_df['Mean_Val_MAPE'] = [t.user_attrs["mean_val_mape"] for t in study.trials]
    hyperparameter_tuning_results_df['Mean_Val_R2'] = [t.user_attrs["mean_val_r2"] for t in study.trials]

    hyperparameter_tuning_results_df = hyperparameter_tuning_results_df.sort_values(by='Mean_Val_RMSE', ignore_index=True)
    
    hyperparameter_tuning_results_df = hyperparameter_tuning_results_df.drop(
        columns=[
            'user_attrs_mean_val_mae',
            'user_attrs_mean_val_mape',
            'user_attrs_mean_val_mse',
            'user_attrs_mean_val_nmae',
            'user_attrs_mean_val_nmape',
            'user_attrs_mean_val_nmse',
            'user_attrs_mean_val_nrmse',
            'user_attrs_mean_val_r2',
            'user_attrs_mean_val_rmse',
        ], 
    )

    res_map = f'Modeling/Results/Baseline_Experiment/{selected_input_setting}_{selected_model_setting}_{selected_target_setting}/{selected_model}/{selected_crypto}/{selected_experiment_name}/{selected_forecast_interval}/Hyperparameter_Tuning/'

    if not os.path.exists(res_map):
        os.makedirs(res_map)

    hyperparameter_tuning_results_df.to_csv(f'{res_map}Optuna_Results.csv')

    print('Best hyperparameters: ', study.best_params)
    print('Best performance: ', study.best_value)


def run_experiment(
        selected_crypto='BTC',
        selected_price_timespan='Baseline',
        selected_price_source='Binance',
        selected_price_frequency='1_Day',

        selected_model='TCN',
        selected_input_setting='VMD',
        selected_model_setting='S',
        selected_target_setting='C',
        selected_experiment_name='M12_R3',
        
        VMD_params={},
        training_params={
            'n_epochs':300,
            'batch_size':64,
            'optimizer':'Adam',
            'learning_rate':1e-4,
            'weight_decay':1e-3,
            'momentum':0.9
        },
        model_params={},
        
        save_results=False,
        n_reps=3,
        optimization_mode=False,
        verbosity=1
):
    start_time_experiment = time.time()

    if optimization_mode:
        months = ['June_2021']
    else:
        months = ['June_2021', 'July_2021', 'August_2021', 'September_2021', 'October_2021', 'November_2021', 'December_2021', 'January_2022', 'February_2022', 'March_2022', 'April_2022', 'May_2022'][:4]
    train_splits_offsets = {
        'June_2021':0, 
        'July_2021':30,  
        'August_2021':61, # 31 (July)
        'September_2021':92, # 31 (August)
        'October_2021':122, # 30 (September)
        'November_2021':153, # 31 (October)
        'December_2021':183, # 30 (November)
        'January_2022':214, # 31 (December)
        'February_2022':245, # 31 (January)
        'March_2022':273, # 28 (February)
        'April_2022':304, # 31 (March)
        'May_2022':334, # 30 (April)
        'June_2022':365, # 31 (May)
        'July_2022':395, # 30 (June)
        'August_2022':426, # 31 (July)
        'September_2022':457, # 31 (August)
        'October_2022':487, # 30 (September)
        'November_2022':518, # 31 (October)
        'December_2022':548, # 30 (November)
        'January_2023':579, # 31 (December)
        'February_2023':610, # 31 (January)
        'March_2023':638, # 28 (February)
        'April_2023':669, # 31 (March)
        'May_2023':699, # 30 (April)
        'June_2023':730, # 31 (May)
        'July_2023':760, # 30 (June)
        'August_2023':791, # 31 (July)
        'September_2023':822, # 31 (August)
        'October_2023':852, # 30 (September)
        'November_2023':883, # 31 (October)
        'December_2023':913, # 30 (November)
        'January_2024':944, # 31 (December)
        'February_2024':975, # 31 (January)
        'March_2024':1003, # 28 (February)
        'April_2024':1034, # 31 (March)
        'May_2024':1064, # 30 (April)
        'June_2024':1095, # 31 (May)
        'July_2024':1125, # 30 (June)
        'August_2024':1156, # 31 (July)
        'September_2024':1187, # 31 (August)
        'October_2024':1217, # 30 (September)
        'November_2024':1248, # 31 (October)
        'December_2024':1278, # 30 (November) 
    } 
    test_portion_sizes = {
        'June_2021':30, 
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
        'May_2022':31,
        'June_2022':30, 
        'July_2022':31, 
        'August_2022':31, 
        'September_2022':30, 
        'October_2022':31, 
        'November_2022':30, 
        'December_2022':31, 
        'January_2023':31, 
        'February_2023':28, 
        'March_2023':31, 
        'April_2023':30, 
        'May_2023':31,
        'June_2023':30, 
        'July_2023':31, 
        'August_2023':31, 
        'September_2023':30, 
        'October_2023':31, 
        'November_2023':30, 
        'December_2023':31, 
        'January_2024':31, 
        'February_2024':28, 
        'March_2024':31, 
        'April_2024':30, 
        'May_2024':31,
        'June_2024':30, 
        'July_2024':31, 
        'August_2024':31, 
        'September_2024':30, 
        'October_2024':31, 
        'November_2024':30, 
        'December_2024':31,
    }
    full_test_sizes = {
        'June_2021':365, 
        'July_2021':335, 
        'August_2021':304, 
        'September_2021':273, 
        'October_2021':243, 
        'November_2021':212, 
        'December_2021':182, 
        'January_2022':151, 
        'February_2022':120, 
        'March_2022':92, 
        'April_2022':61, 
        'May_2022':31,
    }

    rep_results = {}

    mean_val_nMSEs = []
    mean_val_nRMSEs = []
    mean_val_nMAEs = []
    mean_val_nMAPEs = []

    mean_val_MSEs = []
    mean_val_RMSEs = []
    mean_val_MAEs = []
    mean_val_MAPEs = []
    mean_val_R2s = []

    for rep in range(1, n_reps+1):
        rep_results[rep] = {}
        rep_results[rep]['nMSE'], rep_results[rep]['nRMSE'], rep_results[rep]['nMAE'], rep_results[rep]['nMAPE'] = {}, {}, {}, {}
        rep_results[rep]['MSE'], rep_results[rep]['RMSE'], rep_results[rep]['MAE'], rep_results[rep]['MAPE'], rep_results[rep]['R2'] = {}, {}, {}, {}, {}
        rep_results[rep]['Expected'], rep_results[rep]['Predicted'], rep_results[rep]['Expected_Normalized'], rep_results[rep]['Predicted_Normalized'] = {}, {}, {}, {}

        print(f"\n\t\t\t- Current Rep:{rep}")

        for m in months:
            current_train_split_offset = train_splits_offsets[m]
            current_test_portion_size = test_portion_sizes[m]
            current_full_test_size = full_test_sizes[m]

            print(f"\n\t\t\t\t- Current Rep:{rep} and Current Month: {m}, with a Split Offset of {current_train_split_offset} and a Test Prediction Portion Size of {current_test_portion_size}")

            if selected_target_setting == 'C':
                target_IMFs_bool = False
            else:
                target_IMFs_bool = True

            if selected_input_setting in ['VMD', 'VMD_n_Res', 'VMD_n_CEEMDAN', 'VMD_n_CEEMDAN_n_Res']:
                input_C_bool = False
                input_OHL_bool = False

                VMD_decomposition_bool = True
                input_VMD_Residuals_bool = False
                CEEMDAN_Residuals_decomposition_bool = False
                input_CEEMDAN_Residuals_bool = False

                differencing_bool = False
                target_only_differencing_bool = False
                
                if selected_input_setting in ['VMD_n_Res', 'VMD_n_CEEMDAN', 'VMD_n_CEEMDAN_n_Res']:
                    input_VMD_Residuals_bool = True

                if selected_input_setting in ['VMD_n_CEEMDAN', 'VMD_n_CEEMDAN_n_Res']:
                    CEEMDAN_Residuals_decomposition_bool = True
                
                if selected_input_setting == 'VMD_n_CEEMDAN_n_Res':
                    input_CEEMDAN_Residuals_bool = True
            else:
                input_C_bool = True
                input_OHL_bool = False
                differencing_bool = False
                target_only_differencing_bool = False

                VMD_decomposition_bool = False
                input_VMD_Residuals_bool = False
                CEEMDAN_Residuals_decomposition_bool = False
                input_CEEMDAN_Residuals_bool = False

                if selected_input_setting in ['OHLC', 'Diff_OHLC', 'Diff_C_OHL']:
                    input_OHL_bool = True
                    
                if selected_input_setting in ['Diff_C', 'Diff_OHLC', 'Diff_C_OHL']:
                    differencing_bool = True

                    if selected_input_setting == 'Diff_C_OHL':
                        target_only_differencing_bool = True
            
            current_dataset = CryptoDataset(
                crypto=selected_crypto,
                interval=selected_forecast_interval,

                input_OHL=input_OHL_bool,
                input_C=input_C_bool,

                apply_VMD=VMD_decomposition_bool,
                VMD_n_IMFs=VMD_params['K'],
                VMD_alpha=VMD_params['alpha'],
                VMD_tau=VMD_params['tau'],
                include_VMD_res=input_VMD_Residuals_bool,
                apply_res_CEEMDAN=CEEMDAN_Residuals_decomposition_bool,
                include_CEEMDAN_res=input_CEEMDAN_Residuals_bool,
                target_IMFs=target_IMFs_bool,
                
                apply_differencing=differencing_bool,
                target_only_differencing=target_only_differencing_bool,

                split_offset=current_train_split_offset,
                test_size=current_test_portion_size,
            )

            current_dataset.process_dataset()
            current_dataset.summary(tabs="\t\t\t\t\t")

            IMF_models = []
            IMF_models_training_metrics = []

            if selected_model == 'LSTM':
                if selected_model_setting == 'MIC':
                    model = MIC_LSTM(dataset=current_dataset)
                elif selected_model_setting == 'MM':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(
                            MM_LSTM(
                                dataset=current_dataset,
                                selected_IMF_idx=i,
                            )
                        )
                elif selected_model_setting == 'SI':
                    model = SI_LSTM(dataset=current_dataset)
                else:
                    print("ERROR: Model not available in this Model Setting!")

            elif selected_model == 'GRU':
                if selected_model_setting == 'MIC':
                    model = MIC_GRU(dataset=current_dataset)
                elif selected_model_setting == 'MM':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(
                            MM_GRU(
                                dataset=current_dataset,
                                selected_IMF_idx=i,
                            )
                        )
                elif selected_model_setting == 'SI':
                    model = SI_GRU(dataset=current_dataset)
                else:
                    print("ERROR: Model not available in this Model Setting!")

            elif selected_model == 'BiLSTM':
                if selected_model_setting == 'MIC':
                    model = MIC_BiLSTM(dataset=current_dataset)
                elif selected_model_setting == 'MM':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(
                            MM_BiLSTM(
                                dataset=current_dataset,
                                selected_IMF_idx=i,
                            )
                        )
                elif selected_model_setting == 'SI':
                    model = SI_BiLSTM(dataset=current_dataset)
                else:
                    print("ERROR: Model not available in this Model Setting!")

            elif selected_model == 'BiGRU':
                if selected_model_setting == 'MIC':
                    model = MIC_BiGRU(dataset=current_dataset)
                elif selected_model_setting == 'MM':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(
                            MM_BiGRU(
                                dataset=current_dataset,
                                selected_IMF_idx=i,
                            )
                        )
                elif selected_model_setting == 'SI':
                    model = SI_BiGRU(dataset=current_dataset)
                else:
                    print("ERROR: Model not available in this Model Setting!")

            elif selected_model == 'TCN':
                if selected_model_setting == 'MIC':
                    model = MIC_TCN(current_dataset)
                elif selected_model_setting == 'MIS':
                    model = MIS_TCN(current_dataset)
                elif selected_model_setting == 'MM':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(
                            MM_TCN(
                                dataset=current_dataset,
                                selected_IMF_idx=i,
                            )
                        )
                elif selected_model_setting == 'SIMH':
                    model = SIMH_TCN(current_dataset)
                elif selected_model_setting == 'SI':
                    model = SI_TCN(current_dataset)
                else:
                    print("ERROR: Model not available in this Model Setting!")

            elif selected_model == 'MSRCNN_LSTM':
                if selected_model_setting == 'MIC':
                    model = MIC_MSRCNN_LSTM(dataset=current_dataset)
                elif selected_model_setting == 'MM':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(
                            MM_MSRCNN_LSTM(
                                dataset=current_dataset,
                                selected_IMF_idx=i,
                            )
                        )
                elif selected_model_setting == 'SI':
                    model = SI_MSRCNN_LSTM(dataset=current_dataset)
                else:
                    print("ERROR: Model not available in this Model Setting!")

            elif selected_model == 'MSRCNN_GRU':
                if selected_model_setting == 'MIC':
                    model = MIC_MSRCNN_GRU(dataset=current_dataset)
                elif selected_model_setting == 'MM':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(
                            MM_MSRCNN_GRU(
                                dataset=current_dataset,
                                selected_IMF_idx=i,
                            )
                        )
                elif selected_model_setting == 'SI':
                    model = SI_MSRCNN_GRU(dataset=current_dataset)
                else:
                    print("ERROR: Model not available in this Model Setting!")
            
            elif selected_model == 'ATCN':
                if selected_model_setting == 'MIC':
                    model = MIC_ATCN(dataset=current_dataset)
                elif selected_model_setting == 'MM':
                    for i, _ in enumerate(current_dataset.target_columns):
                        IMF_models.append(
                            MM_ATCN(
                                dataset=current_dataset,
                                selected_IMF_idx=i,
                            )
                        )
                elif selected_model_setting == 'SI':
                    model = SI_ATCN(dataset=current_dataset)
                else:
                    print("ERROR: Model not available in this Model Setting!")
                    
            else:
                print("ERROR: No such Model available!")
                return

            if selected_model_setting == 'MM':
                for i, model in enumerate(IMF_models):
                    print(f"\n\t\t\t\t\t- Training {current_dataset.target_columns[i]} Model:")

                    model, training_metrics = PT_Training_and_Prediction.training(
                        model=model,
                        model_setting=selected_model_setting,
                        dataset=current_dataset,
                        
                        n_epochs=training_params['n_epochs'],
                        batch_size=training_params['batch_size'],
                        optimizer_type=training_params['optimizer'],
                        learning_rate=training_params['learning_rate'],
                        weight_decay=training_params['weight_decay'],
                        momentum=training_params['momentum'],
                        
                        early_stopping_model_base_save_path=f"Modeling/Model_Checkpoints/{selected_input_setting}_{selected_model_setting}_{selected_target_setting}/{selected_model}/{selected_experiment_name}/", 
                        
                        targeted_IMF_idx=i,
                        multi_modal=True,
                        
                        optimization_mode=optimization_mode,
                        verbose=verbosity,
                    )

                    IMF_models[i] = model
                    IMF_models_training_metrics.append(training_metrics)
            else:
                model, training_metrics = PT_Training_and_Prediction.training(
                    model=model,
                    model_setting=selected_model_setting,
                    dataset=current_dataset,
                    
                    n_epochs=training_params['n_epochs'],
                    batch_size=training_params['batch_size'],
                    optimizer_type=training_params['optimizer'],
                    learning_rate=training_params['learning_rate'],
                    weight_decay=training_params['weight_decay'],
                    momentum=training_params['momentum'],
                    
                    early_stopping_model_base_save_path=f"Modeling/Model_Checkpoints/{selected_input_setting}_{selected_model_setting}_{selected_target_setting}/{selected_model}/{selected_experiment_name}/", 
                    
                    optimization_mode=optimization_mode,
                    verbose=verbosity,
                )
            

            if not optimization_mode:
                # price_data_X_test_portion_to_predict = current_dataset.price_data_X_test[:current_prediction_portion_size]

                expected_close_prices = current_dataset.y_test_close_price[current_dataset.horizon-1:(len(current_dataset.price_data_X_test)+current_dataset.horizon-1)] # current_dataset.y_test_close_price[current_dataset.horizon-1:(len(price_data_X_test_portion_to_predict)+current_dataset.horizon-1)]
                expected_normalized_close_prices = current_dataset.y_test_normalized_close_price[current_dataset.horizon-1:(len(current_dataset.price_data_X_test)+current_dataset.horizon-1)] # current_dataset.y_test_normalized_close_price[current_dataset.horizon-1:(len(price_data_X_test_portion_to_predict)+current_dataset.horizon-1)]

                if current_dataset.target_IMFs:
                    expected_IMFs = current_dataset.y_test_IMFs[current_dataset.horizon-1:(len(current_dataset.price_data_X_test)+current_dataset.horizon-1)] # current_dataset.y_test_IMFs[current_dataset.horizon-1:(len(price_data_X_test_portion_to_predict)+current_dataset.horizon-1)]
                    expected_normalized_IMFs = current_dataset.y_test_normalized_IMFs[current_dataset.horizon-1:(len(current_dataset.price_data_X_test)+current_dataset.horizon-1)] # current_dataset.y_test_normalized_IMFs[current_dataset.horizon-1:(len(price_data_X_test_portion_to_predict)+current_dataset.horizon-1)]

                if selected_model_setting == 'MM':
                    all_IMFs_preds = []

                    for i, model in enumerate(IMF_models):
                        current_preds = PT_Training_and_Prediction.predict(
                            model=model,
                            price_data_X=current_dataset.price_data_X_test,
                        )

                        current_inverse_transformed_preds = current_dataset.inverse_transform_specific_IMF_predictions(
                            preds=current_preds,
                            selected_IMF_idx=i,
                        )
                        
                        current_IMF_preds = current_inverse_transformed_preds
                        current_IMF_normalized_preds = current_preds

                        Save_Results_Functions.record_rep_results(
                            rep_results=rep_results,
                            rep=rep,

                            expected_vals=expected_IMFs[:,i].reshape(-1,1),
                            normalized_expected_vals=expected_normalized_IMFs[:,i].reshape(-1,1),
                            predicted_vals=current_IMF_preds,
                            normalized_predicted_vals=current_IMF_normalized_preds,
                            
                            var_name=f'Close_Price_IMF_{i+1}',
                        )
                        
                        all_IMFs_preds.append(current_IMF_preds)

                    IMFs_preds = np.concatenate(
                        all_IMFs_preds, 
                        axis=1,
                    )
                    close_price_preds = IMFs_preds.sum(axis=1).reshape(-1,1)
                    normalized_close_price_preds = current_dataset.normalize_price_predictions(preds=close_price_preds) 
                else:
                    preds = PT_Training_and_Prediction.predict(
                        model=model,
                        price_data_X=current_dataset.price_data_X_test,
                    )
                    
                    inverse_transformed_preds = current_dataset.inverse_transform_predictions(
                        preds=preds,
                        inversion_differencing_offset_length=current_full_test_size,
                    )
            
                    if current_dataset.target_IMFs:
                        IMFs_preds = inverse_transformed_preds
                        normalized_IMFs_preds = preds
                        
                        close_price_preds = IMFs_preds.sum(axis=1).reshape(-1,1)
                        normalized_close_price_preds = current_dataset.normalize_price_predictions(preds=close_price_preds) 

                        for i, _ in enumerate(current_dataset.target_columns):
                            Save_Results_Functions.record_rep_results(
                                rep_results=rep_results,
                                rep=rep,
                                
                                expected_vals=expected_IMFs[:,i].reshape(-1,1),
                                normalized_expected_vals=expected_normalized_IMFs[:,i].reshape(-1,1),
                                predicted_vals=IMFs_preds[:,i].reshape(-1,1),
                                normalized_predicted_vals=normalized_IMFs_preds[:,i].reshape(-1,1),

                                var_name=f'Close_Price_IMF_{i+1}',
                            )
                    else:
                        close_price_preds = inverse_transformed_preds
                        normalized_close_price_preds = current_dataset.normalize_price_predictions(preds=close_price_preds)

                rep_results = Save_Results_Functions.record_rep_results(
                    rep_results=rep_results,
                    rep=rep,
                    
                    expected_vals=expected_close_prices,
                    normalized_expected_vals=expected_normalized_close_prices,
                    predicted_vals=close_price_preds,
                    normalized_predicted_vals=normalized_close_price_preds,
                    
                    var_name='Close_Price',
                )

            else:
                expected_close_prices = current_dataset.y_val_close_price
                expected_normalized_close_prices = current_dataset.y_val_normalized_close_price

                if selected_model_setting == 'MM':
                    all_IMFs_preds = []

                    for i, model in enumerate(IMF_models):
                        current_preds = PT_Training_and_Prediction.predict(
                            model=model,
                            price_data_X=current_dataset.price_data_X_val,
                        )

                        current_inverse_transformed_preds = current_dataset.inverse_transform_specific_IMF_predictions(
                            preds=current_preds,
                            selected_IMF_idx=i,
                        )
                        
                        current_IMF_preds = current_inverse_transformed_preds
                        current_IMF_normalized_preds = current_preds
                        
                        all_IMFs_preds.append(current_IMF_preds)

                    IMFs_preds = np.concatenate(
                        all_IMFs_preds, 
                        axis=1,
                    )
                    close_price_preds = IMFs_preds.sum(axis=1).reshape(-1,1)
                    normalized_close_price_preds = current_dataset.normalize_price_predictions(preds=close_price_preds) 
                else:
                    preds = PT_Training_and_Prediction.predict(
                        model=model,
                        price_data_X=current_dataset.price_data_X_val,
                    )

                    inverse_transformed_preds = current_dataset.inverse_transform_predictions(
                        preds=preds,
                        inversion_differencing_offset_length=(len(current_dataset.y_val)) + current_full_test_size,
                    )

                    if current_dataset.target_IMFs:
                        IMFs_preds = inverse_transformed_preds
                        normalized_IMFs_preds = preds
                        
                        close_price_preds = IMFs_preds.sum(axis=1).reshape(-1,1)
                        normalized_close_price_preds = current_dataset.normalize_price_predictions(preds=close_price_preds) 
                    else:
                        close_price_preds = inverse_transformed_preds
                        normalized_close_price_preds = current_dataset.normalize_price_predictions(preds=close_price_preds)

                rep_results = Save_Results_Functions.record_rep_results(
                    rep_results=rep_results,
                    rep=rep,
                    
                    expected_vals=expected_close_prices,
                    normalized_expected_vals=expected_normalized_close_prices,
                    predicted_vals=close_price_preds,
                    normalized_predicted_vals=normalized_close_price_preds,
                    
                    var_name='Close_Price',
                )

        mean_val_nMSEs.append(np.mean(rep_results[rep]['nMSE']['Close_Price']))
        mean_val_nRMSEs.append(np.mean(rep_results[rep]['nRMSE']['Close_Price']))
        mean_val_nMAEs.append(np.mean(rep_results[rep]['nMAE']['Close_Price']))
        mean_val_nMAPEs.append(np.mean(rep_results[rep]['nMAPE']['Close_Price']))
        
        mean_val_MSEs.append(np.mean(rep_results[rep]['MSE']['Close_Price']))
        mean_val_RMSEs.append(np.mean(rep_results[rep]['RMSE']['Close_Price']))
        mean_val_MAEs.append(np.mean(rep_results[rep]['MAE']['Close_Price']))
        mean_val_MAPEs.append(np.mean(rep_results[rep]['MAPE']['Close_Price']))
        mean_val_R2s.append(np.mean(rep_results[rep]['R2']['Close_Price']))

    if save_results:
        base_res_map = f'Modeling/Results/Baseline_Experiment/{selected_model_setting}/{selected_input_setting}__{selected_target_setting}/{selected_model}/{selected_experiment_name}/{selected_crypto}/{selected_forecast_interval}/'

        Save_Results_Functions.save_metrics(
            base_res_map=base_res_map,
            rep_results=rep_results,
        )

        Save_Results_Functions.save_outputs(
            base_res_map=base_res_map,
            rep_results=rep_results,
        )

        Save_Results_Functions.save_average_metrics(
            base_res_map=base_res_map,
            rep_results=rep_results,
        )    

        Save_Results_Functions.save_average_outputs(
            base_res_map=base_res_map,
            rep_results=rep_results,
        )

    end_time_experiment = time.time()
    elapsed_time = end_time_experiment - start_time_experiment

    print(f"Total Time Elapsed Since Starting Experiment: {int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {int(elapsed_time % 60)}s Minutes")

    if optimization_mode:
        return {'mean_val_nmse':np.mean(mean_val_nMSEs),
                'mean_val_nrmse':np.median(mean_val_nRMSEs),
                'mean_val_nmae':np.mean(mean_val_nMAEs),
                'mean_val_nmape':np.median(mean_val_nMAPEs),

                'mean_val_mse':np.mean(mean_val_MSEs),
                'mean_val_rmse':np.median(mean_val_RMSEs),
                'mean_val_mae':np.mean(mean_val_MAEs),
                'mean_val_mape':np.median(mean_val_MAPEs),
                'mean_val_r2':np.median(mean_val_R2s),} 
    else:
        if selected_model_setting == 'MM':
            return IMF_models_training_metrics
        else:
            return training_metrics
                

def main():
    parser = argparse.ArgumentParser(description="Baseline Experiment")

    parser.add_argument(
        '--crypto', 
        type=str, 
        default='BTC', 
        choices=['BTC', 'ETH', 'LTC', 'XMR', 'XRP'], 
        help="The Cryptocurrency of which the Close Price is Forecasted in the Chosen Experiment",
    )
    parser.add_argument(
        '--price_timespan', 
        type=str, 
        default='Baseline', 
        choices=['Baseline', 'Extended', 'Full'], 
        help="The Chosen Experiment",
    )
    parser.add_argument(
        '--price_source', 
        type=str, 
        default='Binance', 
        choices=['Binance', 'CoinMarketCap', 'Investing'], 
        help="The Chosen Price Source",
    )
    parser.add_argument(
        '--price_currency', 
        type=str, 
        default='USDT', 
        choices=['USDT', 'USD'], 
        help="The Chosen Price Data Source",
    )
    parser.add_argument(
        '--data_freq', 
        type=str, 
        default='1_Day', 
        choices=['1_Hour', '2_Hour', '4_Hour', '6_Hour', '8_Hour', '12_Hour', '1_Day'], 
        help="The Frequency of the Cryptocurrency Data",
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='GRU', 
        choices=['LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'TCN', 'ATCN', 'MSRCNN_LSTM', 'MSRCNN_GRU'], 
        help="The Model utilized in executing the Chosen Experiment",
    )
    parser.add_argument(
        '--input_setting', 
        type=str, 
        default='VMD', 
        choices=['C', 'OHLC', 'Diff_C', 'Diff_OHLC', 'Diff_C_OHL', 'VMD', 'VMD_n_Res', 'VMD_n_CEEMDAN', 'VMD_n_CEEMDAN_n_Res'],
        help="The Baseline Input Setting",
    )
    parser.add_argument(
        '--model_setting', 
        type=str, 
        default='SI', 
        choices=['SI', 'SIMH', 'MIC', 'MIS', 'MM'],
        help="The Baseline Model Setting",
    )
    parser.add_argument(
        '--target_setting', 
        type=str, 
        default='C', 
        choices=['C', 'IMFs'],
        help="The Baseline Target Setting",
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='Default', 
        required=False,
        help="The Name of the Experiment"
    )
    parser.add_argument(
        '--hyperparameter_tuning', 
        action='store_true',
    )

    args = parser.parse_args()

    if args.hyperparameter_tuning:
        print(f"\nStarting Hyperparameter Search of Baseline Experiment:")
    else:
        print(f"\nStarting Baseline Experiment:")

    print(f"\t- Crypto to be Forecasted: {args.crypto}")
    print(f"\t- Forecasting Interval: {args.interval}")
    print(f"\t- Model to be Utilized: {args.model}")
    print(f"\t- Experimental Setting: {args.input_setting}_{args.model_setting}_{args.target_setting}\n")

    if args.crypto == 'BTC':
        optimal_vmd_params = {'K':6, # 10 # 13 # 8 # 15 # 6
                              'alpha':250, # 3250 # 2750 # 1750 # 8000 # 250
                              'tau':0.15} # 0.45 # 0.05 # 0.01 # 0 # .15
    elif args.crypto == 'ETH':
        optimal_vmd_params = {'K':14,
                              'alpha':1250,
                              'tau':.25}
    elif args.crypto == 'LTC':
        optimal_vmd_params = {'K':6,
                              'alpha':250,
                              'tau':.3}
    elif args.crypto == 'XMR':
        optimal_vmd_params = {'K':6,
                              'alpha':250,
                              'tau':.3}
    else:
        optimal_vmd_params = {'K':10,
                              'alpha':1000,
                              'tau':.05}

    if args.hyperparameter_tuning:
        optuna_hyperparameter_tuning(
            selected_crypto=args.crypto,
            selected_forecast_interval=args.interval,
            selected_model=args.model,

            selected_input_setting=args.input_setting,
            selected_model_setting=args.model_setting,
            selected_target_setting=args.target_setting,
            selected_experiment_name=args.experiment_name,
        )
    else:
        run_experiment(
            selected_crypto=args.crypto,
            selected_forecast_interval=args.interval,
            selected_model=args.model,

            selected_input_setting=args.input_setting,
            selected_model_setting=args.model_setting,
            selected_target_setting=args.target_setting,
            selected_experiment_name=args.experiment_name,

            VMD_params=optimal_vmd_params,
            n_reps=3,
            save_results=True,
        )


if __name__ == "__main__":
    main()
