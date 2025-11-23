import optuna
import os
import numpy as np

from functools import partial
from Experiments.Experiment import run_experiment


def optuna_objective_function(
        trial,
        
        selected_cryptos,
        selected_experiment,
        selected_data_frequency,
        selected_price_source,
        selected_price_currency,
            
        selected_model,
        selected_input,
        selected_transformation,
        selected_VMD_mode,
        selected_normalization_moment,
        selected_model_setting,
        selected_experiment_name,

        tune_model,
        train_params,
        tune_VMD,
        VMD_params,
):
    val_nmses = []
    val_nrmses = []
    val_nmaes = []
    val_nmapes = []
    val_r2s = []

    for current_crypto in  selected_cryptos:
        if selected_transformation in ['VMD', 'VMD_n_Res', 'VMD_n_CEEMDAN', 'VMD_n_CEEMDAN_n_Res', 'MVMD', 'MVMD_n_Res']:
            if tune_VMD:
                vmd_params = {
                    'K':trial.suggest_int(
                        name=f'{current_crypto}_VMD_K', 
                        low=2, 
                        high=15,
                    ),
                    'alpha':trial.suggest_int(
                        name=f'{current_crypto}_VMD_alpha',
                        low=250, 
                        high=10000, 
                        step=250,
                    ),
                    'tau':trial.suggest_categorical(
                        name=f'{current_crypto}_VMD_tau', 
                        choices=[0.0, 0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                    ),
                }

                if selected_VMD_mode == 'Dyn':
                    vmd_params['window_size'] = trial.suggest_int(
                        name=f'{current_crypto}_window_size', 
                        low=6, 
                        high=730, 
                        step=10,
                    )

            else:
                vmd_params = VMD_params[current_crypto]

        else:
            vmd_params = VMD_params[current_crypto]

        if tune_model:
            training_params = {
                'n_epochs':trial.suggest_int(
                    name='TRAIN_n_epochs', 
                    low=50, 
                    high=600, 
                    step=10,
                ),
                'batch_size':trial.suggest_int(
                    name='TRAIN_batch_size', 
                    low=16, 
                    high=256, 
                    step=16,
                ),
                'optimizer':trial.suggest_categorical(
                    name="TRAIN_optimizer", 
                    choices=['Adam', 'AdamW', 'RMSprop', 'SGD'],
                ),
                'learning_rate':trial.suggest_categorical(
                    name="TRAIN_learning_rate", 
                    choices=[1e-2, 1e-3, 1e-4, 1e-5],
                ),
                'weight_decay':trial.suggest_categorical(
                    name="TRAIN_weight_decay", 
                    choices=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                ),
            }

            if selected_model in ['GRU', 'LSTM', 'BiGRU', 'BiLSTM']:
                model_params = {
                    'RNN_n_layers':trial.suggest_int(
                        name='RNN_n_layers', 
                        low=1, 
                        high=3,
                    ),
                    'RNN_dropout_rate': trial.suggest_float(
                        name='RNN_dropout_rate', 
                        low=0.0, 
                        high=0.5, 
                        step=0.05,
                    ),

                    'FC_n_layers':trial.suggest_int(
                        name='FC_n_layers', 
                        low=1, 
                        high=3,
                    ),
                    'FC_activation':trial.suggest_categorical(
                        name='FC_activation', 
                        choices=['elu', 'gelu', 'leaky_relu', 'relu', 'selu']
                    ),
                    'FC_dropout_rate': trial.suggest_float(
                        name='FC_dropout_rate', 
                        low=0.0, 
                        high=0.5, 
                        step=0.05,
                    ),
                }

                RNN_dim_l1 = trial.suggest_int(
                    name='RNN_dim_l1', 
                    low=32, 
                    high=512, 
                    step=32,
                )

                if model_params['RNN_n_layers'] > 1:
                    RNN_dim_l2 = trial.suggest_int(
                        name='RNN_dim_l2', 
                        low=32, 
                        high=512, 
                        step=32,
                    )

                    if model_params['RNN_n_layers'] == 3:
                        RNN_dim_l3 = trial.suggest_int(
                            name='RNN_dim_l3', 
                            low=32, 
                            high=512, 
                            step=32,
                        )

                        model_params['RNN_dims'] = [RNN_dim_l1, RNN_dim_l2, RNN_dim_l3]
                    else:
                        model_params['RNN_dims'] = [RNN_dim_l1, RNN_dim_l2]
                else:
                    model_params['RNN_dims'] = [RNN_dim_l1]

                FC_dim_l1 = trial.suggest_int(
                    name='FC_dim_l1', 
                    low=16, 
                    high=256, 
                    step=16,
                )

                if model_params['FC_n_layers'] > 1:
                    FC_dim_l2 = trial.suggest_int(
                        name='FC_dim_l2', 
                        low=16, 
                        high=256, 
                        step=16,
                    )

                    if model_params['FC_n_layers'] == 3:
                        FC_dim_l3 = trial.suggest_int(
                            name='FC_dim_l3', 
                            low=16, 
                            high=256, 
                            step=16,
                        )

                        model_params['FC_dims'] = [FC_dim_l1, FC_dim_l2, FC_dim_l3]
                    else:
                        model_params['FC_dims'] = [FC_dim_l1, FC_dim_l2]
                else:
                    model_params['FC_dims'] = [FC_dim_l1]
            
            elif selected_model == 'TCN':
                model_params = {
                    'TCN_n_blocks':trial.suggest_int(
                        name='TCN_n_blocks', 
                        low=2, 
                        high=8,
                    ),
                    'TCN_n_filters':trial.suggest_int(
                        name='TCN_n_filters', 
                        low=16, 
                        high=256, 
                        step=16,
                    ),
                    'TCN_kernel_size':trial.suggest_categorical(
                        name='TCN_kernel_size', 
                        choices=[2, 3, 5, 7],
                    ),
                    'TCN_dilation_base':trial.suggest_categorical(
                        name='TCN_dilation_base', 
                        choices=[2, 3, 4],
                    ),
                    'TCN_dropout_rate':trial.suggest_float(
                        name='TCN_dropout_rate', 
                        low=0.0, 
                        high=0.5, 
                        step=0.05,
                    ),
                    'TCN_activation':trial.suggest_categorical(
                        name='TCN_activation', 
                        choices=['relu', 'gelu', 'selu', 'elu'],
                    ),

                    'FC_n_layers':trial.suggest_int(
                        name='FC_n_layers', 
                        low=0, 
                        high=3,
                    ),
                }

                if model_params['FC_n_layers'] == 0:
                    FC_dim_l1 = 0
                    FC_dim_l2 = 0
                    FC_dim_l3 = 0
                    model_params['FC_activation'] = 'None'
                    model_params['FC_dropout_rate'] = 0.0

                else:
                    FC_dim_l1 = trial.suggest_int(
                        name='FC_dim_l1', 
                        low=16, 
                        high=256, 
                        step=16,
                    )
                    model_params['FC_activation'] = trial.suggest_categorical(
                        name='FC_activation', 
                        choices=['elu', 'gelu', 'leaky_relu', 'relu', 'selu'],
                    )
                    model_params['FC_dropout_rate'] = trial.suggest_float(
                        name='FC_dropout_rate', 
                        low=0.0, 
                        high=0.5, 
                        step=0.05,
                    )
                
                    if model_params['FC_n_layers'] > 1:
                        FC_dim_l2 = trial.suggest_int(
                            name='FC_dim_l2', 
                            low=16, 
                            high=256, 
                            step=16,
                        )

                        if model_params['FC_n_layers'] == 3:
                            FC_dim_l3 = trial.suggest_int(
                                name='FC_dim_l3', 
                                low=16, 
                                high=256, 
                                step=16,
                            ),
                        else:
                            FC_dim_l3 = 0
                    else:
                        FC_dim_l2 = 0
                        FC_dim_l3 = 0

                model_params['FC_dims'] = [FC_dim_l1, FC_dim_l2, FC_dim_l3]

            else:
                model_params = {}
        else:
            training_params = train_params
            model_params = {}

        current_validation_results = run_experiment(
            selected_crypto=current_crypto,
            selected_experiment=selected_experiment,
            selected_data_frequency=selected_data_frequency,
            selected_price_source=selected_price_source,
            selected_price_currency=selected_price_currency,
                
            selected_model=selected_model,
            selected_input=selected_input,
            selected_transformation=selected_transformation,
            selected_VMD_mode=selected_VMD_mode,
            selected_normalization_moment=selected_normalization_moment,
            selected_model_setting=selected_model_setting,
            selected_experiment_name=f'{selected_experiment_name}_HT',
            
            VMD_params=vmd_params,
            training_params=training_params,
            model_params=model_params,
            
            n_reps=1,
            save_results=False,
            optimization_n_folds=1,
            VMD_optimization_mode=tune_VMD,
            model_optimization_mode=tune_model,
            verbosity=0,
        )
        
        val_nmses.append(current_validation_results['mean_val_nmse'])
        val_nrmses.append(current_validation_results['mean_val_nrmse'])
        val_nmaes.append(current_validation_results['mean_val_nmae'])
        val_nmapes.append(current_validation_results['mean_val_nmape'])
        val_r2s.append(current_validation_results['mean_val_r2'])

    trial.set_user_attr('mean_val_nmse', np.array(val_nmses).mean())
    trial.set_user_attr('mean_val_nrmse', np.array(val_nrmses).mean())
    trial.set_user_attr('mean_val_nmae', np.array(val_nmaes).mean())
    trial.set_user_attr('mean_val_nmape', np.array(val_nmapes).mean())
    trial.set_user_attr('mean_val_r2', np.array(val_r2s).mean())

    return np.array(val_nmses).mean()


def optuna_hyperparameter_tuning(
        selected_cryptos,
        selected_experiment,
        selected_data_frequency,
        selected_price_source,
        selected_price_currency,
            
        selected_model,
        selected_input,
        selected_transformation,
        selected_VMD_mode,
        selected_normalization_moment,
        selected_model_setting,

        selected_experiment_name,

        tune_model,
        train_params,
        tune_VMD,
        VMD_params,
):
    
    study = optuna.create_study(direction='minimize')

    study.optimize(
        partial(
            optuna_objective_function, 

            selected_cryptos=selected_cryptos,
            selected_experiment=selected_experiment,
            selected_data_frequency=selected_data_frequency,
            selected_price_source=selected_price_source,
            selected_price_currency=selected_price_currency,
            
            selected_model=selected_model,
            selected_input=selected_input,
            selected_transformation=selected_transformation,
            selected_VMD_mode=selected_VMD_mode,
            selected_normalization_moment=selected_normalization_moment,
            selected_model_setting=selected_model_setting,
            selected_experiment_name=selected_experiment_name,

            tune_model=tune_model,
            train_params=train_params,
            tune_VMD=tune_VMD,
            VMD_params=VMD_params,
        ), 
        n_trials=2,
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
    hyperparameter_tuning_results_df['Mean_Val_R2'] = [t.user_attrs["mean_val_r2"] for t in study.trials]

    hyperparameter_tuning_results_df = hyperparameter_tuning_results_df.sort_values(by='Mean_Val_nMSE', ignore_index=True)
    
    hyperparameter_tuning_results_df = hyperparameter_tuning_results_df.drop(
        columns=[
            'user_attrs_mean_val_nmae',
            'user_attrs_mean_val_nmape',
            'user_attrs_mean_val_nmse',
            'user_attrs_mean_val_nrmse',
            'user_attrs_mean_val_r2',
        ], 
    )

    if tune_VMD:
        subdir = 'VMD_Tuning'
    else:
        subdir = 'Model_Tuning'

    if 'VMD' in selected_transformation:
        res_map = f"Modeling/Results/Hyperparameter_Tuning_Results/{subdir}/{selected_experiment}/{selected_model_setting}_{selected_model}/{selected_normalization_moment}_{selected_VMD_mode}_{selected_transformation}__{'_'.join(selected_input)}/{selected_experiment_name}/{'_'.join(selected_cryptos)}/{selected_data_frequency}/"
    else:
        res_map = f"Modeling/Results/Hyperparameter_Tuning_Results/{subdir}/{selected_experiment}/{selected_model_setting}_{selected_model}/{selected_normalization_moment}_{selected_transformation}__{'_'.join(selected_input)}/{selected_experiment_name}/{'_'.join(selected_cryptos)}/{selected_data_frequency}/"

    if not os.path.exists(res_map):
        os.makedirs(res_map)

    hyperparameter_tuning_results_df.to_csv(f'{res_map}Optuna_Results.csv')

    print('Best hyperparameters: ', study.best_params)
    print('Best performance: ', study.best_value)
