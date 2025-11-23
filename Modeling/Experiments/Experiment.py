import numpy as np
import time

from datetime import datetime
from dateutil.relativedelta import relativedelta

from Models.Individual_Models.Single_Input.RNNs import SI_LSTM, SI_GRU, SI_BiLSTM, SI_BiGRU

from Models.Individual_Models.Single_Input.Others import SI_TCN
from Models.Individual_Models.Multi_Input.Others import MIC_TCN, MIS_TCN 
from Models.Individual_Models.Multi_Modal.Others import MM_TCN

from Models.Hybrid_Models.Single_Input.SI_ATCN import SI_ATCN
from Models.Hybrid_Models.Multi_Input.MI_ATCN import MIC_ATCN
from Models.Hybrid_Models.Multi_Modal.MM_ATCN import MM_ATCN

from Data_Util.Load_Dataset import crypto_dataset_loader
from PT_Util.Training_and_Prediction import uni_modal_model_training, multi_modal_model_training, model_predict
from Results_Util import Save_Results_Functions


def run_experiment(
        selected_crypto,
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
        
        VMD_params,
        training_params,
        model_params={},
        
        save_results=False,
        n_reps=3,
        optimization_n_folds=12,
        VMD_optimization_mode=False,
        model_optimization_mode=False,
        verbosity=1
):
    start_time_experiment = time.time()

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

        if VMD_optimization_mode or model_optimization_mode:
            current_train_test_split_date = datetime(year=2021, month=6, day=1) - relativedelta(months=optimization_n_folds)
            final_next_train_test_split_date = datetime(year=2021, month=6, day=1)
        else:
            current_train_test_split_date = datetime(year=2021, month=6, day=1)

            if selected_experiment == 'Extended':
                final_next_train_test_split_date = datetime(year=2024, month=2, day=1)
            elif selected_experiment == 'Full':
                final_next_train_test_split_date = datetime(year=2025, month=1, day=1)
            else:
                final_next_train_test_split_date = datetime(year=2021, month=8, day=1) # datetime(year=2022, month=6, day=1)

        while current_train_test_split_date != final_next_train_test_split_date:
            print(f"\t\t\t\t- Current Rep:{rep} and Current Month: {current_train_test_split_date.strftime('%B %Y')}")

            next_train_test_split_date = current_train_test_split_date + relativedelta(months=1)
            
            current_dataset = crypto_dataset_loader(
                crypto=selected_crypto,
                experiment=selected_experiment,
                data_frequency=selected_data_frequency,
                price_source=selected_price_source,
                price_currency=selected_price_currency,

                train_test_split_date=current_train_test_split_date,
                next_train_test_split_date=next_train_test_split_date,

                input_window=30,
                output_window=1,
                stride=1,
                horizon=1,

                transformation=selected_transformation,
                normalization_moment=selected_normalization_moment,
                
                VMD_params=VMD_params,
                VMD_mode=selected_VMD_mode,
            )
            current_dataset.process_dataset()

            if verbosity:
                current_dataset.summary(tabs="\t\t\t\t\t")

            current_train_test_split_date = next_train_test_split_date
            IMF_models = []

            if selected_model == 'LSTM':
                if selected_model_setting == 'SI':
                    if model_optimization_mode:
                        model = SI_LSTM(
                            dataset=current_dataset,
                            inputs=selected_input,

                            n_layers=model_params['RNN_n_layers'],
                            dims=model_params['RNN_dims'],
                            dropout_rate=model_params['RNN_dropout_rate'],
                            
                            fc_n_layers=model_params['FC_n_layers'],
                            fc_dims=model_params['FC_dims'],
                            fc_dropout_rate=model_params['FC_dropout_rate'],
                            fc_activation=model_params['FC_activation'],
                        )
                    else:
                        model = SI_LSTM(
                            dataset=current_dataset,
                            inputs=selected_input,
                        )
                else:
                    print("ERROR: Model not available in this Model Setting!")

            elif selected_model == 'GRU':
                if selected_model_setting == 'SI':
                    if model_optimization_mode:
                        model = SI_GRU(
                            dataset=current_dataset,
                            inputs=selected_input,

                            n_layers=model_params['RNN_n_layers'],
                            dims=model_params['RNN_dims'],
                            dropout_rate=model_params['RNN_dropout_rate'],
                            
                            fc_n_layers=model_params['FC_n_layers'],
                            fc_dims=model_params['FC_dims'],
                            fc_dropout_rate=model_params['FC_dropout_rate'],
                            fc_activation=model_params['FC_activation'],
                        )
                    else:
                        model = SI_GRU(
                            dataset=current_dataset,
                            inputs=selected_input,
                        )
                else:
                    print("ERROR: Model not available in this Model Setting!")

            elif selected_model == 'BiLSTM':
                if selected_model_setting == 'SI':
                    if model_optimization_mode:
                        model = SI_BiLSTM(
                            dataset=current_dataset,
                            inputs=selected_input,

                            n_layers=model_params['RNN_n_layers'],
                            dims=model_params['RNN_dims'],
                            dropout_rate=model_params['RNN_dropout_rate'],
                            
                            fc_n_layers=model_params['FC_n_layers'],
                            fc_dims=model_params['FC_dims'],
                            fc_dropout_rate=model_params['FC_dropout_rate'],
                            fc_activation=model_params['FC_activation'],
                        )
                    else:
                        model = SI_BiLSTM(
                            dataset=current_dataset,
                            inputs=selected_input,
                        )
                else:
                    print("ERROR: Model not available in this Model Setting!")

            elif selected_model == 'BiGRU':
                if selected_model_setting == 'SI':
                    if model_optimization_mode:
                        model = SI_BiGRU(
                            dataset=current_dataset,
                            inputs=selected_input,

                            n_layers=model_params['RNN_n_layers'],
                            dims=model_params['RNN_dims'],
                            dropout_rate=model_params['RNN_dropout_rate'],
                            
                            fc_n_layers=model_params['FC_n_layers'],
                            fc_dims=model_params['FC_dims'],
                            fc_dropout_rate=model_params['FC_dropout_rate'],
                            fc_activation=model_params['FC_activation'],
                        )
                    else:
                        model = SI_BiGRU(
                            dataset=current_dataset,
                            inputs=selected_input,
                        )
                else:
                    print("ERROR: Model not available in this Model Setting!")

            elif selected_model == 'TCN':
                if selected_model_setting == 'MIC':
                    model = MIC_TCN(
                        dataset=current_dataset,
                        inputs=selected_input,
                    )
                elif selected_model_setting == 'MIS':
                    model = MIS_TCN(
                        dataset=current_dataset,
                        inputs=selected_input,
                    )
                elif selected_model_setting == 'MM':
                    for i, _ in enumerate(current_dataset.close_price_data_X_names):
                        IMF_models.append(
                            MM_TCN(
                                dataset=current_dataset,
                                inputs=selected_input,
                                selected_IMF_idx=i,
                            )
                        )
                elif selected_model_setting == 'SI':
                    if model_optimization_mode:
                        model = SI_TCN(
                            dataset=current_dataset,
                            inputs=selected_input,

                            n_blocks=model_params['TCN_n_blocks'],
                            n_filters=model_params['TCN_n_filters'],
                            kernel_size=model_params['TCN_kernel_size'],
                            dilation_base=model_params['TCN_dilation_base'],
                            dropout_rate=model_params['TCN_dropout_rate'],
                            activation=model_params['TCN_activation'],

                            fc_n_layers=model_params['FC_n_layers'],
                            fc_dims=model_params['FC_dims'],
                            fc_dropout_rate=model_params['FC_dropout_rate'],
                            fc_activation=model_params['FC_activation'],
                        )
                    else:
                        model = SI_TCN(
                            dataset=current_dataset,
                            inputs=selected_input,
                        )
                else:
                    print("ERROR: Model not available in this Model Setting!")
            
            elif selected_model == 'ATCN':
                if selected_model_setting == 'MIC':
                    model = MIC_ATCN(dataset=current_dataset)
                elif selected_model_setting == 'MM':
                    for i, _ in enumerate(current_dataset.close_price_data_X_names):
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

            if 'VMD' in selected_transformation:
                es_model_base_save_path=f"Modeling/Model_Checkpoints/{selected_experiment}/{selected_model_setting}_{selected_model}/{selected_normalization_moment}_{selected_VMD_mode}_{selected_transformation}__{'_'.join(selected_input)}/{selected_experiment_name}/{selected_crypto}/{selected_data_frequency}/"
            else:
                es_model_base_save_path=f"Modeling/Model_Checkpoints/{selected_experiment}/{selected_model_setting}_{selected_model}/{selected_normalization_moment}_{selected_transformation}__{'_'.join(selected_input)}/{selected_experiment_name}/{selected_crypto}/{selected_data_frequency}/"

            if selected_model_setting == 'MM':
                IMF_models, training_metrics = multi_modal_model_training(
                    models=IMF_models,
                    dataset=current_dataset,
                    
                    batch_size=training_params['batch_size'],
                    optimizer_type=training_params['optimizer'],
                    learning_rate=training_params['learning_rate'],
                    weight_decay=training_params['weight_decay'],
                    
                    early_stopping_model_base_save_path=es_model_base_save_path,
                                        
                    VMD_optimization_mode=VMD_optimization_mode,
                    model_optimization_mode=model_optimization_mode,
                    verbose=verbosity,
                )
            else:
                model, training_metrics = uni_modal_model_training(
                    model=model,
                    dataset=current_dataset,
                    
                    batch_size=training_params['batch_size'],
                    optimizer_type=training_params['optimizer'],
                    learning_rate=training_params['learning_rate'],
                    weight_decay=training_params['weight_decay'],
                    
                    early_stopping_model_base_save_path=es_model_base_save_path, 
                    
                    VMD_optimization_mode=VMD_optimization_mode,
                    model_optimization_mode=model_optimization_mode,
                    verbose=verbosity,
                )
            
            expected_close_prices = current_dataset.y_test_close_price 
            expected_normalized_close_prices = current_dataset.y_test_normalized_close_price 

            if selected_model_setting == 'MM':
                all_IMFs_preds = []

                for i, model in enumerate(IMF_models):
                    current_IMF_preds = model_predict(
                        model=model,

                        open_price_data_X=current_dataset.open_price_data_X_test,
                        high_price_data_X=current_dataset.high_price_data_X_test,
                        low_price_data_X=current_dataset.low_price_data_X_test,
                        close_price_data_X=current_dataset.close_price_data_X_test,
                    )
                    
                    all_IMFs_preds.append(current_IMF_preds)

                IMFs_preds = np.concatenate(
                    all_IMFs_preds, 
                    axis=-1,
                )

                normalized_close_price_preds = np.expand_dims(
                    IMFs_preds.sum(axis=-1), 
                    axis=-1,
                )

                close_price_preds = current_dataset.inverse_transform_predictions(preds=normalized_close_price_preds) 

            else:
                preds = model_predict(
                    model=model,

                    open_price_data_X=current_dataset.open_price_data_X_test,
                    high_price_data_X=current_dataset.high_price_data_X_test,
                    low_price_data_X=current_dataset.low_price_data_X_test,
                    close_price_data_X=current_dataset.close_price_data_X_test,
                )

                inverse_transformed_preds = current_dataset.inverse_transform_predictions(
                    preds=preds,
                )
        
                close_price_preds = inverse_transformed_preds
                normalized_close_price_preds = current_dataset.normalize_price_predictions(preds=close_price_preds)

            rep_results = Save_Results_Functions.record_rep_results(
                rep_results=rep_results,
                rep=rep,
                
                expected_vals=np.squeeze(
                    expected_close_prices, 
                    axis=-1,
                ),
                normalized_expected_vals=np.squeeze(
                    expected_normalized_close_prices, 
                    axis=-1,
                ),
                predicted_vals=np.squeeze(
                    close_price_preds, 
                    axis=-1,
                ),
                normalized_predicted_vals=np.squeeze(
                    normalized_close_price_preds, 
                    axis=-1,
                ),
                
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

        if 'VMD' in selected_transformation:
            base_res_map = f"Modeling/Results/Experiment_Results/{selected_experiment}/{selected_model_setting}_{selected_model}/{selected_normalization_moment}_{selected_VMD_mode}_{selected_transformation}__{'_'.join(selected_input)}/{selected_experiment_name}/{selected_crypto}/{selected_data_frequency}/"
        else:
            base_res_map = f"Modeling/Results/Experiment_Results/{selected_experiment}/{selected_model_setting}_{selected_model}/{selected_normalization_moment}_{selected_transformation}__{'_'.join(selected_input)}/{selected_experiment_name}/{selected_crypto}/{selected_data_frequency}/"

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

    if VMD_optimization_mode or model_optimization_mode:
        return {
            'mean_val_nmse':np.mean(mean_val_nMSEs),
            'mean_val_nrmse':np.median(mean_val_nRMSEs),
            'mean_val_nmae':np.mean(mean_val_nMAEs),
            'mean_val_nmape':np.median(mean_val_nMAPEs),
            
            'mean_val_mse':np.mean(mean_val_MSEs),
            'mean_val_rmse':np.median(mean_val_RMSEs),
            'mean_val_mae':np.mean(mean_val_MAEs),
            'mean_val_mape':np.median(mean_val_MAPEs),
            'mean_val_r2':np.median(mean_val_R2s),
        } 
    else:
        return training_metrics
        