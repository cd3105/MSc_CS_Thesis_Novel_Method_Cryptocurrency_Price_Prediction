from models.dl.gru_new import GRU
from util import plot_training, save_results, dataset_binance_new, r2
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, explained_variance_score, mean_squared_log_error
import numpy as np
import pandas as pd


def main():
    h=0
    targets = ['close']
    cryptos =  ['btc',  'eth', 'ltc', 'xmr', 'xrp']
    retrain = [0, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30]
    outputs =[30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31]   
    scaling = ['minmax']
    tuned =  1
    window = 30

    for t in targets:
        for c in cryptos:
            for rep in range(1, 4):
                add_split_value = 0
                mse, rmse, mape, r2_score, mae = [], [], [], [], []
                n_mse, n_rmse, n_mape, n_r2_score, n_mae = [], [], [], [], []
                all_predictions, all_labels = [], []
                all_n_predictions, all_n_labels = [], []
                inference_time, train_time = [], []

                for index, r in enumerate(retrain):
                    output = outputs[index]
                    
                    experiment_name = f'GRU-{c}-{t}-w{str(window)}-h{str(h)}-{str(len(outputs))}m-r{rep}'
                    ds = dataset_binance_new.BinanceDataset(filename=f'crypto_task_{c}.csv',
                                                            input_window=window, 
                                                            output_window=1,
                                                            horizon=h, 
                                                            training_features=['close'],
                                                            target_name=['close'], 
                                                            train_split_factor=0.8)
                    df, diff_df = ds.differenced_dataset()
                    ds.df = diff_df
                    if index > 0:
                        add_split_value += r
                    
                    ds.add_split_value = add_split_value
                    
                    if tuned:
                        parameters = pd.read_csv(f"Original_Code/param/p_GRU-{t}-{c}-w{window}-h{h}.csv").iloc[0] 
                        p  = {'first_gru_dim': parameters['first_gru_dim'],
                              'gru_activation': parameters['gru_activation'],
                              'first_dense_dim':parameters['first_dense_dim'],
                              'first_dense_activation': parameters['first_dense_activation'],
                              'dense_kernel_init': parameters['dense_kernel_init'],
                              'batch_size': parameters['batch_size'],
                              'epochs':parameters['epochs'],
                              'patience': parameters['patience'],
                              'optimizer': parameters['optimizer'],
                              'lr': parameters['lr'],
                              'momentum':parameters['momentum'],
                              'decay': parameters['decay'],   
                             }
                    else:
                        p = {'first_gru_dim': 75,
                             'gru_activation': 'relu',
                             'first_dense_dim': 100,
                             'first_dense_activation': 'relu',
                             'dense_kernel_init': 'he_normal',
                             'batch_size': 256,
                             'epochs': 200,
                             'patience': 50,
                             'optimizer': 'adam',
                             'lr': 1E-3,
                             'momentum': 0.9,
                             'decay': 1E-3,
                            }
                        
                    model = GRU(experiment_name)
                    model.ds = ds 
                    ds.dataset_creation(df=True, 
                                        detrended= True)
                    ds.dataset_normalization(scaling)
                    ds.data_summary()
                    to_predict = ds.X_test[:output]
                    yhat, train_model = model.training(p, 
                                                       X_test=to_predict)                                                                 
                    preds = np.array(yhat).reshape(-1, 1)
                    np_preds = ds.inverse_transform_predictions(preds=preds)
                    inversed_preds = ds.inverse_differenced_dataset(diff_vals=np_preds, 
                                                                    df=df, 
                                                                    l=(len(ds.y_test_array)))
                    ds.df = df
                    ds.dataset_creation(df=True)
                    labels = ds.y_test_array[(h):(len(inversed_preds)+h)].reshape(-1, 1)
                    ds.add_split_value = 0
                    ds.df = df
                    ds.dataset_creation(df=True)
                    ds.dataset_normalization(scaling)
                    n_preds = ds.scale_predictions(preds=inversed_preds)                               
                    n_labels = ds.scale_predictions(preds=labels)

                    n_mse.append(mean_squared_error(n_labels, 
                                                    n_preds))
                    n_rmse.append(np.sqrt(mean_squared_error(n_labels, 
                                                             n_preds)))
                    n_mae.append(mean_absolute_error(n_labels, 
                                                     n_preds))
                    n_mape.append(mean_absolute_percentage_error(n_labels, 
                                                                 n_preds))
                    n_r2_score.append(r2.r_squared(n_labels, 
                                                   n_preds))
                    
                    mse.append(mean_squared_error(labels, 
                                                  inversed_preds))
                    rmse.append(np.sqrt(mean_squared_error(labels, 
                                                           inversed_preds)))
                    mae.append(mean_absolute_error(labels, 
                                                   inversed_preds))
                    mape.append(mean_absolute_percentage_error(labels, 
                                                               inversed_preds))
                    r2_score.append(r2.r_squared(labels, 
                                                 inversed_preds))
                    
                    n_experiment_name = f'{experiment_name}_N'

                    inference_time.append(model.inference_time)
                    train_time.append(model.train_time)
                    all_predictions.extend(inversed_preds)
                    all_labels.extend(labels)
                    all_n_predictions.extend(n_preds)
                    all_n_labels.extend(n_labels)
                if not tuned:
                    save_results.save_params_csv(model.p, 
                                                 model.name)      
                
                save_results.save_output_csv(preds=all_predictions, 
                                             labels=all_labels, 
                                             feature=t, 
                                             filename=experiment_name, 
                                             res_map='new_res',
                                             model_type='GRU',
                                             bivariate=len(ds.target_name) > 1)

                save_results.save_output_csv(preds=all_n_predictions, 
                                             labels=all_n_labels, 
                                             feature=t, 
                                             filename=n_experiment_name, 
                                             res_map='new_res',
                                             model_type='GRU',
                                             normalized=True,
                                             bivariate=len(ds.target_name) > 1)
                
                save_results.save_metrics_csv(mses=mse, 
                                              maes=mae, 
                                              rmses=rmse, 
                                              mapes=mape, 
                                              r2=r2_score,
                                              filename=experiment_name, 
                                              res_map='new_res',
                                              model_type='GRU',)

                save_results.save_metrics_csv(mses=n_mse, 
                                              maes=n_mae, 
                                              rmses=n_rmse, 
                                              mapes=n_mape, 
                                              r2=n_r2_score,
                                              filename=n_experiment_name, 
                                              res_map='new_res',
                                              normalized=True,
                                              model_type='GRU',)
                
                save_results.save_timing(times=inference_time, 
                                         filename=f'{experiment_name}-inf_time',
                                         res_map='new_res',
                                         model_type='GRU',)
                
                save_results.save_timing(times=train_time, 
                                         filename=f'{experiment_name}-train_time',
                                         res_map='new_res',
                                         model_type='GRU',)              


if __name__ == "__main__":
    main()

