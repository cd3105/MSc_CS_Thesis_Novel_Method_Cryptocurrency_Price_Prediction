from models.dl.gru import GRU
from util import save_results, dataset_binance, r2
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd


def main():
    h=0 # Horizon
    targets = ['close'] # Target Variable
    cryptos =  ['btc', 'eth', 'ltc', 'xmr', 'xrp'] # Selected Cryptos
    retrain = [0, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30] # Values corresponding to days per Month to adjust Train / Test Set to incorporate evaluated months into Training Set
    outputs =[30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31] # Portion of Test Set which is used for evaluation
    scaling = ['minmax'] # Normalization
    tuned =  1 # Designation of whether the model has been tuned or not with 0 indicating the latter
    window = 30 # Previous Timesteps to consider for predicting the next one
    for t in targets: # Loop of Target Variables
        for c in cryptos: # Loop over Cryptos
            for rep in range(1, 4): # Loop over Number of Repetitions
                add_split_value = 0 # Variable for adjusting Train / Test Set based on evaluated months
                mse, rmse, mape, r2_score, mae = [], [], [], [], [] # Evaluation Metrics
                n_mse, n_rmse, n_mape, n_r2_score, n_mae = [], [], [], [], [] # Normalized Evaluation Metrics
                all_predictions, all_labels = [], [] # Actual Values and Precise Predictions
                all_n_predictions, all_n_labels = [], [] # Normalized Actual Values and Precise Predictions
                inference_time, train_time = [], [] # Execution Time
                
                for index, r in enumerate(retrain): # Loop over various values to enlarge Training and shrink Test Set with
                    output = outputs[index] # Current Portion of Test Set to use for Evaluation

                    experiment_name = f'gru-{c}-{t}-w{str(window)}-h{str(h)}-{str(len(outputs))}m-r{rep}'
                    ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv',
                                                        input_window=window,
                                                        output_window=1,
                                                        horizon=h,
                                                        training_features=['close'],
                                                        target_name=['close'],
                                                        train_split_factor=0.8) # Initialization DataSet Object
                    df, diff_df = ds.differenced_dataset() # Retrieve current DataFrame and DataFrame containing Differenced Close Price after subtracting Future Values with their Previous Values
                    ds.df = diff_df # Sets DataFrame variable of Initialized DataSet object to the Differenced DataFrame
                    if index > 0:
                        add_split_value += r # Increase point at which Training & Test Set Split occurs leading to a growing Training Set in accordance with specific Months
                    
                    ds.add_split_value = add_split_value # Set Point Increase pertaining to Training & Test Set Split of DataSet object
                    
                    if tuned: # Set Parameters of Model after Tuning
                        parameters = pd.read_csv(f"Original_Code/param/p_GRU-{t}-{c}-w{window}-h{h}.csv").iloc[0]
                        p  = {'first_gru_dim': int(parameters['first_gru_dim']),
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
                    else: # If Model hasn't been Tuned, then set these Parameters
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

                    model = GRU(experiment_name) # Initialize GRU Object
                    model.ds = ds # Set DataSet used for Training to Initialized DataSet Object
                    ds.dataset_creation(df=True,
                                        detrended= True) # Create Training and Test Sets in varying formats with Differenced DataFrame
                    ds.dataset_normalization(scaling) # Normalize the various formats made from Differenced DataFrame
                    ds.data_summary() # Prints Shapes of Windowed Training and Test sets
                    to_predict = ds.X_test[:output] # Restrict Test Set to Current Month
                    yhat, train_model = model.training(p,
                                                    X_test=to_predict) # Train Model and Retrieve both the Trained Model and the corresponding made Predictions
                    preds = np.array(yhat).reshape(-1, 1) # Reshape Predictions Array
                    np_preds = ds.inverse_transform_predictions(preds = preds) # Inverse Normalization
                    inversed_preds = ds.inverse_differenced_dataset(diff_vals= np_preds,
                                                                    df=df,
                                                                    l = (len(ds.y_test_array))) # Inverse Differencing
                    ds.df = df # Sets DataFrame variable of Initialized DataSet object to the Original DataFrame
                    ds.dataset_creation(df=True) # Create Training and Test Sets in varying formats with Original DataFrame
                    labels = ds.y_test_array[(h):(len(inversed_preds)+h)].reshape(-1, 1) # Retrieve Actual Values corresponding to the Predicted Values
                    ds.add_split_value = 0 # Set Split Value to add in addition to 0
                    ds.df = df # Sets DataFrame variable of Initialized DataSet object to the Original DataFrame
                    ds.dataset_creation(df=True) # Create Training and Test Sets in varying formats with Original DataFrame
                    ds.dataset_normalization(scaling) # Normalize the various formats made from Original DataFrame
                    n_preds = ds.scale_predictions(preds=inversed_preds) # Normalize Predictions
                    n_labels = ds.scale_predictions(preds=labels) # Normalize Actual Values
                    
                    n_mse.append(mean_squared_error(n_labels, n_preds)) # Calculate MSE using Normalized Actual and Predicted Values
                    n_rmse.append(np.sqrt(mean_squared_error(n_labels, n_preds))) # Calculate RMSE using Normalized Actual and Predicted Values
                    n_mae.append(mean_absolute_error(n_labels, n_preds)) # Calculate MAE using Normalized Actual and Predicted Values
                    n_mape.append(mean_absolute_percentage_error(n_labels, n_preds)) # Calculate MAPE using Normalized Actual and Predicted Values
                    n_r2_score.append(r2.r_squared(n_labels, n_preds)) # Calculate R2 using Normalized Actual and Predicted Values

                    mse.append(mean_squared_error(labels, inversed_preds)) # Calculate MSE using Actual and Predicted Values
                    rmse.append(np.sqrt(mean_squared_error(labels, inversed_preds))) # Calculate RMSE using Actual and Predicted Values
                    mae.append(mean_absolute_error(labels, inversed_preds)) # Calculate MAE using Actual and Predicted Values
                    mape.append(mean_absolute_percentage_error(labels, inversed_preds)) # Calculate MAPE using Actual and Predicted Values
                    r2_score.append(r2.r_squared(labels, inversed_preds)) # Calculate R2 using Actual and Predicted Values

                    print("\nNormalized Metrics:\n")
                    print(f"- nMSE: {mean_squared_error(n_labels, n_preds)}") # Print MSE using Normalized Actual and Predicted Values
                    print(f"- nMAE: {mean_absolute_error(n_labels, n_preds)}") # Print MAE using Normalized Actual and Predicted Values
                    print(f"- nMAPE: {mean_absolute_percentage_error(n_labels, n_preds)}") # Print MAPE using Normalized Actual and Predicted Values
                    print(f"- nRMSE: {np.sqrt(mean_squared_error(n_labels, n_preds))}") # Print RMSE using Normalized Actual and Predicted Values
                    print(f"- nR2: {r2.r_squared(n_labels, n_preds)}") # Print R2 using Normalized Actual and Predicted Values

                    print("\nRegular Metrics:\n")
                    print(f"- MSE: {mean_squared_error(labels, inversed_preds)}") # Print MSE using Actual and Predicted Values
                    print(f"- MAE: {mean_absolute_error(labels, inversed_preds)}") # Print MAE using Actual and Predicted Values
                    print(f"- MAPE: {mean_absolute_percentage_error(labels, inversed_preds)}") # Print MAPE using Actual and Predicted Values
                    print(f"- RMSE: {np.sqrt(mean_squared_error(labels, inversed_preds))}") # Print RMSE using Actual and Predicted Values
                    print(f"- R2: {r2.r_squared(labels, inversed_preds)}") # Print R2 using Actual and Predicted Values

                    n_experiment_name = experiment_name + '_N' # Initialize Experiment Name for Normalized Values

                    inference_time.append(model.inference_time) # Save Testing Time to Array
                    train_time.append(model.train_time) # Save Training Time to Array
                    all_predictions.extend(inversed_preds) # Save Normalized Predictions to Array
                    all_labels.extend(labels) # Save Normalized Actual Values to Array
                    all_n_predictions.extend(n_preds) # Save Normalized Predictions to Array
                    all_n_labels.extend(n_labels) # Save Normalized Actual Values to Array
                if not tuned:
                    save_results.save_params_csv(model.p, model.name)      
                
                save_results.save_output_csv(preds=all_predictions,
                                            labels=all_labels,
                                            feature=t,
                                            filename=experiment_name,
                                            model_type="GRU",
                                            bivariate=len(ds.target_name) > 1) # Save Actual Values and Predicted Values
                      
                save_results.save_output_csv(preds=all_n_predictions,
                                            labels=all_n_labels,
                                            feature=t,
                                            filename=n_experiment_name,
                                            model_type="GRU",
                                            bivariate=len(ds.target_name) > 1) # Save Normalized Actual Values and Predicted Values
                
                save_results.save_metrics_csv(mses=mse,
                                            maes=mae,
                                            rmses=rmse,
                                            mapes=mape,
                                            filename=experiment_name,
                                            r2=r2_score,
                                            model_type="GRU") # Save Metrics
                
                save_results.save_metrics_csv(mses=n_mse,
                                            maes=n_mae,
                                            rmses=n_rmse,
                                            mapes=n_mape,
                                            filename=n_experiment_name,
                                            r2=n_r2_score,
                                            model_type="GRU") # Save Normalized Metrics

                save_results.save_timing(times=inference_time,
                                        filename=f'{experiment_name}-inf_time',
                                        model_type="GRU") # Save Testing Time
                
                save_results.save_timing(times=train_time,
                                        filename=f'{experiment_name}-train_time',
                                        model_type="GRU") # Save Training Time        
                                                    
if __name__ == "__main__":
    main()