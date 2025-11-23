import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


def record_rep_results(rep_results,
                       rep,
                       expected_vals,
                       normalized_expected_vals,
                       predicted_vals,
                       normalized_predicted_vals,
                       var_name):
    
    if var_name not in rep_results[rep]['nMSE'].keys():
        rep_results[rep]['nMSE'][var_name], rep_results[rep]['nRMSE'][var_name], rep_results[rep]['nMAE'][var_name], rep_results[rep]['nMAPE'][var_name] = [], [], [], []
        rep_results[rep]['MSE'][var_name], rep_results[rep]['RMSE'][var_name], rep_results[rep]['MAE'][var_name], rep_results[rep]['MAPE'][var_name], rep_results[rep]['R2'][var_name] = [], [], [], [], []
        rep_results[rep]['Expected'][var_name], rep_results[rep]['Predicted'][var_name], rep_results[rep]['Expected_Normalized'][var_name], rep_results[rep]['Predicted_Normalized'][var_name] = [], [], [], []

    rep_results[rep]['nMSE'][var_name].append(mean_squared_error(normalized_expected_vals,
                                                                 normalized_predicted_vals))
    rep_results[rep]['nRMSE'][var_name].append(np.sqrt(mean_squared_error(normalized_expected_vals, 
                                                                          normalized_predicted_vals)))
    rep_results[rep]['nMAE'][var_name].append(mean_absolute_error(normalized_expected_vals, 
                                                                  normalized_predicted_vals))
    rep_results[rep]['nMAPE'][var_name].append(mean_absolute_percentage_error(normalized_expected_vals, 
                                                                              normalized_predicted_vals))


    rep_results[rep]['MSE'][var_name].append(mean_squared_error(expected_vals, 
                                                                predicted_vals))
    rep_results[rep]['RMSE'][var_name].append(np.sqrt(mean_squared_error(expected_vals, 
                                                                         predicted_vals)))
    rep_results[rep]['MAE'][var_name].append(mean_absolute_error(expected_vals, 
                                                                 predicted_vals))
    rep_results[rep]['MAPE'][var_name].append(mean_absolute_percentage_error(expected_vals, 
                                                                             predicted_vals))
    rep_results[rep]['R2'][var_name].append(r2_score(expected_vals,
                                                     predicted_vals))
                            
                        
    rep_results[rep]['Expected'][var_name].extend(list(expected_vals.flatten()))
    rep_results[rep]['Predicted'][var_name].extend(list(predicted_vals.flatten()))
    rep_results[rep]['Expected_Normalized'][var_name].extend(list(normalized_expected_vals.flatten()))
    rep_results[rep]['Predicted_Normalized'][var_name].extend(list(normalized_predicted_vals.flatten()))

    return rep_results


def save_metrics(rep_results,
                 base_res_map,):
    if not os.path.exists(base_res_map):
        os.makedirs(base_res_map)
    
    for rep in rep_results.keys():
        res_map = f'{base_res_map}Repetition_{rep}/Metrics/'

        if not os.path.exists(res_map):
            os.makedirs(res_map)

        for t in rep_results[rep]['nMSE'].keys():
            metrics_df = pd.DataFrame({'nMSE':rep_results[rep]['nMSE'][t],
                                       'nRMSE':rep_results[rep]['nRMSE'][t],
                                       'nMAE':rep_results[rep]['nMAE'][t],
                                       'nMAPE':rep_results[rep]['nMAPE'][t],
                                       'MSE':rep_results[rep]['MSE'][t],
                                       'RMSE':rep_results[rep]['RMSE'][t],
                                       'MAE':rep_results[rep]['MAE'][t],
                                       'MAPE':rep_results[rep]['MAPE'][t],
                                       'R2':rep_results[rep]['R2'][t],})
    
            metrics_df.loc['Mean'] = metrics_df.mean(axis=0)
            metrics_df.to_csv(f"{res_map}{t}_Metrics.csv")


def save_outputs(rep_results,
                 base_res_map,):
    for rep in rep_results.keys():
        res_map = f'{base_res_map}Repetition_{rep}/Outputs/'

        if not os.path.exists(res_map):
            os.makedirs(res_map)

        for t in rep_results[rep]['Expected'].keys():
            outputs_df = pd.DataFrame({'Expected':rep_results[rep]['Expected'][t],
                                       'Predicted':rep_results[rep]['Predicted'][t],
                                       'Expected_Normalized':rep_results[rep]['Expected_Normalized'][t],
                                       'Predicted_Normalized':rep_results[rep]['Predicted_Normalized'][t],})
            outputs_df.to_csv(f"{res_map}{t}_Outputs.csv")
        

def save_average_metrics(rep_results,
                         base_res_map,):
    mean_map = f"{base_res_map}Mean/Metrics/"

    for t in rep_results[1]['Expected'].keys():
        metrics_dfs = []

        for rep in rep_results.keys():
            current_metrics_df = pd.read_csv(f"{base_res_map}/Repetition_{rep}/Metrics/{t}_Metrics.csv", 
                                             index_col=0)
            current_metrics_df = current_metrics_df.iloc[:-1]
            metrics_dfs.append(current_metrics_df)
        
        metrics_rep_array_stack = np.stack([df.values for df in metrics_dfs])
        avg_metrics_array = np.mean(metrics_rep_array_stack, 
                                    axis=0)
        avg_metrics_df = pd.DataFrame(avg_metrics_array, 
                                      columns=metrics_dfs[0].columns)
        
        mean_row = avg_metrics_df.mean(numeric_only=True)
        mean_row.name = 'Mean'
        avg_metrics_df_w_mean = pd.concat([avg_metrics_df, 
                                           pd.DataFrame([mean_row])])

        if not os.path.exists(mean_map):
            os.makedirs(mean_map)

        print(f'{t} Results:')
        print(avg_metrics_df_w_mean)

        avg_metrics_df_w_mean.to_csv(f"{mean_map}{t}_Metrics.csv")


def save_average_outputs(rep_results,
                         base_res_map,):
    mean_map = f"{base_res_map}Mean/Outputs/"

    for t in rep_results[1]['Expected'].keys():
        preds = []
        normalized_preds = []

        for rep in rep_results.keys():
            current_outputs_df = pd.read_csv(f"{base_res_map}/Repetition_{rep}/Outputs/{t}_Outputs.csv", 
                                             index_col=0)
            preds.append(np.array(list(current_outputs_df['Predicted'])))
            normalized_preds.append(np.array(list(current_outputs_df['Predicted_Normalized'])))

        preds = np.array(preds)
        normalized_preds = np.array(normalized_preds)

        avg_outputs_df = pd.DataFrame({'Expected':list(current_outputs_df['Expected']),
                                       'Predicted':list(np.mean(preds,
                                                                axis=0)),
                                       'Expected_Normalized':list(current_outputs_df['Expected_Normalized']),
                                       'Predicted_Normalized':list(np.mean(normalized_preds,
                                                                           axis=0)),})

        if not os.path.exists(mean_map):
            os.makedirs(mean_map)

        avg_outputs_df.to_csv(f"{mean_map}{t}_Outputs.csv")
