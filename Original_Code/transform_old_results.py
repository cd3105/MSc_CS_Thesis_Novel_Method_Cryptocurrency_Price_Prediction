import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from util import r2, dataset_binance


models = ["ARIMA", "GRU", "HYBRID", "KNN", "LSTM", "RF", "SVR", "TCN", "TFT"] # "HYBRID"
cryptos = ['btc', 'eth', 'ltc', 'xmr', 'xrp']
normalized = [True, False]

for m in models:
    for c in cryptos:
        current_original_metrics_path = f"Original_Code/old_res/metrics_{m}-close-{c}-w30-h0-windowed-12m-0_N.csv"
        new_original_metrics_path = f"Original_Code/old_res_transformed/{m}/Normalized/Original_Metrics/metrics_{m.lower()}-close-{c}-w30-h0-windowed-12m-0_N.csv"

        current_original_metrics_df = pd.read_csv(current_original_metrics_path, index_col=0).iloc[:12]
        current_mean_row = current_original_metrics_df.mean(numeric_only=True)
        current_mean_row.name = 'Mean'
        pd.concat([current_original_metrics_df, pd.DataFrame([current_mean_row])]).to_csv(new_original_metrics_path)


month_division_365 = [30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31]
month_division_360 = [25, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31]

for m in models:
    for c in cryptos:
        for n in normalized:
            metrics_dfs = []

            if n:
                suffix = '_N'
                map = 'Normalized'
            else:
                suffix = ''
                map = 'Regular'

            current_predictions_path = f"Original_Code/old_res/output_preds_{m}-close-{c}-w30-h0-12m{suffix}.csv"
            current_actual_values_path = f"Original_Code/old_res/output_labels_{m}-close-{c}-w30-h0-12m{suffix}.csv"
            current_predictions_df = pd.read_csv(current_predictions_path, index_col=0)
            current_actual_values_df = pd.read_csv(current_actual_values_path, index_col=0)
            repetition_count = min(len(current_predictions_df), len(current_actual_values_df))

            for r in range(repetition_count):
                mses, rmses, maes, mapes, r2_scores = [], [], [], [], []
                current_predictions = current_predictions_df.iloc[r].tolist()
                current_actual_values = current_actual_values_df.iloc[r].tolist()

                if (len(current_predictions) == 365) and (len(current_actual_values) == 365):
                    current_month_lengths = month_division_365
                    current_l = 365
                else:
                    current_month_lengths = month_division_360

                    if (len(current_predictions) == 365):
                        current_predictions = current_predictions[(365 - 360):]

                    if (len(current_actual_values) == 365):
                        current_actual_values = current_actual_values[(365 - 360):]

                current_start_idx = 0 
                current_end_idx = 0
                
                for ml in current_month_lengths:
                    current_end_idx += ml

                    mses.append(mean_squared_error(current_actual_values[current_start_idx:current_end_idx], 
                                                   current_predictions[current_start_idx:current_end_idx])) 
                    rmses.append(np.sqrt(mean_squared_error(current_actual_values[current_start_idx:current_end_idx], 
                                                            current_predictions[current_start_idx:current_end_idx]))) 
                    maes.append(mean_absolute_error(current_actual_values[current_start_idx:current_end_idx], 
                                                    current_predictions[current_start_idx:current_end_idx])) 
                    mapes.append(mean_absolute_percentage_error(current_actual_values[current_start_idx:current_end_idx], 
                                                                current_predictions[current_start_idx:current_end_idx])) 
                    r2_scores.append(r2.r_squared(current_actual_values[current_start_idx:current_end_idx], 
                                                  current_predictions[current_start_idx:current_end_idx])) 
                    
                    current_start_idx += ml
                
                current_metrics_df = pd.DataFrame({"MSE": mses,
                                                   "RMSE": rmses,
                                                   "MAE": maes,
                                                   "MAPE": mapes,
                                                   "R2": r2_scores})
                metrics_dfs.append(current_metrics_df)
                current_mean_row = current_metrics_df.mean(numeric_only=True)
                current_mean_row.name = 'Mean'
                pd.concat([current_metrics_df, pd.DataFrame([current_mean_row])]).to_csv(f"Original_Code/old_res_transformed/{m}/{map}/Self_Calculated_Metrics/metrics_{m}-close-{c}-w30-h0-windowed-12m-r{r+1}{suffix}.csv")
            
            metrics_rep_array_stack = np.stack([df.values for df in metrics_dfs])
            avg_metrics_array = np.mean(metrics_rep_array_stack, axis=0)
            avg_metrics_df = pd.DataFrame(avg_metrics_array, columns=metrics_dfs[0].columns)
            
            mean_row = avg_metrics_df.mean(numeric_only=True)
            mean_row.name = 'Mean'
            pd.concat([avg_metrics_df, pd.DataFrame([mean_row])]).to_csv(f"Original_Code/old_res_transformed/{m}/{map}/Self_Calculated_Metrics/Average_Metrics/average_metrics_{m}-close-{c}-w30-h0-windowed-12m{suffix}.csv")
            pd.concat([avg_metrics_df, pd.DataFrame([mean_row])]).to_csv(f"Original_Code/old_res_transformed/{m}/{map}/Self_Calculated_Metrics/Average_Metrics/average_metrics_{m}-close-{c}-w30-h0-windowed-12m{suffix}.tex")


alt_month_division_365 = [30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31]
alt_month_division_360 = [30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 26]

for m in models:
    for c in cryptos:
        for n in normalized:
            metrics_dfs = []

            if n:
                suffix = '_N'
                map = 'Normalized'
            else:
                suffix = ''
                map = 'Regular'

            current_predictions_path = f"Original_Code/old_res/output_preds_{m}-close-{c}-w30-h0-12m{suffix}.csv"
            current_actual_values_path = f"Original_Code/old_res/output_labels_{m}-close-{c}-w30-h0-12m{suffix}.csv"
            current_predictions_df = pd.read_csv(current_predictions_path, index_col=0)
            current_actual_values_df = pd.read_csv(current_actual_values_path, index_col=0)
            repetition_count = min(len(current_predictions_df), len(current_actual_values_df))

            for r in range(repetition_count):
                mses, rmses, maes, mapes, r2_scores = [], [], [], [], []
                current_predictions = current_predictions_df.iloc[r].tolist()
                current_actual_values = current_actual_values_df.iloc[r].tolist()

                if (len(current_predictions) == 365) and (len(current_actual_values) == 365):
                    current_month_lengths = alt_month_division_365
                    current_l = 365
                else:
                    current_month_lengths = alt_month_division_360

                    if (len(current_predictions) == 365):
                        current_predictions = current_predictions[:360]

                    if (len(current_actual_values) == 365):
                        current_actual_values = current_actual_values[:360]

                current_start_idx = 0 
                current_end_idx = 0
                
                for ml in current_month_lengths:
                    current_end_idx += ml

                    mses.append(mean_squared_error(current_actual_values[current_start_idx:current_end_idx], 
                                                   current_predictions[current_start_idx:current_end_idx])) 
                    rmses.append(np.sqrt(mean_squared_error(current_actual_values[current_start_idx:current_end_idx], 
                                                            current_predictions[current_start_idx:current_end_idx]))) 
                    maes.append(mean_absolute_error(current_actual_values[current_start_idx:current_end_idx], 
                                                    current_predictions[current_start_idx:current_end_idx])) 
                    mapes.append(mean_absolute_percentage_error(current_actual_values[current_start_idx:current_end_idx], 
                                                                current_predictions[current_start_idx:current_end_idx])) 
                    r2_scores.append(r2.r_squared(current_actual_values[current_start_idx:current_end_idx], 
                                                  current_predictions[current_start_idx:current_end_idx])) 
                    
                    current_start_idx += ml
                
                current_metrics_df = pd.DataFrame({"MSE": mses,
                                                   "RMSE": rmses,
                                                   "MAE": maes,
                                                   "MAPE": mapes,
                                                   "R2": r2_scores})
                metrics_dfs.append(current_metrics_df)
                current_mean_row = current_metrics_df.mean(numeric_only=True)
                current_mean_row.name = 'Mean'
                pd.concat([current_metrics_df, pd.DataFrame([current_mean_row])]).to_csv(f"Original_Code/old_res_transformed/{m}/{map}/Alt_Self_Calculated_Metrics/metrics_{m}-close-{c}-w30-h0-windowed-12m-r{r+1}{suffix}.csv")
            
            metrics_rep_array_stack = np.stack([df.values for df in metrics_dfs])
            avg_metrics_array = np.mean(metrics_rep_array_stack, axis=0)
            avg_metrics_df = pd.DataFrame(avg_metrics_array, columns=metrics_dfs[0].columns)
            
            mean_row = avg_metrics_df.mean(numeric_only=True)
            mean_row.name = 'Mean'
            pd.concat([avg_metrics_df, pd.DataFrame([mean_row])]).to_csv(f"Original_Code/old_res_transformed/{m}/{map}/Alt_Self_Calculated_Metrics/Average_Metrics/average_metrics_{m}-close-{c}-w30-h0-windowed-12m{suffix}.csv")
            pd.concat([avg_metrics_df, pd.DataFrame([mean_row])]).to_csv(f"Original_Code/old_res_transformed/{m}/{map}/Alt_Self_Calculated_Metrics/Average_Metrics/average_metrics_{m}-close-{c}-w30-h0-windowed-12m{suffix}.tex")


for m in models:
    for c in cryptos:
        metrics_dfs = []
        current_normalized_predictions_path = f"Original_Code/old_res/output_preds_{m}-close-{c}-w30-h0-12m_N.csv"
        current_normalized_actual_values_path = f"Original_Code/old_res/output_labels_{m}-close-{c}-w30-h0-12m_N.csv"
        current_normalized_predictions_df = pd.read_csv(current_normalized_predictions_path, index_col=0)
        current_normalized_actual_values_df = pd.read_csv(current_normalized_actual_values_path, index_col=0)
        repetition_count = min(len(current_normalized_predictions_df), len(current_normalized_actual_values_df))

        for r in range(repetition_count):
            mses, rmses, maes, mapes, r2_scores = [], [], [], [], []
                
            current_normalized_predictions = current_normalized_predictions_df.iloc[r].to_list()
            current_normalized_actual_values = current_normalized_actual_values_df.iloc[r].to_list()

            ds = dataset_binance.BinanceDataset(filename=f'crypto_task_{c}.csv', 
                                            input_window=30, 
                                            output_window=1,
                                            horizon=0, 
                                            training_features=['close'],
                                            target_name=['close'], 
                                            train_split_factor=0.8) 
        
            df, diff_df = ds.differenced_dataset()
            ds.df = df
            ds.dataset_creation(df=True, days_365=False)
            current_actual_values = np.array(ds.y_test_array).reshape(-1, 1)
            ds.dataset_normalization()
            current_predictions = ds.inverse_transform_predictions(preds=np.array(current_normalized_predictions).reshape(-1, 1))

            if (len(current_predictions) == 365) and (len(current_actual_values) == 365):
                current_month_lengths = month_division_365
                current_l = 365
            else:
                current_month_lengths = month_division_360

                if (len(current_predictions) == 365):
                    current_predictions = current_predictions[(365 - 360):]

                if (len(current_actual_values) == 365):
                    current_actual_values = current_actual_values[(365 - 360):]

            current_start_idx = 0 
            current_end_idx = 0
                
            for ml in current_month_lengths:
                current_end_idx += ml

                mses.append(mean_squared_error(current_actual_values[current_start_idx:current_end_idx], 
                                               current_predictions[current_start_idx:current_end_idx])) 
                rmses.append(np.sqrt(mean_squared_error(current_actual_values[current_start_idx:current_end_idx], 
                                                        current_predictions[current_start_idx:current_end_idx]))) 
                maes.append(mean_absolute_error(current_actual_values[current_start_idx:current_end_idx], 
                                                current_predictions[current_start_idx:current_end_idx])) 
                mapes.append(mean_absolute_percentage_error(current_actual_values[current_start_idx:current_end_idx], 
                                                            current_predictions[current_start_idx:current_end_idx])) 
                r2_scores.append(r2.r_squared(current_actual_values[current_start_idx:current_end_idx], 
                                              current_predictions[current_start_idx:current_end_idx])) 
                    
                current_start_idx += ml

                current_metrics_df = pd.DataFrame({"MSE": mses,
                                                   "RMSE": rmses,
                                                   "MAE": maes,
                                                   "MAPE": mapes,
                                                   "R2": r2_scores})
            metrics_dfs.append(current_metrics_df)
            current_mean_row = current_metrics_df.mean(numeric_only=True)
            current_mean_row.name = 'Mean'
            pd.concat([current_metrics_df, pd.DataFrame([current_mean_row])]).to_csv(f"Original_Code/old_res_transformed/{m}/Regular/Self_Calculated_Metrics_Transform/metrics_{m}-close-{c}-w30-h0-windowed-12m-r{r+1}.csv")
            
        metrics_rep_array_stack = np.stack([df.values for df in metrics_dfs])
        avg_metrics_array = np.mean(metrics_rep_array_stack, axis=0)
        avg_metrics_df = pd.DataFrame(avg_metrics_array, columns=metrics_dfs[0].columns)
            
        mean_row = avg_metrics_df.mean(numeric_only=True)
        mean_row.name = 'Mean'
        pd.concat([avg_metrics_df, pd.DataFrame([mean_row])]).to_csv(f"Original_Code/old_res_transformed/{m}/Regular/Self_Calculated_Metrics_Transform/Average_Metrics/average_metrics_{m}-close-{c}-w30-h0-windowed-12m.csv")
        pd.concat([avg_metrics_df, pd.DataFrame([mean_row])]).to_csv(f"Original_Code/old_res_transformed/{m}/Regular/Self_Calculated_Metrics_Transform/Average_Metrics/average_metrics_{m}-close-{c}-w30-h0-windowed-12m.tex")


for m in models:
    for c in cryptos:
        metrics_dfs = []
        current_normalized_predictions_path = f"Original_Code/old_res/output_preds_{m}-close-{c}-w30-h0-12m_N.csv"
        current_normalized_actual_values_path = f"Original_Code/old_res/output_labels_{m}-close-{c}-w30-h0-12m_N.csv"
        current_normalized_predictions_df = pd.read_csv(current_normalized_predictions_path, index_col=0)
        current_normalized_actual_values_df = pd.read_csv(current_normalized_actual_values_path, index_col=0)
        repetition_count = min(len(current_normalized_predictions_df), len(current_normalized_actual_values_df))

        for r in range(repetition_count):
            mses, rmses, maes, mapes, r2_scores = [], [], [], [], []
                
            current_normalized_predictions = current_normalized_predictions_df.iloc[r].to_list()
            current_normalized_actual_values = current_normalized_actual_values_df.iloc[r].to_list()

            ds = dataset_binance.BinanceDataset(filename=f'crypto_task_{c}.csv', 
                                                input_window=30, 
                                                output_window=1,
                                                horizon=0, 
                                                training_features=['close'],
                                                target_name=['close'], 
                                                train_split_factor=0.8) 
        
            df, diff_df = ds.differenced_dataset()
            ds.df = df
            ds.dataset_creation(df=True, days_365=False)
            current_actual_values = np.array(ds.y_test_array).reshape(-1, 1)
            ds.dataset_normalization()
            current_predictions = ds.inverse_transform_predictions(preds=np.array(current_normalized_predictions).reshape(-1, 1))

            if (len(current_predictions) == 365) and (len(current_actual_values) == 365):
                current_month_lengths = alt_month_division_365
                current_l = 365
            else:
                current_month_lengths = alt_month_division_360

                if (len(current_predictions) == 365):
                    current_predictions = current_predictions[:360]

                if (len(current_actual_values) == 365):
                    current_actual_values = current_actual_values[:360]

            current_start_idx = 0 
            current_end_idx = 0
                
            for ml in current_month_lengths:
                current_end_idx += ml

                mses.append(mean_squared_error(current_actual_values[current_start_idx:current_end_idx], 
                                               current_predictions[current_start_idx:current_end_idx])) 
                rmses.append(np.sqrt(mean_squared_error(current_actual_values[current_start_idx:current_end_idx], 
                                                        current_predictions[current_start_idx:current_end_idx]))) 
                maes.append(mean_absolute_error(current_actual_values[current_start_idx:current_end_idx], 
                                                current_predictions[current_start_idx:current_end_idx])) 
                mapes.append(mean_absolute_percentage_error(current_actual_values[current_start_idx:current_end_idx], 
                                                            current_predictions[current_start_idx:current_end_idx])) 
                r2_scores.append(r2.r_squared(current_actual_values[current_start_idx:current_end_idx], 
                                              current_predictions[current_start_idx:current_end_idx])) 
                    
                current_start_idx += ml

                current_metrics_df = pd.DataFrame({"MSE": mses,
                                                   "RMSE": rmses,
                                                   "MAE": maes,
                                                   "MAPE": mapes,
                                                   "R2": r2_scores})
            metrics_dfs.append(current_metrics_df)
            current_mean_row = current_metrics_df.mean(numeric_only=True)
            current_mean_row.name = 'Mean'
            pd.concat([current_metrics_df, pd.DataFrame([current_mean_row])]).to_csv(f"Original_Code/old_res_transformed/{m}/Regular/Alt_Self_Calculated_Metrics_Transform/metrics_{m}-close-{c}-w30-h0-windowed-12m-r{r+1}.csv")
            
        metrics_rep_array_stack = np.stack([df.values for df in metrics_dfs])
        avg_metrics_array = np.mean(metrics_rep_array_stack, axis=0)
        avg_metrics_df = pd.DataFrame(avg_metrics_array, columns=metrics_dfs[0].columns)
            
        mean_row = avg_metrics_df.mean(numeric_only=True)
        mean_row.name = 'Mean'
        pd.concat([avg_metrics_df, pd.DataFrame([mean_row])]).to_csv(f"Original_Code/old_res_transformed/{m}/Regular/Alt_Self_Calculated_Metrics_Transform/Average_Metrics/average_metrics_{m}-close-{c}-w30-h0-windowed-12m.csv")
        pd.concat([avg_metrics_df, pd.DataFrame([mean_row])]).to_csv(f"Original_Code/old_res_transformed/{m}/Regular/Alt_Self_Calculated_Metrics_Transform/Average_Metrics/average_metrics_{m}-close-{c}-w30-h0-windowed-12m.tex")
