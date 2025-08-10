import pandas as pd
import numpy as np
import os

models = ["ARIMA", "GRU", "HYBRID", "KNN", "LSTM", "RF", "SVR", "TCN", "TFT"] 
cryptos = ['btc', 'eth', 'ltc', 'xmr', 'xrp']
normalized = [True, False]

for c in cryptos:
    for m in models:
        for n in normalized:
            metrics_dfs = []

            if n:
                current_base_metric_results_path = f"Original_Code/old_res_rerun/{m}/Metrics/Normalized/"
                save_path = f"Original_Code/old_res_rerun/{m}/Average_Metrics/Normalized/"
                average_metrics_file_name = f"average_metrics_{m}-{c}-close-w30-h0-12m_N"
                suffix = "_N"
            else:
                current_base_metric_results_path = f"Original_Code/old_res_rerun/{m}/Metrics/Regular/"
                save_path = f"Original_Code/old_res_rerun/{m}/Average_Metrics/Regular/"
                average_metrics_file_name = f"average_metrics_{m}-{c}-close-w30-h0-12m"
                suffix = ""

            for rep in range(1,4):
                current_file_path = f"{current_base_metric_results_path}metrics_{m}-{c}-close-w30-h0-12m-r{rep}{suffix}.csv"
                current_metrics_df = pd.read_csv(current_file_path, index_col=0)
                current_metrics_df = current_metrics_df.iloc[:-1]
                metrics_dfs.append(current_metrics_df)

            metrics_rep_array_stack = np.stack([df.values for df in metrics_dfs])
            avg_metrics_array = np.mean(metrics_rep_array_stack, axis=0)
            avg_metrics_df = pd.DataFrame(avg_metrics_array, columns=metrics_dfs[0].columns)
            
            mean_row = avg_metrics_df.mean(numeric_only=True)
            mean_row.name = 'Mean'
            avg_metrics_df_w_mean = pd.concat([avg_metrics_df, pd.DataFrame([mean_row])])

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            avg_metrics_df_w_mean.to_csv(f"{save_path}{average_metrics_file_name}.csv")
        