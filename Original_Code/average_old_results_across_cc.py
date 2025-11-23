import pandas as pd
import numpy as np
import os

models = ["ARIMA", "GRU", "HYBRID", "KNN", "LSTM", "RF", "SVR", "TCN", "TFT"] 
cryptos = ['btc', 'eth', 'ltc', 'xmr', 'xrp']
normalized = [True, False]


for n in normalized:
    model_average_results_dfs = []

    for m in models:
        avg_rows = []
        
        for c in cryptos:
            if n:
                current_average_metric_results_base_path = f"Original_Code/old_res_rerun/{m}/Average_Metrics/Normalized/"
                average_metrics_file_name = f"average_metrics_{m}-{c}-close-w30-h0-12m_N"
            else:
                current_average_metric_results_base_path = f"Original_Code/old_res_rerun/{m}/Average_Metrics/Regular/"
                average_metrics_file_name = f"average_metrics_{m}-{c}-close-w30-h0-12m"
            
            current_average_results_per_cc_df = pd.read_csv(
                filepath_or_buffer=f'{current_average_metric_results_base_path}{average_metrics_file_name}.csv',
                index_col=0,
            )
            avg_rows.append(current_average_results_per_cc_df.iloc[-1:].values)

        current_model_average_results_df = pd.DataFrame(
            np.mean(
                avg_rows, 
                axis=0
            ), 
            columns=current_average_results_per_cc_df.columns,
        )
        current_model_average_results_df.index = [m]
        model_average_results_dfs.append(current_model_average_results_df)
    
    current_average_results_df = pd.concat(model_average_results_dfs)

    model_average_results_path = f'Original_Code/old_res_rerun/_Average_Results/'
    
    if n:
        model_average_results_fn = 'Normalized_Average_Metrics.csv'
    else:
        model_average_results_fn = 'Average_Metrics.csv'
    
    if not os.path.exists(model_average_results_path):
        os.makedirs(model_average_results_path)

    current_average_results_df.to_csv(f'{model_average_results_path}{model_average_results_fn}')
