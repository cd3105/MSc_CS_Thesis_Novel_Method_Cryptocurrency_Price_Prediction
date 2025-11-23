import argparse
from Experiments.Experiment import run_experiment
from Experiments.Hyperparameter_Search import optuna_hyperparameter_tuning

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental Parameters")

    parser.add_argument(
        '--cc', 
        type=str, 
        default=['BTC'],
        nargs='+', 
        help="The Cryptocurrency of which the Close Price is Forecasted in the Chosen Experiment",
    )
    parser.add_argument(
        '--exp', 
        type=str, 
        default='Baseline', 
        choices=['Baseline', 'Extended', 'Full'], 
        help="The Chosen Experiment",
    )
    parser.add_argument(
        '--df', 
        type=str, 
        default='1_Day', 
        choices=['1_Day'], # '1_Hour', '2_Hour', '4_Hour', '6_Hour', '8_Hour', '12_Hour',
        help="The Frequency of the Cryptocurrency Data",
    )
    parser.add_argument(
        '--ps', 
        type=str, 
        default='Binance', 
        choices=['Binance', 'CoinMarketCap', 'Investing'], 
        help="The Chosen Price Source",
    )
    parser.add_argument(
        '--pc', 
        type=str, 
        default='USDT', 
        choices=['USDT', 'USD'], 
        help="The Chosen Price Currency",
    )
    parser.add_argument(
        '--mod', 
        type=str, 
        default='TCN', 
        choices=['LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'TCN', 'ATCN', 'MSRCNN_LSTM', 'MSRCNN_GRU'], 
        help="The Model utilized in executing the Chosen Experiment",
    )
    parser.add_argument(
        '--ip', 
        type=str, 
        default=['C'],
        nargs='+', 
        help="The Selected Input",
    )
    parser.add_argument(
        '--dt', 
        type=str, 
        default='VMD', 
        choices=['VMD', 'VMD_n_Res', 'VMD_n_CEEMDAN', 'VMD_n_CEEMDAN_n_Res', 'MVMD', 'MVMD_n_Res', 'Diff_C', 'Diff_C_OHL', 'Diff_OHLC', 'None'],
        help="The Selected Data Transformation",
    )
    parser.add_argument(
        '--VMD_mode', 
        type=str, 
        default='MAX', 
        choices=['MAX', 'WIN'],
        help="The Selected Mode of applying VMD",
    )
    parser.add_argument(
        '--nm', 
        type=str, 
        default='AWN', 
        choices=['BWN', 'AWN'],
        help="The Selected Normalization Moment",
    )
    parser.add_argument(
        '--ms', 
        type=str, 
        default='SI', 
        choices=['SI', 'MIC', 'MIS', 'MM'],
        help="The Selected Model Setting",
    )
    parser.add_argument(
        '--en',
        type=str,
        default='Default', 
        required=False,
        help="The Name of the Experiment"
    )
    parser.add_argument(
        '--VMD_ht', 
        action='store_true',
        help="Pass to tune VMD",
    )
    parser.add_argument(
        '--model_ht', 
        action='store_true',
        help="Pass to tune Model",
    )
    parser.add_argument(
        '--combined_ht', 
        action='store_true',
        help="Pass to perform Combined Hyperparameter Tuning",
    )

    args = parser.parse_args()

    if args.VMD_ht or args.model_ht:
        print(f"\nStarting Hyperparameter Search of Selected Experiment:")
    else:
        print(f"\nStarting Experiment:")

    print(f"\t- Crypto(s) to be Forecasted: {args.cc}")
    print(f"\t- Experiment Type: {args.exp}")
    print(f"\t- Data Frequency: {args.df}")
    print(f"\t- Price Source: {args.ps}")
    print(f"\t- Price Currency: {args.pc}")
    print(f"\t- Model to be Utilized: {args.mod}")
    print(f"\t- Experimental Setting:\n\t\t- Chosen Input: {args.ip}\n\t\t- Chosen Data Transformation: {args.dt}\n\t\t- Chosen VMD Mode: {args.VMD_mode}\n\t\t- Chosen Normalization Moment: {args.nm}\n\t\t- Chosen Model Setting: {args.ms}\n")

    optimal_vmd_params = {
        'BTC': {
            'K':6, # 10 # 13 # 8 # 15 # 6
            'alpha':250, # 3250 # 2750 # 1750 # 8000 # 250
            'tau':0.15,
            'window_size':30,
        },
        'ETH': {
            'K':14,
            'alpha':1250,
            'tau':.25,
            'window_size':30,
        },
        'LTC': {
            'K':6,
            'alpha':250,
            'tau':.3,
            'window_size':30,
        },
        'XMR': {
            'K':6,
            'alpha':250,
            'tau':.3,
            'window_size':30,
        },
        'XRP': {
            'K':10,
            'alpha':1000,
            'tau':.05,
            'window_size':30,
        },
    }
    
    optimal_training_params = {
        'n_epochs':500,
        'batch_size':64,
        'optimizer':'Adam',
        'learning_rate':1e-4,
        'weight_decay':1e-3,
        'momentum':0.9,
    }

    if args.VMD_ht or args.model_ht:
        if args.combined_ht:
            optuna_hyperparameter_tuning(
                selected_cryptos=args.cc,
                selected_experiment=args.exp,
                selected_data_frequency=args.df,
                selected_price_source=args.ps,
                selected_price_currency=args.pc,
                
                selected_model=args.mod,
                selected_input=args.ip,
                selected_transformation=args.dt,
                selected_VMD_mode=args.VMD_mode,
                selected_normalization_moment=args.nm,
                selected_model_setting=args.ms,
                selected_experiment_name=args.en,

                tune_model=args.model_ht,
                train_params=optimal_training_params,
                tune_VMD=args.VMD_ht,
                VMD_params=optimal_vmd_params,
            )

        else:
            for current_crypto in args.cc:
                optuna_hyperparameter_tuning(
                    selected_cryptos=[current_crypto],
                    selected_experiment=args.exp,
                    selected_data_frequency=args.df,
                    selected_price_source=args.ps,
                    selected_price_currency=args.pc,
                    
                    selected_model=args.mod,
                    selected_input=args.ip,
                    selected_transformation=args.dt,
                    selected_VMD_mode=args.VMD_mode,
                    selected_normalization_moment=args.nm,
                    selected_model_setting=args.ms,
                    selected_experiment_name=args.en,

                    tune_model=args.model_ht,
                    train_params=optimal_training_params,
                    tune_VMD=args.VMD_ht,
                    VMD_params=optimal_vmd_params,
                )
    else:
        for current_crypto in args.cc:
            run_experiment(
                selected_crypto=current_crypto,
                selected_experiment=args.exp,
                selected_data_frequency=args.df,
                selected_price_source=args.ps,
                selected_price_currency=args.pc,
                
                selected_model=args.mod,
                selected_input=args.ip,
                selected_transformation=args.dt,
                selected_VMD_mode=args.VMD_mode,
                selected_normalization_moment=args.nm,
                selected_model_setting=args.ms,
                selected_experiment_name=args.en,
                
                VMD_params=optimal_vmd_params[current_crypto],
                training_params=optimal_training_params,
                
                n_reps=3,
                save_results=True,
            )
