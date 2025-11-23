from Data_Util.Crypto_Dataset import Crypto_Dataset

def crypto_dataset_loader(
        crypto,
        experiment,
        data_frequency,
        price_source,
        price_currency,
        
        train_test_split_date,
        next_train_test_split_date,

        input_window, 
        output_window, 
        stride,
        horizon,

        transformation,
        normalization_moment,

        VMD_params,
        VMD_mode,
):
    if VMD_mode == 'Win':
        windowed_VMD = True
    else:
        windowed_VMD = False
    
    if normalization_moment == 'BWN':
        normalize_before_windowing = True
    else:
        normalize_before_windowing = False

    if transformation in ['VMD', 'VMD_n_Res', 'VMD_n_CEEMDAN', 'VMD_n_CEEMDAN_n_Res', 'MVMD', 'MVMD_n_Res']:
        VMD_decomposition = True
        MVMD_decomposition = False
        input_VMD_Residuals = False
        CEEMDAN_Residuals_decomposition = False
        input_CEEMDAN_Residuals = False

        differencing = False
        target_only_differencing = False

        if transformation in ['MVMD', 'MVMD_n_Res']:
            MVMD_decomposition = True
                
        if transformation in ['VMD_n_Res', 'VMD_n_CEEMDAN', 'VMD_n_CEEMDAN_n_Res', 'MVMD_n_Res']:
            input_VMD_Residuals = True

            if transformation in ['VMD_n_CEEMDAN', 'VMD_n_CEEMDAN_n_Res']:
                CEEMDAN_Residuals_decomposition = True
                
                if transformation == 'VMD_n_CEEMDAN_n_Res':
                    input_CEEMDAN_Residuals = True
    else:
        differencing = False
        target_only_differencing = False

        VMD_decomposition = False
        MVMD_decomposition = False
        input_VMD_Residuals = False
        CEEMDAN_Residuals_decomposition = False
        input_CEEMDAN_Residuals = False
                    
        if transformation in ['Diff_C', 'Diff_OHLC', 'Diff_C_OHL']:
            differencing = True

            if transformation == 'Diff_C_OHL':
                target_only_differencing = True
            
    return Crypto_Dataset(
        crypto=crypto,
        experiment=experiment,
        freq=data_frequency,
        source=price_source,
        currency=price_currency,
        
        train_test_split_date=train_test_split_date,
        next_train_test_split_date=next_train_test_split_date,

        input_window=VMD_params['window_size'] if windowed_VMD else input_window,
        output_window=output_window,
        stride=stride,
        horizon=horizon,
        normalize_before_windowing=normalize_before_windowing,
        
        apply_VMD=VMD_decomposition,
        apply_MVMD=MVMD_decomposition,
        VMD_n_IMFs=VMD_params['K'],
        VMD_alpha=VMD_params['alpha'],
        VMD_tau=VMD_params['tau'],
        include_VMD_res=input_VMD_Residuals,
        apply_res_CEEMDAN=CEEMDAN_Residuals_decomposition,
        include_CEEMDAN_res=input_CEEMDAN_Residuals,
        windowed_VMD=windowed_VMD,
        
        apply_differencing=differencing,
        target_only_differencing=target_only_differencing,
        differencing_interval=1,
    )
