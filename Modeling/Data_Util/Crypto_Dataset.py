import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pysdkit import VMD, MVMD, CEEMDAN

class Crypto_Dataset:
    def __init__(
            self, 
            crypto,
            experiment,
            freq,
            source,
            currency,
            
            train_test_split_date,
            next_train_test_split_date,
            
            input_window, 
            output_window, 
            stride,
            horizon,
            normalize_before_windowing,

            apply_VMD,
            apply_MVMD,
            VMD_n_IMFs,
            VMD_alpha,
            VMD_tau,
            include_VMD_res,
            apply_res_CEEMDAN,
            include_CEEMDAN_res,
            windowed_VMD,

            apply_differencing,
            target_only_differencing,
            differencing_interval,
    ):
        self.selected_crypto = crypto
        self.selected_interval = freq
        self.selected_experiment = experiment
        self.selected_source = source
        self.currency = currency

        self.train_test_split_date = train_test_split_date
        self.next_train_test_split_date = next_train_test_split_date

        self.input_window = input_window
        self.stride = stride
        self.output_window = output_window
        self.horizon = horizon
        self.scaler_type = StandardScaler
        self.normalize_before_windowing = normalize_before_windowing

        self.apply_VMD = apply_VMD
        self.apply_MVMD = apply_MVMD
        
        self.VMD_n_IMFs = VMD_n_IMFs
        self.VMD_alpha = VMD_alpha
        self.VMD_tau = VMD_tau
        self.include_VMD_res = include_VMD_res
        self.apply_res_CEEMDAN = apply_res_CEEMDAN
        self.include_CEEMDAN_res = include_CEEMDAN_res
        self.windowed_VMD = windowed_VMD

        self.apply_differencing = apply_differencing
        self.target_only_differencing = target_only_differencing
        self.differencing_interval = differencing_interval

        self.price_data_df_fn = f'All_Crypto_Data/Adopted_Data/Processed/{experiment}/Price_Data/{source}/{crypto}/{freq}/{crypto}_Price_Data.csv'
        self.price_data_df = pd.read_csv(self.price_data_df_fn)
        self.price_data_df['TIMESTAMP'] = pd.to_datetime(self.price_data_df['TIMESTAMP'])
        self.price_data_df = self.price_data_df[self.price_data_df['TIMESTAMP'] < next_train_test_split_date]
        
        self.original_price_data_df = self.price_data_df.copy()
        
        self.close_price_data_X_train = []
        self.close_price_data_X_test = []
        self.close_price_scaler = None

        self.open_price_data_X_train = []
        self.open_price_data_X_test = []
        self.open_price_scaler = None

        self.high_price_data_X_train = []
        self.high_price_data_X_test = []
        self.high_price_scaler = None

        self.low_price_data_X_train = []
        self.low_price_data_X_test = []
        self.low_price_scaler = None

        self.y_train = []
        self.y_test = []
        self.y_windowed_scalers = {}

        self.y_train_close_price = []
        self.y_train_normalized_close_price = []
        self.y_test_close_price = []
        self.y_test_normalized_close_price = []
        self.y_close_price_scaler = {}
        self.y_windowed_close_price_scalers = {}

        self.close_price_data_X_names = []


    def summary(
            self, 
            tabs="",
    ):  
        print(f'{tabs}- Open Price Features:')
        print(f'{tabs}\t- Train Set Shape: {self.open_price_data_X_train.shape}')
        print(f'{tabs}\t- Testing Set Shape: {self.open_price_data_X_test.shape}')

        print(f'{tabs}- High Price Features:')
        print(f'{tabs}\t- Train Set Shape: {self.high_price_data_X_train.shape}')
        print(f'{tabs}\t- Testing Set Shape: {self.high_price_data_X_test.shape}')

        print(f'{tabs}- Low Price Features:')
        print(f'{tabs}\t- Train Set Shape: {self.low_price_data_X_train.shape}')
        print(f'{tabs}\t- Testing Set Shape: {self.low_price_data_X_test.shape}')

        print(f'{tabs}- Close Price Features:')
        print(f'{tabs}\t- Train Set Shape: {self.close_price_data_X_train.shape}')
        print(f'{tabs}\t- Testing Set Shape: {self.close_price_data_X_test.shape}')

        print(f'{tabs}- Outputs:')
        print(f'{tabs}\t- Train Set Shape: {self.y_train.shape}')
        print(f'{tabs}\t- Testing Set Shape: {self.y_test.shape}\n')


    def apply_price_differencing(self):
        if self.target_only_differencing:
            columns_to_difference = self.target_columns
            columns_to_ignore = [c for c in self.price_data_df.columns[1:] if c not in columns_to_difference] 
        else:
            columns_to_difference = self.price_data_df.columns[1:]
            columns_to_ignore = []

        differenced_price_data_df = pd.DataFrame()

        for c in columns_to_ignore:
            values = []
            time_steps = []

            for i in range(self.differencing_interval, len(self.price_data_df)):
                time_steps.append(self.price_data_df['TIMESTAMP'][i])
                values.append(self.price_data_df[c][i])
            
            differenced_price_data_df['TIMESTAMP'] = time_steps
            differenced_price_data_df[c] = values

        for c in columns_to_difference:
            differences = []
            time_steps = []

            for i in range(self.differencing_interval, len(self.price_data_df)):
                current_differenced_value = self.price_data_df[c][i] - self.price_data_df[c][i - self.differencing_interval]

                time_steps.append(self.price_data_df['TIMESTAMP'][i])
                differences.append(current_differenced_value)
            
            differenced_price_data_df['TIMESTAMP'] = time_steps
            differenced_price_data_df[c] = differences

        self.price_data_df = differenced_price_data_df
    

    def apply_VMD_decomposition(
            self,
            close_price_ts,
    ):
        all_IMFs = []

        vmd = VMD(
            K=self.VMD_n_IMFs,
            alpha=self.VMD_alpha, 
            tau=self.VMD_tau, 
            tol=1e-5,
        )
        
        VMD_IMFs = vmd.fit_transform(signal=close_price_ts)
        all_IMFs.append(VMD_IMFs)

        if self.include_VMD_res:
            VMD_residual = close_price_ts.copy()

            for imf in VMD_IMFs:
                VMD_residual -= imf.reshape(-1,1)

            if self.apply_res_CEEMDAN:
                ceemdan = CEEMDAN(
                    trials=500,
                    epsilon=2.0,
                    max_imfs=15,
                )
                
                VMD_residual_CEEMDAN_IMFs = ceemdan.fit_transform(signal=VMD_residual.squeeze())
                
                all_IMFs.append(VMD_residual_CEEMDAN_IMFs)

                CEEMDAN_residual = VMD_residual.copy()

                for imf in VMD_residual_CEEMDAN_IMFs:
                    CEEMDAN_residual -= imf.reshape(-1,1)
                
                if self.include_CEEMDAN_res:
                    all_IMFs.append(CEEMDAN_residual.reshape(1,-1))

            else:
                all_IMFs.append(VMD_residual.reshape(1,-1))

        return np.concatenate(all_IMFs).T


    def apply_MVMD_decomposition(
            self,
            open_price_ts,
            high_price_ts,
            low_price_ts,
            close_price_ts,
    ):
        all_open_price_IMFs = []
        all_high_price_IMFs = []
        all_low_price_IMFs = []
        all_close_price_IMFs = []

        multivariate_ts = np.vstack([
            open_price_ts.squeeze(), 
            high_price_ts.squeeze(), 
            low_price_ts.squeeze(), 
            close_price_ts.squeeze(),
        ])

        mvmd = MVMD(
            K=self.VMD_n_IMFs,
            alpha=self.VMD_alpha, 
            tau=self.VMD_tau, 
            tol=1e-5,
        )

        MVMD_IMFs = mvmd.fit_transform(signal=multivariate_ts)

        all_open_price_IMFs.append(MVMD_IMFs[:,:,0])
        all_high_price_IMFs.append(MVMD_IMFs[:,:,1])
        all_low_price_IMFs.append(MVMD_IMFs[:,:,2])
        all_close_price_IMFs.append(MVMD_IMFs[:,:,3])

        if self.include_VMD_res:
            open_price_MVMD_residual = open_price_ts.copy()
            high_price_MVMD_residual = high_price_ts.copy()
            low_price_MVMD_residual = low_price_ts.copy()
            close_price_MVMD_residual = close_price_ts.copy()

            for imf_i in range(MVMD_IMFs.shape[0]):
                open_price_MVMD_residual -= MVMD_IMFs[imf_i,:,0]
                high_price_MVMD_residual -= MVMD_IMFs[imf_i,:,1]
                low_price_MVMD_residual -= MVMD_IMFs[imf_i,:,2]
                close_price_MVMD_residual -= MVMD_IMFs[imf_i,:,3]
            
            all_open_price_IMFs.append(open_price_MVMD_residual.reshape(1, -1))
            all_high_price_IMFs.append(high_price_MVMD_residual.reshape(1, -1))
            all_low_price_IMFs.append(low_price_MVMD_residual.reshape(1, -1))
            all_close_price_IMFs.append(close_price_MVMD_residual.reshape(1, -1))

        return np.concatenate(all_open_price_IMFs).T, np.concatenate(all_high_price_IMFs).T, np.concatenate(all_low_price_IMFs).T, np.concatenate(all_close_price_IMFs).T


    def create_windowed_static_VMD_dataset(
            self,
            
            open_price_ts_train,
            open_price_ts_test,
            high_price_ts_train,
            high_price_ts_test,
            low_price_ts_train,
            low_price_ts_test,
            close_price_ts_train,
            close_price_ts_test,
    ):
        if (len(close_price_ts_train) % 2) != 0:
            open_price_ts_train = open_price_ts_train[1:]
            high_price_ts_train = high_price_ts_train[1:]
            low_price_ts_train = low_price_ts_train[1:]
            close_price_ts_train = close_price_ts_train[1:]

        if self.apply_MVMD:
            open_price_ts_train_IMFs, high_price_ts_train_IMFs, low_price_ts_train_IMFs, close_price_ts_train_IMFs  = self.apply_MVMD_decomposition(
                open_price_ts=open_price_ts_train,
                high_price_ts=high_price_ts_train,
                low_price_ts=low_price_ts_train,
                close_price_ts=close_price_ts_train,
            )

        else:
            close_price_ts_train_IMFs = self.apply_VMD_decomposition(close_price_ts=close_price_ts_train)

        VMD_window_size = len(close_price_ts_train_IMFs)

        # Training Set

        for i in range(0, len(close_price_ts_train) - self.input_window - self.output_window - self.horizon + 2, self.stride):
            current_close_price_X_train_instance = close_price_ts_train_IMFs[i:i + self.input_window]

            if self.apply_MVMD:
                current_open_price_X_train_instance = open_price_ts_train_IMFs[i:i + self.input_window]
                current_high_price_X_train_instance = high_price_ts_train_IMFs[i:i + self.input_window]
                current_low_price_X_train_instance = low_price_ts_train_IMFs[i:i + self.input_window]
            else:
                current_open_price_X_train_instance = open_price_ts_train[i:i + self.input_window].reshape(-1, 1)
                current_high_price_X_train_instance = high_price_ts_train[i:i + self.input_window].reshape(-1, 1)
                current_low_price_X_train_instance = low_price_ts_train[i:i + self.input_window].reshape(-1, 1)

            current_y_train_instance = close_price_ts_train[i + self.input_window + self.horizon - 1:i + self.input_window + self.horizon - 1 + self.output_window]

            self.open_price_data_X_train.append(current_open_price_X_train_instance)
            self.high_price_data_X_train.append(current_high_price_X_train_instance)
            self.low_price_data_X_train.append(current_low_price_X_train_instance)
            self.close_price_data_X_train.append(current_close_price_X_train_instance)
            self.y_train.append(current_y_train_instance)

        # Test Set
        
        for i in list(range(0, len(close_price_ts_test) - self.output_window - self.horizon + 2, self.stride)):
            if i == 0:
                if self.apply_MVMD:
                    current_historical_open_price_ts_IMFs = open_price_ts_train_IMFs
                    current_historical_high_price_ts_IMFs = high_price_ts_train_IMFs
                    current_historical_low_price_ts_IMFs = low_price_ts_train_IMFs
                        
                current_historical_close_price_ts_IMFs = close_price_ts_train_IMFs
            else:
                current_historical_close_price_ts = np.concatenate([
                    close_price_ts_train,
                    close_price_ts_test[:i],
                ])[-VMD_window_size:]

                if self.apply_MVMD:
                    current_historical_open_price_ts = np.concatenate([
                        open_price_ts_train,
                        open_price_ts_test[:i],
                    ])[-VMD_window_size:]

                    current_historical_high_price_ts = np.concatenate([
                        high_price_ts_train,
                        high_price_ts_test[:i],
                    ])[-VMD_window_size:]

                    current_historical_low_price_ts = np.concatenate([
                        low_price_ts_train,
                        low_price_ts_test[:i],
                    ])[-VMD_window_size:]

                    current_historical_open_price_ts_IMFs, current_historical_high_price_ts_IMFs, current_historical_low_price_ts_IMFs, current_historical_close_price_ts_IMFs = self.apply_MVMD_decomposition(
                        open_price_ts=current_historical_open_price_ts,
                        high_price_ts=current_historical_high_price_ts,
                        low_price_ts=current_historical_low_price_ts,
                        close_price_ts=current_historical_close_price_ts
                    )
                

                else:
                    current_historical_close_price_ts_IMFs = self.apply_VMD_decomposition(close_price_ts=current_historical_close_price_ts)

            current_close_price_X_test_instance = current_historical_close_price_ts_IMFs[-self.input_window:]

            if self.apply_MVMD:
                current_open_price_X_test_instance = current_historical_open_price_ts_IMFs[-self.input_window:]
                current_high_price_X_test_instance = current_historical_high_price_ts_IMFs[-self.input_window:]
                current_low_price_X_test_instance = current_historical_low_price_ts_IMFs[-self.input_window:]
            else:
                current_open_price_X_test_instance = np.concatenate([
                    open_price_ts_train,
                    open_price_ts_test,
                ])[len(open_price_ts_train) - self.input_window + i:len(open_price_ts_train) + i]

                current_high_price_X_test_instance = np.concatenate([
                    high_price_ts_train,
                    high_price_ts_test,
                ])[len(high_price_ts_train) - self.input_window + i:len(high_price_ts_train) + i]
                
                current_low_price_X_test_instance = np.concatenate([
                    low_price_ts_train,
                    low_price_ts_test,
                ])[len(low_price_ts_train) - self.input_window + i:len(low_price_ts_train) + i]
        
            current_y_test_instance = close_price_ts_test[i + self.horizon - 1:i + self.horizon - 1 + self.output_window]

            self.open_price_data_X_test.append(current_open_price_X_test_instance)
            self.high_price_data_X_test.append(current_high_price_X_test_instance)
            self.low_price_data_X_test.append(current_low_price_X_test_instance)
            self.close_price_data_X_test.append(current_close_price_X_test_instance)
            self.y_test.append(current_y_test_instance)
    

    def create_windowed_dynamic_VMD_dataset(
            self,
            
            open_price_ts_train,
            open_price_ts_test,
            high_price_ts_train,
            high_price_ts_test,
            low_price_ts_train,
            low_price_ts_test,
            close_price_ts_train,
            close_price_ts_test,
    ):
        VMD_window_size = self.input_window

        # Training Set

        for i in range(0, len(close_price_ts_train) - self.input_window - self.output_window - self.horizon + 2, self.stride):
            current_historical_close_price_ts = close_price_ts_train[i:i + VMD_window_size] 

            if self.apply_MVMD:
                current_historical_open_price_ts = open_price_ts_train[i:i + VMD_window_size] 
                current_historical_high_price_ts = high_price_ts_train[i:i + VMD_window_size] 
                current_historical_low_price_ts = low_price_ts_train[i:i + VMD_window_size] 

                current_historical_open_price_ts_IMFs, current_historical_high_price_ts_IMFs, current_historical_low_price_ts_IMFs, current_historical_close_price_ts_IMFs = self.apply_MVMD_decomposition(
                    open_price_ts=current_historical_open_price_ts,
                    high_price_ts=current_historical_high_price_ts,
                    low_price_ts=current_historical_low_price_ts,
                    close_price_ts=current_historical_close_price_ts
                )

            else:
                current_historical_close_price_ts_IMFs = self.apply_VMD_decomposition(close_price_ts=current_historical_close_price_ts)

            current_close_price_X_train_instance = current_historical_close_price_ts_IMFs[-self.input_window:]

            if self.apply_MVMD:
                current_open_price_X_train_instance = current_historical_open_price_ts_IMFs[-self.input_window:]
                current_high_price_X_train_instance = current_historical_high_price_ts_IMFs[-self.input_window:]
                current_low_price_X_train_instance = current_historical_low_price_ts_IMFs[-self.input_window:]
            else:
                current_open_price_X_train_instance = open_price_ts_train[i:i + self.input_window]
                current_high_price_X_train_instance = high_price_ts_train[i:i + self.input_window]
                current_low_price_X_train_instance = low_price_ts_train[i:i + self.input_window]
        
            current_y_train_instance = close_price_ts_train[i + self.input_window + self.horizon - 1:i + self.input_window + self.horizon - 1 + self.output_window]

            self.open_price_data_X_train.append(current_open_price_X_train_instance)
            self.high_price_data_X_train.append(current_high_price_X_train_instance)
            self.low_price_data_X_train.append(current_low_price_X_train_instance)
            self.close_price_data_X_train.append(current_close_price_X_train_instance)
            self.y_train.append(current_y_train_instance)

        # Test Set
        
        for i in list(range(0, len(close_price_ts_test) - self.output_window - self.horizon + 2, self.stride)):
            current_historical_close_price_ts = np.concatenate([
                close_price_ts_train,
                close_price_ts_test[:i],
            ])[-VMD_window_size:]

            if self.apply_MVMD:
                current_historical_open_price_ts = np.concatenate([
                    open_price_ts_train,
                    open_price_ts_test[:i],
                ])[-VMD_window_size:]

                current_historical_high_price_ts = np.concatenate([
                    high_price_ts_train,
                    high_price_ts_test[:i],
                ])[-VMD_window_size:]

                current_historical_low_price_ts = np.concatenate([
                    low_price_ts_train,
                    low_price_ts_test[:i],
                ])[-VMD_window_size:]

                current_historical_open_price_ts_IMFs, current_historical_high_price_ts_IMFs, current_historical_low_price_ts_IMFs, current_historical_close_price_ts_IMFs = self.apply_MVMD_decomposition(
                    open_price_ts=current_historical_open_price_ts,
                    high_price_ts=current_historical_high_price_ts,
                    low_price_ts=current_historical_low_price_ts,
                    close_price_ts=current_historical_close_price_ts
                )

            else:
                current_historical_close_price_ts_IMFs = self.apply_VMD_decomposition(close_price_ts=current_historical_close_price_ts)

            current_close_price_X_test_instance = current_historical_close_price_ts_IMFs[-self.input_window:]

            if self.apply_MVMD:
                current_open_price_X_test_instance = current_historical_open_price_ts_IMFs[-self.input_window:]
                current_high_price_X_test_instance = current_historical_high_price_ts_IMFs[-self.input_window:]
                current_low_price_X_test_instance = current_historical_low_price_ts_IMFs[-self.input_window:]
            else:
                current_open_price_X_test_instance = np.concatenate([
                    open_price_ts_train,
                    open_price_ts_test,
                ])[len(open_price_ts_train) - self.input_window + i:len(open_price_ts_train) + i]

                current_high_price_X_test_instance = np.concatenate([
                    high_price_ts_train,
                    high_price_ts_test,
                ])[len(high_price_ts_train) - self.input_window + i:len(high_price_ts_train) + i]
                
                current_low_price_X_test_instance = np.concatenate([
                    low_price_ts_train,
                    low_price_ts_test,
                ])[len(low_price_ts_train) - self.input_window + i:len(low_price_ts_train) + i]
        
            current_y_test_instance = close_price_ts_test[i + self.horizon - 1:i + self.horizon - 1 + self.output_window].reshape(-1, 1)
            
            self.open_price_data_X_test.append(current_open_price_X_test_instance)
            self.high_price_data_X_test.append(current_high_price_X_test_instance)
            self.low_price_data_X_test.append(current_low_price_X_test_instance)
            self.close_price_data_X_test.append(current_close_price_X_test_instance)
            self.y_test.append(current_y_test_instance)


    def create_default_windowed_dataset(
            self,
            
            open_price_ts_train,
            open_price_ts_test,
            high_price_ts_train,
            high_price_ts_test,
            low_price_ts_train,
            low_price_ts_test,
            close_price_ts_train,
            close_price_ts_test,
    ):
        # Training Set

        for i in range(0, len(close_price_ts_train) - self.input_window - self.output_window - self.horizon + 2, self.stride):
            current_open_price_X_train_instance = open_price_ts_train[i:i + self.input_window]
            current_high_price_X_train_instance = high_price_ts_train[i:i + self.input_window]
            current_low_price_X_train_instance = low_price_ts_train[i:i + self.input_window]
            current_close_price_X_train_instance = close_price_ts_train[i:i + self.input_window]

            current_y_train_instance = close_price_ts_train[i + self.input_window + self.horizon - 1:i + self.input_window + self.horizon - 1 + self.output_window]

            self.open_price_data_X_train.append(current_open_price_X_train_instance)
            self.high_price_data_X_train.append(current_high_price_X_train_instance)
            self.low_price_data_X_train.append(current_low_price_X_train_instance)
            self.close_price_data_X_train.append(current_close_price_X_train_instance)
            self.y_train.append(current_y_train_instance)

        # Test Set
        
        for i in list(range(0, len(close_price_ts_test) - self.output_window - self.horizon + 2, self.stride)):
            current_open_price_X_test_instance = np.concatenate([
                open_price_ts_train,
                open_price_ts_test,
            ])[len(open_price_ts_train) - self.input_window + i:len(open_price_ts_train) + i]

            current_high_price_X_test_instance = np.concatenate([
                high_price_ts_train,
                high_price_ts_test,
            ])[len(high_price_ts_train) - self.input_window + i:len(high_price_ts_train) + i]
            
            current_low_price_X_test_instance = np.concatenate([
                low_price_ts_train,
                low_price_ts_test,
            ])[len(low_price_ts_train) - self.input_window + i:len(low_price_ts_train) + i]

            current_close_price_X_test_instance = np.concatenate([
                close_price_ts_train,
                close_price_ts_test,
            ])[len(close_price_ts_train) - self.input_window + i:len(close_price_ts_train) + i]

            current_y_test_instance = close_price_ts_test[i + self.horizon - 1:i + self.horizon - 1 + self.output_window]

            self.open_price_data_X_test.append(current_open_price_X_test_instance)
            self.high_price_data_X_test.append(current_high_price_X_test_instance)
            self.low_price_data_X_test.append(current_low_price_X_test_instance)
            self.close_price_data_X_test.append(current_close_price_X_test_instance)
            self.y_test.append(current_y_test_instance)


    def normalize_windowed_input(
            self,
            X_train,
            X_test,
    ):
        n_features = X_train.shape[-1]

        for i in range(n_features):
            current_scaler = self.scaler_type()
            X_train[:, :, i] = current_scaler.fit_transform(X_train[:, :, i])
            X_test[:, :, i] = current_scaler.transform(X_test[:, :, i])
        
        return X_train, X_test


    def normalize_windowed_output(self):
        for i in range(self.y_train.shape[-1]):
            self.y_windowed_scalers[i] = self.scaler_type()
            self.y_train[:, :, i] = self.y_windowed_scalers[i].fit_transform(self.y_train[:, :, i])
            self.y_test[:, :, i] = self.y_windowed_scalers[i].transform(self.y_test[:, :, i])


    def windowed_dataset_normalization(self):
        self.open_price_data_X_train, self.open_price_data_X_test = self.normalize_windowed_input(
            X_train=self.open_price_data_X_train,
            X_test=self.open_price_data_X_test,
        )
        self.high_price_data_X_train, self.high_price_data_X_test = self.normalize_windowed_input(
            X_train=self.high_price_data_X_train,
            X_test=self.high_price_data_X_test,
        )
        self.low_price_data_X_train, self.low_price_data_X_test = self.normalize_windowed_input(
            X_train=self.low_price_data_X_train,
            X_test=self.low_price_data_X_test,
        )
        self.close_price_data_X_train, self.close_price_data_X_test = self.normalize_windowed_input(
            X_train=self.close_price_data_X_train,
            X_test=self.close_price_data_X_test,
        )
    
        self.normalize_windowed_output()

    
    def regular_dataset_normalization(
            self,
            train_test_split_idx,
            ts,
        ):
        scaler = self.scaler_type()
        ts_train = ts[:train_test_split_idx]
        ts_test = ts[train_test_split_idx:]

        normalized_ts_train = scaler.fit_transform(ts_train)
        normalized_ts_test = scaler.transform(ts_test)

        return normalized_ts_train, normalized_ts_test, scaler


    def create_windowed_dataset(
            self,
            train_test_split_idx,
    ):  
        open_price_ts = self.price_data_df[f'{self.selected_crypto}_OPEN_PRICE_{self.currency}'].to_numpy().reshape(-1,1)
        high_price_ts = self.price_data_df[f'{self.selected_crypto}_HIGH_PRICE_{self.currency}'].to_numpy().reshape(-1,1)
        low_price_ts = self.price_data_df[f'{self.selected_crypto}_LOW_PRICE_{self.currency}'].to_numpy().reshape(-1,1)
        close_price_ts = self.price_data_df[f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}'].to_numpy().reshape(-1,1)

        if self.normalize_before_windowing:
            open_price_ts_train, open_price_ts_test, self.open_price_scaler = self.regular_dataset_normalization(
                train_test_split_idx=train_test_split_idx,
                ts=open_price_ts
            )
            high_price_ts_train, high_price_ts_test, self.high_price_scaler = self.regular_dataset_normalization(
                train_test_split_idx=train_test_split_idx,
                ts=high_price_ts
            )
            low_price_ts_train, low_price_ts_test, self.low_price_scaler = self.regular_dataset_normalization(
                train_test_split_idx=train_test_split_idx,
                ts=low_price_ts
            )
            close_price_ts_train, close_price_ts_test, self.close_price_scaler = self.regular_dataset_normalization(
                train_test_split_idx=train_test_split_idx,
                ts=close_price_ts
            )
        else:
            open_price_ts_train, open_price_ts_test = (
                open_price_ts[:train_test_split_idx], 
                open_price_ts[train_test_split_idx:]
            )
            high_price_ts_train, high_price_ts_test = (
                high_price_ts[:train_test_split_idx], 
                high_price_ts[train_test_split_idx:]
            )
            low_price_ts_train, low_price_ts_test = (
                low_price_ts[:train_test_split_idx], 
                low_price_ts[train_test_split_idx:]
            )
            close_price_ts_train, close_price_ts_test = (
                close_price_ts[:train_test_split_idx], 
                close_price_ts[train_test_split_idx:]
            )

        if self.apply_VMD:
            if self.windowed_VMD:
                self.create_windowed_dynamic_VMD_dataset(
                    open_price_ts_train=open_price_ts_train,
                    open_price_ts_test=open_price_ts_test,

                    high_price_ts_train=high_price_ts_train,
                    high_price_ts_test=high_price_ts_test,

                    low_price_ts_train=low_price_ts_train,
                    low_price_ts_test=low_price_ts_test,

                    close_price_ts_train=close_price_ts_train,
                    close_price_ts_test=close_price_ts_test,
                )
            else:
                self.create_windowed_static_VMD_dataset(
                    open_price_ts_train=open_price_ts_train,
                    open_price_ts_test=open_price_ts_test,

                    high_price_ts_train=high_price_ts_train,
                    high_price_ts_test=high_price_ts_test,

                    low_price_ts_train=low_price_ts_train,
                    low_price_ts_test=low_price_ts_test,

                    close_price_ts_train=close_price_ts_train,
                    close_price_ts_test=close_price_ts_test,
                )
        else:
            self.create_default_windowed_dataset(
                open_price_ts_train=open_price_ts_train,
                open_price_ts_test=open_price_ts_test,

                high_price_ts_train=high_price_ts_train,
                high_price_ts_test=high_price_ts_test,

                low_price_ts_train=low_price_ts_train,
                low_price_ts_test=low_price_ts_test,

                close_price_ts_train=close_price_ts_train,
                close_price_ts_test=close_price_ts_test,
            )
    
        self.open_price_data_X_train = np.array(self.open_price_data_X_train)
        self.high_price_data_X_train = np.array(self.high_price_data_X_train)
        self.low_price_data_X_train = np.array(self.low_price_data_X_train)
        self.close_price_data_X_train = np.array(self.close_price_data_X_train)

        self.open_price_data_X_test = np.array(self.open_price_data_X_test)
        self.high_price_data_X_test = np.array(self.high_price_data_X_test)
        self.low_price_data_X_test = np.array(self.low_price_data_X_test)
        self.close_price_data_X_test = np.array(self.close_price_data_X_test)

        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        if not self.normalize_before_windowing:
            self.windowed_dataset_normalization()

    
    def define_close_price_ground_truth(
            self,
            train_test_split_idx,
        ):
        close_price_ts = self.original_price_data_df[f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}'].to_numpy().reshape(-1, 1)

        if self.windowed_VMD:
            start_offset = int((train_test_split_idx % 2) != 0)
        else:
            start_offset = int(self.apply_differencing)

        close_price_ts_train = close_price_ts[start_offset:train_test_split_idx + int(self.apply_differencing)]
        close_price_ts_test = close_price_ts[train_test_split_idx + int(self.apply_differencing):]

        if not self.normalize_before_windowing:
            for i in range(0, len(close_price_ts_train) - self.input_window - self.output_window - self.horizon + 2, self.stride):
                self.y_train_close_price.append(close_price_ts_train[i + self.input_window + self.horizon - 1:i + self.input_window + self.horizon - 1 + self.output_window])

            for i in list(range(0, len(close_price_ts_test) - self.output_window - self.horizon + 2, self.stride)):
                self.y_test_close_price.append(close_price_ts_test[i + self.horizon - 1:i + self.horizon - 1 + self.output_window])

            self.y_train_close_price = np.array(self.y_train_close_price)
            self.y_test_close_price = np.array(self.y_test_close_price)

            self.y_train_normalized_close_price = self.y_train_close_price.copy()
            self.y_test_normalized_close_price = self.y_test_close_price.copy()

            for i in range(self.y_train_normalized_close_price.shape[-1]):
                self.y_windowed_close_price_scalers[i] = self.scaler_type()
                self.y_train_normalized_close_price[:, :, i] = self.y_windowed_close_price_scalers[i].fit_transform(self.y_train_normalized_close_price[:, :, i])
                self.y_test_normalized_close_price[:, :, i] = self.y_windowed_close_price_scalers[i].transform(self.y_test_normalized_close_price[:, :, i])
        else:
            close_price_ts_train_normalized, close_price_ts_test_normalized, self.y_close_price_scaler = self.regular_dataset_normalization(
                train_test_split_idx=train_test_split_idx,
                ts=close_price_ts,
            )

            close_price_ts_train_normalized = close_price_ts_train_normalized[start_offset:]

            for i in range(0, len(close_price_ts_train) - self.input_window - self.output_window - self.horizon + 2, self.stride):
                self.y_train_close_price.append(close_price_ts_train[i + self.input_window + self.horizon - 1:i + self.input_window + self.horizon - 1 + self.output_window])
                self.y_train_normalized_close_price.append(close_price_ts_train_normalized[i + self.input_window + self.horizon - 1:i + self.input_window + self.horizon - 1 + self.output_window])

            for i in list(range(0, len(close_price_ts_test) - self.output_window - self.horizon + 2, self.stride)):
                self.y_test_close_price.append(close_price_ts_test[i + self.horizon - 1:i + self.horizon - 1 + self.output_window])
                self.y_test_normalized_close_price.append(close_price_ts_test_normalized[i + self.horizon - 1:i + self.horizon - 1 + self.output_window])
            
            self.y_train_close_price = np.array(self.y_train_close_price)
            self.y_test_close_price = np.array(self.y_test_close_price)
            self.y_train_normalized_close_price = np.array(self.y_train_normalized_close_price)
            self.y_test_normalized_close_price = np.array(self.y_test_normalized_close_price)


    def process_dataset(self):
        if not self.apply_VMD and self.apply_differencing:
            self.apply_price_differencing()
        
        train_test_split_idx = list(self.price_data_df['TIMESTAMP']).index(self.train_test_split_date)

        self.create_windowed_dataset(train_test_split_idx=train_test_split_idx)
        self.define_close_price_ground_truth(train_test_split_idx=train_test_split_idx)

        if self.apply_VMD:
            n_IMFs = self.close_price_data_X_train.shape[-1]

            if self.apply_MVMD:
                VMD_tag = 'MVMD'
            else:
                VMD_tag = 'VMD'

            for i in range(n_IMFs):
                if i < self.VMD_n_IMFs:
                    self.close_price_data_X_names.append(f'{VMD_tag} IMF{i+1}')
                elif i == (n_IMFs - 1):
                    if self.include_CEEMDAN_res:
                        self.close_price_data_X_names.append(f'CEEMDAN Residual')
                    elif self.apply_res_CEEMDAN:
                        self.close_price_data_X_names.append(f'CEEMDAN IMF{i-self.VMD_n_IMFs+1}')
                    elif self.include_VMD_res:
                        self.close_price_data_X_names.append(f'{VMD_tag} Residual')
                    else:
                        self.close_price_data_X_names.append(f'{VMD_tag} IMF{i+1}')
                else:
                    self.close_price_data_X_names.append(f'CEEMDAN IMF{i-self.VMD_n_IMFs+1}')
        else:
            self.close_price_data_X_names.append(f'Close Price')


    def inverse_price_differencing_predictions(
            self, 
            differenced_preds, 
    ):
        difference_inverted_values = np.zeros_like(differenced_preds)
        start_point = len(self.original_price_data_df) - len(differenced_preds) - self.differencing_interval

        for i in range(differenced_preds.shape[0]): 
            for t in range(differenced_preds.shape[1]):
                difference_inverted_values[i,t,0] = differenced_preds[i,t,0] + self.original_price_data_df[f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}'][start_point + i + t + self.horizon - 1] 
        
        return difference_inverted_values


    def inverse_normalization_predictions(
            self, 
            preds,
    ):
        inverse_normalization_preds = np.zeros_like(preds)
        
        if self.normalize_before_windowing:
            inverse_normalization_preds[:, :, 0] = self.close_price_scaler.inverse_transform(preds[:, :, 0])
        else:
            for f in range(preds.shape[-1]):
                inverse_normalization_preds[:, :, f] = self.y_windowed_scalers[f].inverse_transform(preds[:, :, f]) 

        return inverse_normalization_preds 


    def inverse_transform_specific_IMF_predictions(
            self,
            preds,
            selected_IMF_idx,
    ):
        inverse_transformed_specific_IMF_preds = np.zeros_like(preds)

        inverse_transformed_specific_IMF_preds[:,:,0] = self.y_windowed_scalers[selected_IMF_idx].inverse_transform(preds[:,:,0])

        return inverse_transformed_specific_IMF_preds 


    def inverse_transform_predictions(
            self, 
            preds,
    ):
        inverse_transformed_preds = self.inverse_normalization_predictions(preds=preds)

        if self.apply_differencing:
            inverse_transformed_preds = self.inverse_price_differencing_predictions(
                differenced_preds=inverse_transformed_preds,
            )

        return inverse_transformed_preds 
    
    
    def normalize_predictions(
            self, 
            preds,
    ):
        normalized_preds = np.zeros_like(preds)

        if self.normalize_before_windowing:
            normalized_preds[:, :, 0] = self.close_price_scaler.transform(preds[:, :, 0]) 
        else:
            for i in range(preds.shape[-1]):
                normalized_preds[:, :, i] = self.y_windowed_scalers[i].transform(preds[:, :, i]) 

        return normalized_preds 
    

    def normalize_price_predictions(
            self, 
            preds,
    ):  
        normalized_price_preds = np.zeros_like(preds)

        if self.normalize_before_windowing:
            normalized_price_preds[:, :, 0] = self.y_close_price_scaler.transform(preds[:, :, 0]) 
        else:
            for i in range(preds.shape[-1]):
                normalized_price_preds[:, :, i] = self.y_windowed_close_price_scalers[i].transform(preds[:, :, i]) 

        return normalized_price_preds      

    # def create_windowed_static_VMD_dataset(
    #         self,
            
    #         open_price_ts_train,
    #         open_price_ts_test,
    #         high_price_ts_train,
    #         high_price_ts_test,
    #         low_price_ts_train,
    #         low_price_ts_test,
    #         close_price_ts_train,
    #         close_price_ts_test,
    # ):
    #     if (len(close_price_ts_train) % 2) != 0:
    #         open_price_ts_train = open_price_ts_train[1:]
    #         high_price_ts_train = high_price_ts_train[1:]
    #         low_price_ts_train = low_price_ts_train[1:]
    #         close_price_ts_train = close_price_ts_train[1:]

    #     if self.apply_MVMD:
    #         open_price_ts_train_IMFs, high_price_ts_train_IMFs, low_price_ts_train_IMFs, close_price_ts_train_IMFs  = self.apply_MVMD_decomposition(
    #             open_price_ts=open_price_ts_train,
    #             high_price_ts=high_price_ts_train,
    #             low_price_ts=low_price_ts_train,
    #             close_price_ts=close_price_ts_train,
    #         )

    #     else:
    #         close_price_ts_train_IMFs = self.apply_VMD_decomposition(close_price_ts=close_price_ts_train)
        
    #     VMD_window_size = len(close_price_ts_train_IMFs)

    #     # Training Set

    #     for i in range(0, len(close_price_ts_train) - self.input_window - self.output_window - self.horizon + 2, self.stride):
    #         current_close_price_X_train_instance = close_price_ts_train_IMFs[i:i + self.input_window]

    #         if self.apply_MVMD:
    #             current_open_price_X_train_instance = open_price_ts_train_IMFs[i:i + self.input_window]
    #             current_high_price_X_train_instance = high_price_ts_train_IMFs[i:i + self.input_window]
    #             current_low_price_X_train_instance = low_price_ts_train_IMFs[i:i + self.input_window]
    #         else:
    #             current_open_price_X_train_instance = open_price_ts_train[i:i + self.input_window].reshape(-1, 1)
    #             current_high_price_X_train_instance = high_price_ts_train[i:i + self.input_window].reshape(-1, 1)
    #             current_low_price_X_train_instance = low_price_ts_train[i:i + self.input_window].reshape(-1, 1)

    #         if self.target_IMFs:
    #             current_y_train_instance = close_price_ts_train_IMFs[i + self.input_window + self.horizon - 1:i + self.input_window + self.horizon - 1 + self.output_window]

    #         else:
    #             current_y_train_instance = close_price_ts_train[i + self.input_window + self.horizon - 1:i + self.input_window + self.horizon - 1 + self.output_window].reshape(-1, 1)

    #         self.open_price_data_X_train.append(current_open_price_X_train_instance)
    #         self.high_price_data_X_train.append(current_high_price_X_train_instance)
    #         self.low_price_data_X_train.append(current_low_price_X_train_instance)
    #         self.close_price_data_X_train.append(current_close_price_X_train_instance)
    #         self.y_train.append(current_y_train_instance)

    #     # Test Set
        
    #     for i in list(range(0, len(close_price_ts_test) - self.output_window - self.horizon + 2, self.stride)):
    #         correct_historical_defined = False

    #         if (i == 0):
    #             if self.apply_MVMD:
    #                 current_historical_open_price_ts_IMFs = open_price_ts_train_IMFs
    #                 current_historical_high_price_ts_IMFs = high_price_ts_train_IMFs
    #                 current_historical_low_price_ts_IMFs = low_price_ts_train_IMFs
                        
    #             current_historical_close_price_ts_IMFs = close_price_ts_train_IMFs
    #             correct_historical_defined = True

    #         if not correct_historical_defined:
    #             current_historical_close_price_ts = np.concatenate([
    #                 close_price_ts_train,
    #                 close_price_ts_test[:i],
    #             ])[-VMD_window_size:]

    #             if self.apply_MVMD:
    #                 current_historical_open_price_ts = np.concatenate([
    #                     open_price_ts_train,
    #                     open_price_ts_test[:i],
    #                 ])[-VMD_window_size:]

    #                 current_historical_high_price_ts = np.concatenate([
    #                     high_price_ts_train,
    #                     high_price_ts_test[:i],
    #                 ])[-VMD_window_size:]

    #                 current_historical_low_price_ts = np.concatenate([
    #                     low_price_ts_train,
    #                     low_price_ts_test[:i],
    #                 ])[-VMD_window_size:]

    #                 current_historical_open_price_ts_IMFs, current_historical_high_price_ts_IMFs, current_historical_low_price_ts_IMFs, current_historical_close_price_ts_IMFs = self.apply_MVMD_decomposition(
    #                     open_price_ts=current_historical_open_price_ts,
    #                     high_price_ts=current_historical_high_price_ts,
    #                     low_price_ts=current_historical_low_price_ts,
    #                     close_price_ts=current_historical_close_price_ts
    #                 )

    #             else:
    #                 current_historical_close_price_ts_IMFs = self.apply_VMD_decomposition(close_price_ts=current_historical_close_price_ts)

    #         if self.target_IMFs:
    #             current_y_test_instance = []

    #             for j in range(self.output_window):
    #                 current_present_close_price_ts = np.concatenate([
    #                     close_price_ts_train,
    #                     close_price_ts_test[:i + self.horizon - 1 + j + 1],
    #                 ])[-VMD_window_size:]

    #                 if self.apply_MVMD:
    #                     current_present_open_price_ts = np.concatenate([
    #                         open_price_ts_train,
    #                         open_price_ts_test[:i + self.horizon - 1 + j + 1],
    #                     ])[-VMD_window_size:]

    #                     current_present_high_price_ts = np.concatenate([
    #                         high_price_ts_train,
    #                         high_price_ts_test[:i + self.horizon - 1 + j + 1],
    #                     ])[-VMD_window_size:]

    #                     current_present_low_price_ts = np.concatenate([
    #                         low_price_ts_train,
    #                         low_price_ts_test[:i + self.horizon - 1 + j + 1],
    #                     ])[-VMD_window_size:]

    #                     current_present_open_price_ts_IMFs, current_present_high_price_ts_IMFs, current_present_low_price_ts_IMFs, current_present_close_price_ts_IMFs = self.apply_MVMD_decomposition(
    #                         open_price_ts=current_present_open_price_ts,
    #                         high_price_ts=current_present_high_price_ts,
    #                         low_price_ts=current_present_low_price_ts,
    #                         close_price_ts=current_present_close_price_ts
    #                     )

    #                 else:
    #                     current_present_close_price_ts_IMFs = self.apply_VMD_decomposition(close_price_ts=current_present_close_price_ts)

    #                 current_y_test_instance.append(current_present_close_price_ts_IMFs[-1])
                    
    #                 if (i + j) == (i + self.stride - 1):
    #                     if self.apply_MVMD:
    #                         next_historical_open_price_ts_IMFs = current_present_open_price_ts_IMFs
    #                         next_historical_high_price_ts_IMFs = current_present_high_price_ts_IMFs
    #                         next_historical_low_price_ts_IMFs = current_present_low_price_ts_IMFs
                        
    #                     next_historical_close_price_ts_IMFs = current_present_close_price_ts_IMFs
    #                     correct_historical_defined = True

    #             if self.apply_MVMD:
    #                 current_open_price_X_test_instance = current_historical_open_price_ts_IMFs[-self.input_window:]
    #                 current_historical_open_price_ts_IMFs = next_historical_open_price_ts_IMFs

    #                 current_high_price_X_test_instance = current_historical_high_price_ts_IMFs[-self.input_window:]
    #                 current_historical_high_price_ts_IMFs = next_historical_high_price_ts_IMFs

    #                 current_low_price_X_test_instance = current_historical_low_price_ts_IMFs[-self.input_window:]
    #                 current_historical_low_price_ts_IMFs = next_historical_low_price_ts_IMFs
                    
    #             current_close_price_X_test_instance = current_historical_close_price_ts_IMFs[-self.input_window:]
    #             current_historical_close_price_ts_IMFs = next_historical_close_price_ts_IMFs

    #             current_y_test_instance = np.array(current_y_test_instance)

    #         else:
    #             if self.apply_MVMD:
    #                 current_open_price_X_test_instance = current_historical_open_price_ts_IMFs[-self.input_window:]
    #                 current_high_price_X_test_instance = current_historical_high_price_ts_IMFs[-self.input_window:]
    #                 current_low_price_X_test_instance = current_historical_low_price_ts_IMFs[-self.input_window:]
            
    #             current_close_price_X_test_instance = current_historical_close_price_ts_IMFs[-self.input_window:]

    #             current_y_test_instance = close_price_ts_test[i + self.horizon - 1:i + self.horizon - 1 + self.output_window].reshape(-1, 1)

    #         if not self.apply_MVMD:
    #             current_open_price_X_test_instance = np.concatenate([
    #                 open_price_ts_train,
    #                 open_price_ts_test,
    #             ])[len(open_price_ts_train) - self.input_window + i:len(open_price_ts_train) + i].reshape(-1, 1)

    #             current_high_price_X_test_instance = np.concatenate([
    #                 high_price_ts_train,
    #                 high_price_ts_test,
    #             ])[len(high_price_ts_train) - self.input_window + i:len(high_price_ts_train) + i].reshape(-1, 1)
                
    #             current_low_price_X_test_instance = np.concatenate([
    #                 low_price_ts_train,
    #                 low_price_ts_test,
    #             ])[len(low_price_ts_train) - self.input_window + i:len(low_price_ts_train) + i].reshape(-1, 1)

    #         self.open_price_data_X_test.append(current_open_price_X_test_instance)
    #         self.high_price_data_X_test.append(current_high_price_X_test_instance)
    #         self.low_price_data_X_test.append(current_low_price_X_test_instance)
    #         self.close_price_data_X_test.append(current_close_price_X_test_instance)

    #         self.y_test.append(current_y_test_instance)
    

    # def create_windowed_dynamic_VMD_dataset(
    #         self,
            
    #         open_price_ts_train,
    #         open_price_ts_test,
    #         high_price_ts_train,
    #         high_price_ts_test,
    #         low_price_ts_train,
    #         low_price_ts_test,
    #         close_price_ts_train,
    #         close_price_ts_test,
    # ):
    #     VMD_window_size = self.input_window

    #     # Training Set

    #     for i in range(0, len(close_price_ts_train) - self.input_window - self.output_window - self.horizon + 2, self.stride):
    #         correct_historical_defined = False

    #         if not correct_historical_defined:
    #             current_historical_close_price_ts = close_price_ts_train[i:i + VMD_window_size] 

    #             if self.apply_MVMD:
    #                 current_historical_open_price_ts = open_price_ts_train[i:i + VMD_window_size] 
    #                 current_historical_high_price_ts = high_price_ts_train[i:i + VMD_window_size] 
    #                 current_historical_low_price_ts = low_price_ts_train[i:i + VMD_window_size] 

    #                 current_historical_open_price_ts_IMFs, current_historical_high_price_ts_IMFs, current_historical_low_price_ts_IMFs, current_historical_close_price_ts_IMFs = self.apply_MVMD_decomposition(
    #                     open_price_ts=current_historical_open_price_ts,
    #                     high_price_ts=current_historical_high_price_ts,
    #                     low_price_ts=current_historical_low_price_ts,
    #                     close_price_ts=current_historical_close_price_ts
    #                 )

    #             else:
    #                 current_historical_close_price_ts_IMFs = self.apply_VMD_decomposition(close_price_ts=current_historical_close_price_ts)

    #         if self.target_IMFs:
    #             current_y_train_instance = []

    #             for j in range(self.output_window):
    #                 current_present_close_price_ts = close_price_ts_train[i + self.horizon - 1 + j + 1:i + VMD_window_size + self.horizon - 1 + j + 1]

    #                 if self.apply_MVMD:
    #                     current_present_open_price_ts = open_price_ts_train[i + self.horizon - 1 + j + 1:i + VMD_window_size + self.horizon - 1 + j + 1]
    #                     current_present_high_price_ts = high_price_ts_train[i + self.horizon - 1 + j + 1:i + VMD_window_size + self.horizon - 1 + j + 1]
    #                     current_present_low_price_ts = low_price_ts_train[i + self.horizon - 1 + j + 1:i + VMD_window_size + self.horizon - 1 + j + 1]

    #                     current_present_open_price_ts_IMFs, current_present_high_price_ts_IMFs, current_present_low_price_ts_IMFs, current_present_close_price_ts_IMFs = self.apply_MVMD_decomposition(
    #                         open_price_ts=current_present_open_price_ts,
    #                         high_price_ts=current_present_high_price_ts,
    #                         low_price_ts=current_present_low_price_ts,
    #                         close_price_ts=current_present_close_price_ts
    #                     )

    #                 else:
    #                     current_present_close_price_ts_IMFs = self.apply_VMD_decomposition(close_price_ts=current_present_close_price_ts)

    #                 current_y_train_instance.append(current_present_close_price_ts_IMFs[-1])
                    
    #                 if (i + j) == (i + self.stride - 1):
    #                     if self.apply_MVMD:
    #                         next_historical_open_price_ts_IMFs = current_present_open_price_ts_IMFs
    #                         next_historical_high_price_ts_IMFs = current_present_high_price_ts_IMFs
    #                         next_historical_low_price_ts_IMFs = current_present_low_price_ts_IMFs
                        
    #                     next_historical_close_price_ts_IMFs = current_present_close_price_ts_IMFs
    #                     correct_historical_defined = True

    #             if self.apply_MVMD:
    #                 current_open_price_X_train_instance = current_historical_open_price_ts_IMFs[-self.input_window:]
    #                 current_historical_open_price_ts_IMFs = next_historical_open_price_ts_IMFs

    #                 current_high_price_X_train_instance = current_historical_high_price_ts_IMFs[-self.input_window:]
    #                 current_historical_high_price_ts_IMFs = next_historical_high_price_ts_IMFs

    #                 current_low_price_X_train_instance = current_historical_low_price_ts_IMFs[-self.input_window:]
    #                 current_historical_low_price_ts_IMFs = next_historical_low_price_ts_IMFs                    
            
    #             current_close_price_X_train_instance = current_historical_close_price_ts_IMFs[-self.input_window:]
    #             current_historical_close_price_ts_IMFs = next_historical_close_price_ts_IMFs

    #             current_y_train_instance = np.array(current_y_train_instance)

    #         else:
    #             if self.apply_MVMD:
    #                 current_open_price_X_train_instance = current_historical_open_price_ts_IMFs[-self.input_window:]
    #                 current_high_price_X_train_instance = current_historical_high_price_ts_IMFs[-self.input_window:]
    #                 current_low_price_X_train_instance = current_historical_low_price_ts_IMFs[-self.input_window:]
            
    #             current_close_price_X_train_instance = current_historical_close_price_ts_IMFs[-self.input_window:]

    #             current_y_train_instance = close_price_ts_train[i + self.input_window + self.horizon - 1:i + self.input_window + self.horizon - 1 + self.output_window].reshape(-1, 1)

    #         if not self.apply_MVMD:
    #             current_open_price_X_train_instance = open_price_ts_train[i:i + self.input_window].reshape(-1, 1)
    #             current_high_price_X_train_instance = high_price_ts_train[i:i + self.input_window].reshape(-1, 1)
    #             current_low_price_X_train_instance = low_price_ts_train[i:i + self.input_window].reshape(-1, 1)

    #         self.open_price_data_X_train.append(current_open_price_X_train_instance)
    #         self.high_price_data_X_train.append(current_high_price_X_train_instance)
    #         self.low_price_data_X_train.append(current_low_price_X_train_instance)
    #         self.close_price_data_X_train.append(current_close_price_X_train_instance)
    #         self.y_train.append(current_y_train_instance)

    #     # Test Set
        
    #     for i in list(range(0, len(close_price_ts_test) - self.output_window - self.horizon + 2, self.stride)):
    #         correct_historical_defined = False

    #         if not correct_historical_defined:
    #             current_historical_close_price_ts = np.concatenate([
    #                 close_price_ts_train,
    #                 close_price_ts_test[:i],
    #             ])[-VMD_window_size:]

    #             if self.apply_MVMD:
    #                 current_historical_open_price_ts = np.concatenate([
    #                     open_price_ts_train,
    #                     open_price_ts_test[:i],
    #                 ])[-VMD_window_size:]

    #                 current_historical_high_price_ts = np.concatenate([
    #                     high_price_ts_train,
    #                     high_price_ts_test[:i],
    #                 ])[-VMD_window_size:]

    #                 current_historical_low_price_ts = np.concatenate([
    #                     low_price_ts_train,
    #                     low_price_ts_test[:i],
    #                 ])[-VMD_window_size:]

    #                 current_historical_open_price_ts_IMFs, current_historical_high_price_ts_IMFs, current_historical_low_price_ts_IMFs, current_historical_close_price_ts_IMFs = self.apply_MVMD_decomposition(
    #                     open_price_ts=current_historical_open_price_ts,
    #                     high_price_ts=current_historical_high_price_ts,
    #                     low_price_ts=current_historical_low_price_ts,
    #                     close_price_ts=current_historical_close_price_ts
    #                 )

    #             else:
    #                 current_historical_close_price_ts_IMFs = self.apply_VMD_decomposition(close_price_ts=current_historical_close_price_ts)

    #         if self.target_IMFs:
    #             current_y_test_instance = []

    #             for j in range(self.output_window):
    #                 current_present_close_price_ts = np.concatenate([
    #                     close_price_ts_train,
    #                     close_price_ts_test[:i + self.horizon - 1 + j + 1],
    #                 ])[-VMD_window_size:]

    #                 if self.apply_MVMD:
    #                     current_present_open_price_ts = np.concatenate([
    #                         open_price_ts_train,
    #                         open_price_ts_test[:i + self.horizon - 1 + j + 1],
    #                     ])[-VMD_window_size:]

    #                     current_present_high_price_ts = np.concatenate([
    #                         high_price_ts_train,
    #                         high_price_ts_test[:i + self.horizon - 1 + j + 1],
    #                     ])[-VMD_window_size:]

    #                     current_present_low_price_ts = np.concatenate([
    #                         low_price_ts_train,
    #                         low_price_ts_test[:i + self.horizon - 1 + j + 1],
    #                     ])[-VMD_window_size:]

    #                     current_present_open_price_ts_IMFs, current_present_high_price_ts_IMFs, current_present_low_price_ts_IMFs, current_present_close_price_ts_IMFs = self.apply_MVMD_decomposition(
    #                         open_price_ts=current_present_open_price_ts,
    #                         high_price_ts=current_present_high_price_ts,
    #                         low_price_ts=current_present_low_price_ts,
    #                         close_price_ts=current_present_close_price_ts
    #                     )

    #                 else:
    #                     current_present_close_price_ts_IMFs = self.apply_VMD_decomposition(close_price_ts=current_present_close_price_ts)

    #                 current_y_test_instance.append(current_present_close_price_ts_IMFs[-1])
                    
    #                 if (i + j) == (i + self.stride - 1):
    #                     if self.apply_MVMD:
    #                         next_historical_open_price_ts_IMFs = current_present_open_price_ts_IMFs
    #                         next_historical_high_price_ts_IMFs = current_present_high_price_ts_IMFs
    #                         next_historical_low_price_ts_IMFs = current_present_low_price_ts_IMFs
                        
    #                     next_historical_close_price_ts_IMFs = current_present_close_price_ts_IMFs
    #                     correct_historical_defined = True

    #             if self.apply_MVMD:
    #                 current_open_price_X_test_instance = current_historical_open_price_ts_IMFs[-self.input_window:]
    #                 current_historical_open_price_ts_IMFs = next_historical_open_price_ts_IMFs

    #                 current_high_price_X_test_instance = current_historical_high_price_ts_IMFs[-self.input_window:]
    #                 current_historical_high_price_ts_IMFs = next_historical_high_price_ts_IMFs

    #                 current_low_price_X_test_instance = current_historical_low_price_ts_IMFs[-self.input_window:]
    #                 current_historical_low_price_ts_IMFs = next_historical_low_price_ts_IMFs
                    
    #             current_close_price_X_test_instance = current_historical_close_price_ts_IMFs[-self.input_window:]
    #             current_historical_close_price_ts_IMFs = next_historical_close_price_ts_IMFs

    #             current_y_test_instance = np.array(current_y_test_instance)

    #         else:
    #             if self.apply_MVMD:
    #                 current_open_price_X_test_instance = current_historical_open_price_ts_IMFs[-self.input_window:]
    #                 current_high_price_X_test_instance = current_historical_high_price_ts_IMFs[-self.input_window:]
    #                 current_low_price_X_test_instance = current_historical_low_price_ts_IMFs[-self.input_window:]
            
    #             current_close_price_X_test_instance = current_historical_close_price_ts_IMFs[-self.input_window:]

    #             current_y_test_instance = close_price_ts_test[i + self.horizon - 1:i + self.horizon - 1 + self.output_window].reshape(-1, 1)
            
    #         if not self.apply_MVMD:
    #             current_open_price_X_test_instance = np.concatenate([
    #                 open_price_ts_train,
    #                 open_price_ts_test,
    #             ])[len(open_price_ts_train) - self.input_window + i:len(open_price_ts_train) + i].reshape(-1, 1)

    #             current_high_price_X_test_instance = np.concatenate([
    #                 high_price_ts_train,
    #                 high_price_ts_test,
    #             ])[len(high_price_ts_train) - self.input_window + i:len(high_price_ts_train) + i].reshape(-1, 1)
                
    #             current_low_price_X_test_instance = np.concatenate([
    #                 low_price_ts_train,
    #                 low_price_ts_test,
    #             ])[len(low_price_ts_train) - self.input_window + i:len(low_price_ts_train) + i].reshape(-1, 1)

    #         self.open_price_data_X_test.append(current_open_price_X_test_instance)
    #         self.high_price_data_X_test.append(current_high_price_X_test_instance)
    #         self.low_price_data_X_test.append(current_low_price_X_test_instance)
    #         self.close_price_data_X_test.append(current_close_price_X_test_instance)

    #         self.y_test.append(current_y_test_instance)
