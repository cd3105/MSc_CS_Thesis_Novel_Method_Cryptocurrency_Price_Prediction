import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from pysdkit import VMD, CEEMDAN

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

            input_OHL,
            input_C,
            target_IMFs,
            
            apply_VMD,
            VMD_n_IMFs,
            VMD_alpha,
            VMD_tau,
            include_VMD_res,
            apply_res_CEEMDAN,
            include_CEEMDAN_res,

            apply_differencing,
            target_only_differencing,
            differencing_interval=1,

            input_window=30, 
            output_window=1, 
            stride=1,
            horizon=1,
    ):
        self.selected_crypto = crypto
        self.selected_interval = freq
        self.selected_experiment = experiment
        self.selected_source = source
        self.currency = currency
    
        self.input_OHL = input_OHL
        self.input_C = input_C

        self.apply_VMD = apply_VMD
        self.VMD_n_IMFs = VMD_n_IMFs
        self.VMD_alpha = VMD_alpha
        self.VMD_tau = VMD_tau
        self.include_VMD_res = include_VMD_res
        self.apply_res_CEEMDAN = apply_res_CEEMDAN
        self.include_CEEMDAN_res = include_CEEMDAN_res
        self.target_IMFs = target_IMFs

        self.apply_differencing = apply_differencing
        self.target_only_differencing = target_only_differencing
        self.differencing_interval = differencing_interval

        self.train_test_split_date = train_test_split_date
        self.next_train_test_split_date = next_train_test_split_date

        self.input_window = input_window
        self.stride = stride
        self.output_window = output_window
        self.horizon = horizon

        self.price_data_df_fn = f'All_Crypto_Data/Adopted_Data/Processed/{experiment}/Price_Data/{source}/{crypto}/{freq}/{crypto}_Price_Data.csv'
        self.price_data_df = pd.read_csv(self.price_data_df_fn)
        self.price_data_df['TIMESTAMP'] = pd.to_datetime(self.price_data_df['TIMESTAMP'])
        self.price_data_df = self.price_data_df[self.price_data_df['TIMESTAMP'] < next_train_test_split_date]

        if ((len(self.price_data_df) % 2) != 0) and self.apply_VMD:
            self.price_data_df = self.price_data_df.iloc[1:].reset_index(drop=True)
        
        self.original_price_data_df = self.price_data_df.copy()

        self.price_data_X = []
        self.price_data_X_train = []
        self.price_data_X_val = []
        self.price_data_X_test = []
        self.price_data_X_scalers = {}

        self.y = []
        self.y_train = []
        self.y_test = []
        self.y_scalers = {}

        self.y_train_close_price = []
        self.y_train_normalized_close_price = []
        self.y_test_close_price = []
        self.y_test_normalized_close_price = []
        self.y_close_price_scaler = None

        self.y_train_IMFs = []
        self.y_train_normalized_IMFs = []
        self.y_test_IMFs = []
        self.y_test_normalized_IMFs = []
        self.y_IMFs_scaler = None

        self.IMFs_ground_truth = []


    def summary(
            self, 
            tabs="",
    ):
        print(f'{tabs}- Train Set Shape: {self.price_data_X_train.shape} and {self.y_train.shape}')
        print(f'{tabs}- Testing Set Shape: {self.price_data_X_test.shape} and {self.y_test.shape}\n')


    def apply_leaky_price_decomposition(self):
        close_price_ts = np.array(self.price_data_df[f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}'])
        close_price_IMF_columns = []

        vmd = VMD(
            K=self.VMD_n_IMFs,
            alpha=self.VMD_alpha, 
            tau=self.VMD_tau, 
            tol=1e-5,
        )
        decomposed_close_price_VMD_IMFs = vmd.fit_transform(signal=close_price_ts)

        for i, imf in enumerate(decomposed_close_price_VMD_IMFs):
            self.price_data_df[f'DECOMPOSED_{self.selected_crypto}_CLOSE_PRICE_{self.currency}_VMD_IMF_{i+1}'] = imf.copy()
            close_price_IMF_columns.append(f'DECOMPOSED_{self.selected_crypto}_CLOSE_PRICE_{self.currency}_VMD_IMF_{i+1}')
        
        if self.include_VMD_res:
            close_price_VMD_residual = close_price_ts.copy()

            for imf in decomposed_close_price_VMD_IMFs:
                close_price_VMD_residual -= imf

            if not self.apply_res_CEEMDAN:
                self.price_data_df[f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}_VMD_Residual'] = close_price_VMD_residual
                close_price_IMF_columns.append(f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}_VMD_Residual')
            else:
                ceemdan = CEEMDAN(
                    trials=500,
                    epsilon=2.0,
                    max_imfs=15,
                )
                
                decomposed_close_price_VMD_residual_CEEMDAN_IMFs = ceemdan.fit_transform(signal=close_price_VMD_residual)

                for i, imf in enumerate(decomposed_close_price_VMD_residual_CEEMDAN_IMFs):
                    self.price_data_df[f'DECOMPOSED_{self.selected_crypto}_CLOSE_PRICE_{self.currency}_CEEMDAN_IMF_{i+1}'] = imf.copy()
                    close_price_IMF_columns.append(f'DECOMPOSED_{self.selected_crypto}_CLOSE_PRICE_{self.currency}_CEEMDAN_IMF_{i+1}')

                close_price_CEEMDAN_residual = close_price_VMD_residual.copy()

                for imf in decomposed_close_price_VMD_residual_CEEMDAN_IMFs:
                    close_price_CEEMDAN_residual -= imf
                
                if self.include_CEEMDAN_res:
                    self.price_data_df[f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}_CEEMDAN_Residual'] = close_price_CEEMDAN_residual
                    close_price_IMF_columns.append(f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}_CEEMDAN_Residual')

        return close_price_IMF_columns


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


    def dataset_normalization(self):
        for i, ic in enumerate(self.price_data_input_columns):
            self.price_data_X_scalers[i] = MinMaxScaler()
            self.price_data_X_train[:, :, i] = self.price_data_X_scalers[i].fit_transform(self.price_data_X_train[:, :, i])
            self.price_data_X_test[:, :, i] = self.price_data_X_scalers[i].transform(self.price_data_X_test[:, :, i])


        for i, tc in enumerate(self.target_columns):
            self.y_scalers[i] = MinMaxScaler()
            self.y_train[:, :, i] = self.y_scalers[i].fit_transform(self.y_train[:, :, i])
            self.y_test[:, :, i] = self.y_scalers[i].transform(self.y_test[:, :, i])
        

    def create_windowed_dataset(
            self,
            train_test_split_idx,
    ):
        input_vars = self.price_data_df[self.price_data_input_columns].to_numpy()
        target_vars = self.price_data_df[self.target_columns].to_numpy()

        self.price_data_X = []
        self.y = []

        for i in range(0, len(input_vars) - self.input_window - self.output_window - self.horizon + 2, self.stride):
            self.price_data_X.append(input_vars[i:i + self.input_window])
            self.y.append(target_vars[i + self.input_window + self.horizon - 1:i + self.input_window + self.horizon - 1 + self.output_window])

        self.price_data_X = np.array(self.price_data_X)
        self.y = np.array(self.y)

        self.price_data_X_train, self.price_data_X_test = (self.price_data_X[:train_test_split_idx], self.price_data_X[train_test_split_idx:]) # (self.price_data_X[:train_test_split_idx], self.price_data_X[train_test_split_idx:])

        self.y_train, self.y_test = (self.y[:train_test_split_idx], self.y[train_test_split_idx:]) # (self.y[:train_test_split_idx], self.y[train_test_split_idx:])

    
    def define_ground_truth(self):
        train_test_split_idx = list(self.original_price_data_df['TIMESTAMP']).index(self.train_test_split_date)
        close_price = np.array(self.original_price_data_df[f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}'])

        if self.apply_differencing:
            self.y_train_close_price = close_price[self.input_window + self.differencing_interval:train_test_split_idx].reshape(-1, 1)
        else:
            self.y_train_close_price = close_price[self.input_window:train_test_split_idx].reshape(-1, 1)
        
        self.y_test_close_price = close_price[train_test_split_idx:].reshape(-1, 1) # close_price[train_test_split_idx:].reshape(-1, 1)

        self.y_close_price_scaler = MinMaxScaler()
        self.y_train_normalized_close_price = self.y_close_price_scaler.fit_transform(self.y_train_close_price)
        self.y_test_normalized_close_price = self.y_close_price_scaler.transform(self.y_test_close_price)

        if self.target_IMFs:
            IMFs = np.array(self.price_data_df[self.target_columns])
            
            self.y_train_IMFs, self.y_test_IMFs = (IMFs[self.input_window:train_test_split_idx], IMFs[train_test_split_idx:]) # (IMFs[self.input_window:train_test_split_idx], IMFs[train_test_split_idx:])

            self.y_IMFs_scaler = MinMaxScaler()
            self.y_train_normalized_IMFs = self.y_IMFs_scaler.fit_transform(self.y_train_IMFs)
            self.y_test_normalized_IMFs = self.y_IMFs_scaler.transform(self.y_test_IMFs)


    def process_dataset(self):
        mode_columns = []

        if not self.input_OHL:
            self.price_data_df = self.price_data_df.drop(
                columns=[f'{self.selected_crypto}_OPEN_PRICE_{self.currency}',
                         f'{self.selected_crypto}_HIGH_PRICE_{self.currency}',
                         f'{self.selected_crypto}_LOW_PRICE_{self.currency}',]
            )

        if self.target_IMFs and not self.apply_VMD:
            self.apply_VMD = True

        if self.apply_VMD:
            mode_columns = self.apply_leaky_price_decomposition()       

        if self.target_IMFs:
            self.target_columns = mode_columns
        else:
            self.target_columns = [f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}']

        self.price_data_input_columns = list(self.price_data_df.columns)[1:]

        if not self.apply_VMD and self.apply_differencing:
            self.apply_differencing()
        
        if (self.apply_VMD and not self.input_C) or (self.input_OHL and not self.input_C):
            self.price_data_input_columns.remove(f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}')
        
        train_test_split_idx = list(self.price_data_df['TIMESTAMP']).index(self.train_test_split_date) - self.input_window
        
        self.create_windowed_dataset(
            train_test_split_idx=train_test_split_idx,
        )

        if self.apply_VMD:
            self.IMFs_ground_truth = self.y_test.copy()

        self.dataset_normalization()
        self.define_ground_truth()


    def inverse_price_differencing_predictions(
            self, 
            differenced_preds, 
            offset_length=0,
    ):
        target = self.target_columns[0]
        difference_inverted_values = []

        if offset_length == 0: 
            start_point = len(self.original_price_data_df) - len(differenced_preds) - self.differencing_interval
        else:
            start_point = len(self.original_price_data_df) - offset_length - self.differencing_interval

        for i in range(len(differenced_preds)): 
            current_inversed_difference_value = differenced_preds[i] + self.original_price_data_df[target][start_point + i] 
            difference_inverted_values.append(current_inversed_difference_value) 
        
        return np.array(difference_inverted_values)


    def inverse_normalization_predictions(
            self, 
            preds,
    ):
        inverse_normalization_preds = np.zeros_like(preds)

        for i, tc in enumerate(self.target_columns):
            current_column_preds = preds[:, i].reshape(-1, 1)
            current_column_inverse_normalization_preds = self.y_scalers[i].inverse_transform(current_column_preds) 
            inverse_normalization_preds[:, i] = current_column_inverse_normalization_preds.ravel()

        return inverse_normalization_preds 


    def inverse_transform_specific_IMF_predictions(
            self,
            preds,
            selected_IMF_idx,
    ):
        return self.y_scalers[selected_IMF_idx].inverse_transform(preds)  


    def inverse_transform_predictions(
            self, 
            preds,
            inversion_differencing_offset_length=0,
    ):
        inverse_transformed_preds = self.inverse_normalization_predictions(preds=preds)

        if self.apply_differencing:
            inverse_transformed_preds = self.inverse_price_differencing_predictions(
                differenced_preds=inverse_transformed_preds,
                offset_length=inversion_differencing_offset_length
            )

        return inverse_transformed_preds 
    
    
    def normalize_predictions(
            self, 
            preds,
    ):             
        normalized_preds = np.zeros_like(preds)

        for i, tc in enumerate(self.target_columns):
            current_column_preds = preds[:, i].reshape(-1, 1)
            current_column_normalized_preds = self.y_scalers[i].transform(current_column_preds) 
            normalized_preds[:, i] = current_column_normalized_preds.ravel()

        return normalized_preds 
    

    def normalize_price_predictions(
            self, 
            preds,
    ):  
        return self.y_close_price_scaler.transform(preds)       
