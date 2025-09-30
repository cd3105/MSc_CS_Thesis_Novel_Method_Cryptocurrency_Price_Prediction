import pandas as pd
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timezone
from pysdkit import VMD


class CryptoDataset:
    def __init__(self, 
                 crypto="BTC",
                 interval="1_Day",
                 time_span="Baseline",
                 experiment="Baseline",
                 source="Binance",
                 currency="USDT",

                 apply_vmd=True,
                 apply_differencing=False,
                 target_only_differencing=False,

                 input_ohl=False,
                 input_c=False,
                 target_IMFs=False,
                 
                 differencing_interval=1,

                 vmd_n_IMFs=6,
                 vmd_alpha=250,
                 vmd_tau=0.15,
                 vmd_tol=1e-6,
                 vmd_DC=False,
                 vmd_init='uniform',

                 split_offset=0,
                 validation_size=0.2,
                 validation_split_before_normalization=True,

                 input_window=30, 
                 output_window=1, 
                 stride=1,
                 horizon=1,):
        
        self.selected_crypto = crypto
        self.selected_interval = interval
        self.selected_time_span = time_span
        self.selected_experiment = experiment
        self.selected_source = source
        self.currency = currency

        self.price_data_df_fn = f'All_Crypto_Data/Adopted_Data/Processed/{time_span}/Price_Data/{source}/{crypto}/{interval}/{crypto}_Price_Data.csv'
        self.price_data_df = pd.read_csv(self.price_data_df_fn)
        self.price_data_df['TIMESTAMP'] = pd.to_datetime(self.price_data_df['TIMESTAMP'])
        self.original_price_data_df = self.price_data_df.copy()
    
        self.input_ohl = input_ohl
        self.input_c = input_c
        self.target_IMFs = target_IMFs

        self.differencing_interval = differencing_interval

        self.apply_vmd = apply_vmd
        self.vmd_n_IMFs = vmd_n_IMFs
        self.vmd_alpha = vmd_alpha
        self.vmd_tau = vmd_tau
        self.vmd_tol = vmd_tol
        self.vmd_DC = vmd_DC
        self.vmd_init = vmd_init

        self.apply_differencing = apply_differencing
        self.target_only_differencing = target_only_differencing

        self.split_offset = split_offset
        self.validation_size = validation_size
        self.validation_split_before_normalization = validation_split_before_normalization

        self.input_window = input_window
        self.stride = stride
        self.output_window = output_window
        self.horizon = horizon

        self.price_data_X = []
        self.price_data_X_train = []
        self.price_data_X_val = []
        self.price_data_X_test = []
        self.price_data_X_scalers = {}

        self.y = []
        self.y_train = []
        self.y_val = []
        self.y_test = []
        self.y_scalers = {}

        self.y_train_close_price = []
        self.y_train_normalized_close_price = []
        self.y_val_close_price = []
        self.y_val_normalized_close_price = []
        self.y_test_close_price = []
        self.y_test_normalized_close_price = []
        self.y_close_price_scaler = None

        self.y_train_IMFs = []
        self.y_train_normalized_IMFs = []
        self.y_val_IMFs = []
        self.y_val_normalized_IMFs = []
        self.y_test_IMFs = []
        self.y_test_normalized_IMFs = []
        self.y_IMFs_scaler = None

        self.IMFs_ground_truth = []


    def summary(self,
                tabs=""):
        print(f'{tabs}- Train & Val Set Shape: {(self.price_data_X_train.shape[0] + self.price_data_X_val.shape[0], self.price_data_X_train.shape[1], self.price_data_X_train.shape[2])} and {(self.y_train.shape[0] + self.y_val.shape[0], self.y_train.shape[1], self.y_train.shape[2])}')
        print(f'{tabs}\t- Train Set Shape: {self.price_data_X_train.shape} and {self.y_train.shape}')
        print(f'{tabs}\t- Val Set Shape: {self.price_data_X_val.shape} and {self.y_val.shape}')
        print(f'{tabs}- Testing Set Shape: {self.price_data_X_test.shape} and {self.y_test.shape}\n')


    def apply_price_decomposition(self):
        close_price_ts = np.array(self.price_data_df[f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}'])
        vmd = VMD(K=self.vmd_n_IMFs,
                  alpha=self.vmd_alpha, 
                  tau=self.vmd_tau, 
                  tol=self.vmd_tol,
                  DC=self.vmd_DC,
                  init=self.vmd_init)
        decomposed_close_price_IMFs = vmd.fit_transform(signal=close_price_ts)
        IMF_columns = []

        for i, m in enumerate(decomposed_close_price_IMFs):
            self.price_data_df[f'DECOMPOSED_{self.selected_crypto}_CLOSE_PRICE_{self.currency}_IMF_{i+1}'] = m
            IMF_columns.append(f'DECOMPOSED_{self.selected_crypto}_CLOSE_PRICE_{self.currency}_IMF_{i+1}')
        
        return IMF_columns


    def apply_price_differencing(self):
        if self.target_only_differencing:
            columns_to_difference = self.target_columns
        else:
            columns_to_difference = self.price_data_df.columns

        differenced_price_data_df = pd.DataFrame()

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
            self.price_data_X_val[:, :, i] = self.price_data_X_scalers[i].transform(self.price_data_X_val[:, :, i])
            self.price_data_X_test[:, :, i] = self.price_data_X_scalers[i].transform(self.price_data_X_test[:, :, i])


        for i, tc in enumerate(self.target_columns):
            self.y_scalers[i] = MinMaxScaler()
            self.y_train[:, :, i] = self.y_scalers[i].fit_transform(self.y_train[:, :, i])
            self.y_val[:, :, i] = self.y_scalers[i].transform(self.y_val[:, :, i])
            self.y_test[:, :, i] = self.y_scalers[i].transform(self.y_test[:, :, i])
        

    def create_windowed_dataset(self,
                                train_test_split_idx,
                                train_val_split_idx):
        input_vars = self.price_data_df[self.price_data_input_columns].to_numpy()
        input_vars = tf.data.Dataset.from_tensor_slices(input_vars)
        input_vars = input_vars.window(self.input_window, 
                                       stride=self.stride, 
                                       shift=1,
                                       drop_remainder=True) 
        input_vars = input_vars.flat_map(lambda window: window.batch(self.input_window))
        input_vars = list(input_vars.as_numpy_iterator())[:-self.output_window]

        target_vars = self.price_data_df[self.target_columns].to_numpy()
        target_vars = tf.data.Dataset.from_tensor_slices(target_vars)
        target_vars = target_vars.window(self.output_window, 
                                         stride=self.stride, 
                                         shift=1,
                                         drop_remainder=True) 
        target_vars = target_vars.flat_map(lambda window: window.batch(self.output_window))
        target_vars = list(target_vars.as_numpy_iterator())[self.input_window:]

        for i, input in enumerate(input_vars):
            if i == len(input_vars) - self.horizon + 1:
                break 

            self.price_data_X.append(input)
            self.y.append(target_vars[i + self.horizon - 1])

        self.price_data_X = np.array(self.price_data_X)
        self.y = np.array(self.y)
        
        self.price_data_X_train, self.price_data_X_test = (self.price_data_X[:train_test_split_idx],
                                                           self.price_data_X[train_test_split_idx:])

        self.y_train, self.y_test = (self.y[:train_test_split_idx],
                                     self.y[train_test_split_idx:])

        self.price_data_X_train, self.price_data_X_val = (self.price_data_X_train[:train_val_split_idx], 
                                                          self.price_data_X_train[train_val_split_idx:])
        
        self.y_train, self.y_val = (self.y_train[:train_val_split_idx], 
                                    self.y_train[train_val_split_idx:])
        
    
    def define_ground_truth(self):
        train_test_split_idx = list(self.original_price_data_df['TIMESTAMP']).index(self.initial_split_date) + self.split_offset
        close_price = np.array(self.original_price_data_df[f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}'])

        if self.apply_differencing:
            self.y_train_close_price = close_price[self.input_window + self.differencing_interval:train_test_split_idx].reshape(-1, 1)
        else:
            self.y_train_close_price = close_price[self.input_window:train_test_split_idx].reshape(-1, 1)
        
        self.y_test_close_price = close_price[train_test_split_idx:].reshape(-1, 1)
        
        train_val_split_idx = int(round(len(self.y_train_close_price) * (1 - self.validation_size)))

        self.y_train_close_price, self.y_val_close_price = (self.y_train_close_price[:train_val_split_idx].reshape(-1, 1), 
                                                            self.y_train_close_price[train_val_split_idx:].reshape(-1, 1))

        self.y_close_price_scaler = MinMaxScaler()
        self.y_train_normalized_close_price = self.y_close_price_scaler.fit_transform(self.y_train_close_price)
        self.y_val_normalized_close_price = self.y_close_price_scaler.transform(self.y_val_close_price)
        self.y_test_normalized_close_price = self.y_close_price_scaler.transform(self.y_test_close_price)

        if self.target_IMFs:
            IMFs = np.array(self.price_data_df[self.target_columns])
            
            self.y_train_IMFs, self.y_test_IMFs = (IMFs[self.input_window:train_test_split_idx],
                                                   IMFs[train_test_split_idx:])
            self.y_train_IMFs, self.y_val_IMFs = (self.y_train_IMFs[:train_val_split_idx],
                                                  self.y_train_IMFs[train_val_split_idx:])

            self.y_IMFs_scaler = MinMaxScaler()
            self.y_train_normalized_IMFs = self.y_IMFs_scaler.fit_transform(self.y_train_IMFs)
            self.y_val_normalized_IMFs = self.y_IMFs_scaler.transform(self.y_val_IMFs)
            self.y_test_normalized_IMFs = self.y_IMFs_scaler.transform(self.y_test_IMFs)


    def process_dataset(self):
        mode_columns = []

        if self.selected_experiment == "Baseline":
            self.initial_split_date = datetime(2021, 6, 1)

        if not self.input_ohl:
            self.price_data_df = self.price_data_df.drop(columns=[f'{self.selected_crypto}_OPEN_PRICE_{self.currency}',
                                                                  f'{self.selected_crypto}_HIGH_PRICE_{self.currency}',
                                                                  f'{self.selected_crypto}_LOW_PRICE_{self.currency}',])

        if self.target_IMFs and not self.apply_vmd:
            self.apply_vmd = True

        if self.apply_vmd:
            mode_columns = self.apply_price_decomposition()       

        if self.target_IMFs:
            self.target_columns = mode_columns
        else:
            self.target_columns = [f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}']

        self.price_data_input_columns = list(self.price_data_df.columns)[1:]

        if not self.apply_vmd and self.apply_differencing:
            self.apply_price_differencing()
        
        if (self.apply_vmd and not self.input_c) or (self.input_ohl and not self.input_c):
            self.price_data_input_columns.remove(f'{self.selected_crypto}_CLOSE_PRICE_{self.currency}')
        
        train_test_split_idx = list(self.price_data_df['TIMESTAMP']).index(self.initial_split_date) + self.split_offset - self.input_window
        train_val_split_idx = int(round(train_test_split_idx * (1 - self.validation_size)))

        self.create_windowed_dataset(train_test_split_idx=train_test_split_idx,
                                     train_val_split_idx=train_val_split_idx)

        if self.apply_vmd:
            self.IMFs_ground_truth = self.y_test.copy()

        self.dataset_normalization()
        self.define_ground_truth()


    def inverse_price_differencing_predictions(self, 
                                               differenced_preds, 
                                               start_point=0,
                                               offset_length=0,):
        target = self.target_columns[0]
        difference_inverted_values = []

        if start_point == 0:
            if offset_length == 0: 
                start_point = len(self.original_price_data_df) - len(differenced_preds) - self.differencing_interval
            else:
                start_point = len(self.original_price_data_df) - offset_length - self.differencing_interval

        for i in range(len(differenced_preds)): 
            current_inversed_difference_value = differenced_preds[i] + self.original_price_data_df[target][start_point + i] 
            difference_inverted_values.append(current_inversed_difference_value) 
        
        return np.array(difference_inverted_values)


    def inverse_normalization_predictions(self, 
                                          preds):
        inverse_normalization_preds = np.zeros_like(preds)

        for i, tc in enumerate(self.target_columns):
            current_column_preds = preds[:, i].reshape(-1, 1)
            current_column_inverse_normalization_preds = self.y_scalers[i].inverse_transform(current_column_preds) 
            inverse_normalization_preds[:, i] = current_column_inverse_normalization_preds.ravel()

        return inverse_normalization_preds 


    def inverse_transform_specific_IMF_predictions(self,
                                                   preds,
                                                   selected_IMF_idx):
        return self.y_scalers[selected_IMF_idx].inverse_transform(preds)  


    def inverse_transform_predictions(self, 
                                      preds,
                                      inversion_differencing_start_point=0,
                                      inversion_differencing_offset_length=0):
        inverse_transformed_preds = self.inverse_normalization_predictions(preds=preds)

        if self.apply_differencing:
            inverse_transformed_preds = self.inverse_price_differencing_predictions(differenced_preds=inverse_transformed_preds,
                                                                                    start_point=inversion_differencing_start_point,
                                                                                    offset_length=inversion_differencing_offset_length)

        return inverse_transformed_preds 
    
    
    def normalize_predictions(self, 
                              preds):             
        normalized_preds = np.zeros_like(preds)

        for i, tc in enumerate(self.target_columns):
            current_column_preds = preds[:, i].reshape(-1, 1)
            current_column_normalized_preds = self.y_scalers[i].transform(current_column_preds) 
            normalized_preds[:, i] = current_column_normalized_preds.ravel()

        return normalized_preds 
    

    def normalize_price_predictions(self, 
                                    preds):  
        return self.y_close_price_scaler.transform(preds)       

    