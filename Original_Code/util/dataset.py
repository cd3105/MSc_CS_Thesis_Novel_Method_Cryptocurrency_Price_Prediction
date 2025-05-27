"""
Interface of a Dataset class with shared functionalities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import pickle
from datetime import datetime, timezone
from darts import TimeSeries
from statsmodels.tsa.seasonal import seasonal_decompose
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import time

class DatasetInterface:
    def __init__(self, filename="", input_window=10, output_window=1, horizon=0, training_features=[], target_name=[],
                 train_split_factor=0.8):
        """
        Constructor of the DatasetInterface class
        :param filename: string: path of the dataset in .csv format. Default = ""
        :param input_window: int: input sequence, number of timestamps of the time series used for training the model
        :param output_window: int: output sequence, length of the prediction. Default = 1 (one-step-ahead prediction)
        :param horizon: int: index of the first future timestamp to predict. Default = 0
        :param training_features: array of string: names of the features used for the training. Default = []
        :param target_name: array of strings: names of the column to predict. Default = []
        :param train_split_factor: float: Training/Test split factor Default = 0.8
        """
        self.name = filename
        "string: name pof the experiment"

        self.X = []
        """list: Full dataset features in windowed format"""
        self.y = []
        """list: Full dataset labels in windowed format"""
        self.X_array = []
        """list: Full dataset features in series format """
        self.y_array = []
        """list: Full dataset labels in series format """
        self.X_train = []
        """list: Training features in windowed format"""
        self.X_test = []
        """list: Test features in windowed format"""
        self.y_train = []
        """list: Training labels in windowed format"""
        self.y_test = []
        """list: Test labels in windowed format"""
        self.X_train_array = []
        """list: Training features in series format"""
        self.y_train_array = []
        """list: Training labels in series format"""
        self.X_test_array = []
        """list: Test features in series format"""
        self.y_test_array = []
        """list: Test labels in series format"""
        self.ts_test = None
        """timeseries: test time series"""
        self.ts_train = None
        """timeseries: train time series"""
        self.tcov = None 
        """timeseries: future covariates"""
        self.train_cov = None
        """timeseries: training covariates"""
        self.training_features = training_features
        """list of strings: columns names of the features for the training"""
        self.target_name = target_name
        """list of strings: Columns names of the labels to predict"""
        self.channels = len(self.training_features)
        """int: number of input dimensions"""

        # Input files
        self.data_file = filename
        """string: dataset name"""
        self.data_path = 'Original_Code/saved_data/'
        """string: directory path of the dataset"""

        self.train_split_factor = train_split_factor
        """float: training/Test split factor"""

        self.normalization = None
        """list of strings: list of normalization methods to apply to features columns"""
        self.X_scalers = {}
        """dict: dictionary of scaler used for the features"""
        self.y_scalers = {}
        """dict: dictionary of scaler used for the labels"""
        self.input_window = input_window
        """int:  input sequence, number of timestamps of the time series used for training the model"""
        self.stride = 1
        """int: stride for the windowed dataset creation"""
        self.output_window = output_window
        """int: index of the first future timestamp to predict"""
        self.horizon = horizon
        """int: index of the first future timestamp to predict"""

        self.verbose = 1
        """int: level of verbosity of the dataset operations"""
        self.add_split_value = 0
        

    def data_save(self, name):
        """
        Save the dataset using pickle package
        :param name: string: name of the output file
        :return: None
        """
        with open(self.data_path + name, 'wb') as file:
            pickle.dump(self, file)
            print("File saved in " + self.data_path + name)


    def data_load(self, name):
        """
        Load the dataset using pickle package
        :param name: string: name of the inout file
        :return: None
        """
        with open(self.data_path + name, 'rb') as file:
            return pickle.load(file)


    def data_summary(self):
        """
        Print a summary of the dataset
        :return: None
        """
        print('Training', self.X_train.shape, 'Testing', self.X_test.shape)


    def dataset_creation(self, df = False, detrended = False, days_365=True):
        """
        Create all the datasets components with the training and test sets split.
        :param Boolean detrended: checks whether self.df is already set [Default = False]
        :return: None
        """
        if self.data_file is None: # Review if File Path has been specified
            print("ERROR: Source files not specified")
            return
        if self.verbose: # Print Data Load statement if Verbose has been passed
            print("Data load")

        # read the csv file into a pandas dataframe
        if not df: # Read in Original DataFrame if df has not yet been initialized
            self.df = pd.read_csv(self.data_path + self.data_file)

        columns = self.df[self.training_features].to_numpy() # Convert Selected Columns of Pandas Dataframe to Numpy Array
        self.X, self.y = self.__windowed_dataset(columns) # Retrieve Input Variable Numpy Array (X) and Target Variable Numpy Array (y) after Windowing
        test_start_date_idx = list(self.df['timestamp']).index(int(datetime(2021, 6, 1, tzinfo=timezone.utc).timestamp()))

        if not days_365:
            if detrended:
                window_split_value = (int(self.X.shape[0] * self.train_split_factor)-1) + self.add_split_value
            else:
                window_split_value = (int(self.X.shape[0] * self.train_split_factor)) + self.add_split_value
        else:
            window_split_value = (test_start_date_idx - self.input_window) + self.add_split_value

        # windowed dataset creation

        # Split X and y into Training and Test Set

        self.y_train = self.y[:window_split_value]
        self.y_test = self.y[window_split_value:]
        self.X_train = self.X[:window_split_value]
        self.X_test = self.X[window_split_value:]


        # unidimensional dataset creation
        self.X_array = self.df[self.target_name].to_numpy() # Convert Target Variable Columns to Numpy Array

        if len(self.target_name) == 1: # Reshape Target Variable Numpy Array in case there is a singular Target Variable
            self.X_array = self.X_array.reshape(-1, 1)

        if not days_365:
            if detrended: 
                split_value = (int(self.X_array.shape[0] * self.train_split_factor)-1) + self.add_split_value
            else: 
                split_value = split_value = (int(self.X_array.shape[0] * self.train_split_factor)) + self.add_split_value
        else:
            split_value = (test_start_date_idx - 1) + self.add_split_value

        self.X_train_array = self.X_array[:split_value] # Retrieve Training Portion of X Non-Windowed
        self.y_train_array = self.X_array[self.horizon + 1:self.horizon + split_value + 1] # Retrieve Training Portion of y Non-Windowed
        self.ts_train, self.ts_val, self.ts_test, self.train_cov, self.cov, self.ts_ttrain = self.get_ts_data(df=self.df) # Convert DataFrame to Darts Timeseries Data
       
        if self.horizon: # Retrieve Test Portion of X Non-Windowed depending on whether Horizon > 0
            self.X_test_array = self.X_array[split_value: -self.horizon - 1]
        else:
            self.X_test_array = self.X_array[split_value:-1]
        self.y_test_array = self.X_array[self.horizon + split_value + 1:]

        if self.verbose: # Print info if Verbose > 0 has been passed
            print("Data size ", self.X.shape)

        if self.verbose: # Print info if Verbose > 0 has been passed
            print("Training size ", self.X_train.shape, self.X_train_array.shape)
            print("Training labels size", self.y_train.shape, self.y_train_array.shape)
            print("Training Time Series size ", self.ts_train._xa.shape)
            print("Validation Time Series size ", self.ts_val._xa.shape)
            print("Test size ", self.X_test.shape, self.X_test_array.shape)
            print("Test labels size", self.y_test.shape, self.y_test_array.shape)
            print("Test Time Series size ", self.ts_test._xa.shape)


    def dataset_normalization(self, methods=["minmax"], scale_range=(0, 1)):
        """
        Normalize the data column according to the specify parameters.
        :param methods: list of strings: normalization methods to apply to each column.
                        Options: ['minmax', 'standard', None], Default = ["minmax"]
        :param scale_range: list of tuples: scale_range for each scaler. Default=(0,1) for each MinMax scaler
        :return: None
        """
        if self.verbose:
            print("Data normalization")
        if methods is not None and self.channels != len(methods):
            print("ERROR: You have to specify a scaling method for each feature")
            exit(1)

        self.X_scalers = {} # Initialize Dictionary of Scalers for Input Variables
        self.y_scalers = {} # Initialize Dictionary of Scalers for Target Variables
        if methods is not None: # Perform Normalization if a Normalization Method has been set
            self.normalization = methods # Set Normalization Method to the passed Method
            for i in range(self.channels): # Loop over the number of Training Features
                if self.normalization[i] is not None:
                    if self.normalization[i] == "standard": # Set Normalization Method of each element i in the Dictionary to Standardization
                        self.X_scalers[i] = StandardScaler()
                        self.y_scalers[i] = StandardScaler()
                        
                    elif self.normalization[i] == "minmax": # Set Normalization Method of each element i in the Dictionary to MinMax
                        self.X_scalers[i] = MinMaxScaler(scale_range)
                        self.y_scalers[i] = MinMaxScaler(scale_range)
                    #time series dataset
                    self.__ts_normalisation(method = self.normalization[i], range = scale_range) # Apply Normalization to Darts Timeseries Components
                    # window dataset    
                    self.X_train[:, :, i] = self.X_scalers[i].fit_transform(self.X_train[:, :, i]) # Fit and Transform the current Feature in the Training Set using Normalization Scaler
                    self.X_test[:, :, i] = self.X_scalers[i].transform(self.X_test[:, :, i]) # Transform the current Feature in the Test Set using Normalization Scaler
                    
                    for j, feature in enumerate(self.target_name): # Loop over Target Variables
                        if i == self.training_features.index(feature): # If Feature Index matches the Index of a Target Variable
                            self.y_train[:, :, j] = self.y_scalers[i].fit_transform(self.y_train[:, :, j]) # Fit and Transform the current Target Feature in the Training Set using Normalization Scaler
                            self.y_test[:, :, j] = self.y_scalers[i].transform(self.y_test[:, :, j]) # Transform the current Target Feature in the Test Set using Normalization Scaler
                    # unidimensional dataset
                    self.X_train_array = self.X_scalers[i].fit_transform(self.X_train_array) # Fit and Transform the Input Variable Training Set using Normalization Scaler
                    self.X_test_array = self.X_scalers[i].transform(self.X_test_array) # Transform the Input Variable Test Set using Normalization Scaler
                    self.y_train_array = self.y_scalers[i].fit_transform(self.y_train_array) # Fit and Transform the Target Variable Training Set using Normalization Scaler
                    self.y_test_array = self.y_scalers[i].transform(self.y_test_array) # Transform the Target Variable Test Set using Normalization Scaler


    def metadata_creation(self):
        """
        Add metadata to the dataset features. To implement according to the data format.
        :return: None
        """
        pass


    def __windowed_dataset(self, dataset):
        """

        :param dataset: np.array: features of the dataset
        :return: X: np.array: windowed version of the features
                 y: np.array: windowed version of the labels
        """
        dataset = tf.data.Dataset.from_tensor_slices(dataset) # Convert Numpy of Convert DataFrame Columns to Tensorflow Dataset
        dataset = dataset.window(self.input_window + self.output_window, stride=self.stride, shift=1,
                                 drop_remainder=True) # Create Windowed Dataset Consisting of Windows of Shape (N_Features, Input Window + Output Window)
        dataset = dataset.flat_map(lambda window: window.batch(self.input_window + self.output_window)) # Map TF Windowed Dataset to TF Dataset consisting of Batches of Shape (Input Window + Output Windows, N_Features)
        dataset = dataset.map(lambda window: (window[:-self.output_window], window[-self.output_window:])) # Map TF Dataset to TF Dataset consisting of Tensor Tuples of size ((Input Window, N_Features), (Output Window, N_Features))

        X, y = [], [] # Initialize X (Input Variables) and y (Target Variable)
        a = list(dataset.as_numpy_iterator()) # Convert TF Dataset to Numpy Array / Matrix containing Numpy Array Tuples
        for i, (A, b) in enumerate(a): # Iterate over Tuples in Dataset
            if i == len(a) - self.horizon: # Stop Iterating once Length of Dataset has been reached (since Horizon = 0, meaning in this case actually a Horizon of 1 is used and including the last value everything is predicted)
                break
            X.append(A) # Append Input Window to Input Variable List
            y.append(a[i + self.horizon][1]) # Append Output Window to Target Variable List
        X = np.array(X) # Turn Input Variable List to Numpy Array
        y = np.array(y) # Turn Output Variable List to Numpy Array
        indexes = [] # Initialize List of Indexes
        for feature in self.target_name: # Retrieve Index of each Target Variable
            indexes.append(self.training_features.index(feature))
        return X, y[:, :, indexes] # Return Input Variable Numpy Array and Target Variable Numpy Array only containing selected Target Variable


    def __ts_build_timeseries(self, df):
        """
        Generate darts.TimeSeries from DataFrame
        :param pd.DataFrame df: DataFrame with Values
        :return darts.TimeSeries ts: TimeSeries Data Array of DataFrame Values
        """
        #Compatible with Univariate Data Only
        #Setting the Daily Frequency of the Dataset

        df_col = self.target_name[0] # Retrieve Target Variable Column Name
        df[df_col] = df[df_col].astype(np.float32) # Convert to Target Column to Numpy Float32

        #Converting DataFrame to Series
        start_date = datetime.fromtimestamp(df['timestamp'].iloc[0]).strftime('%m/%d/%y') # Retrieve First Timestamp and Convert to Datetime String
        end_date = datetime.fromtimestamp(df['timestamp'].iloc[-1]).strftime('%m/%d/%y') # Retrieve Last Timestamp and Convert to Datetime String
        series_ts = pd.Series(data = df[df_col].values, index = pd.date_range(start_date, end_date, freq = 'D')) # Create Series from Values in Target Variable Column with Daily Dates as Index

        #Converting Series to DataFrame with DatetimeIndex
        df.index = pd.DatetimeIndex(np.hstack([series_ts.index[:-1],
                                               series_ts.index[-1:]]), freq='D')

        df_ts = df[self.target_name].copy()
        #Converting the DatatimeIndex DataFrame to timeseries 
        ts = TimeSeries.from_series(df_ts[df_col]) # Convert Target Variable Column with DateTimeIndex to Darts Timeseries

        return ts


    def get_ts_data(self, df):
        """
        Creation of Time Series Data Arrays for TFT Model
        :param   pd.DataFrame df: features of the dataset
        :return: darts.TimeSeries ts_train: Train Time Series array
                 darts.TimeSeries ts_test: Test Time Series array
                 darts.TimeSeries ts_val: Validation Time Series array
                 darts.TimeSeries train_cov: Training Set of Static Covariates
                 darts.TimeSeries cov: Future Static Covariates
                 darts.TimeSeries ts_ttrain: Training Time Series array before validation split    
        """
        ts = self.__ts_build_timeseries(df) # Build Darts Timeseries from DataFrame
        # Test Split
        split_value = (int(self.X_array.shape[0] * self.train_split_factor))  + self.add_split_value # Value on which to Split in Training and Test Set
        ts_ttrain, ts_test = ts.split_before(split_value) # Split Timeseries into Training and Test Set
        # Train and Validation Split
        ts_train, ts_val = ts_ttrain.split_before(0.8) # Split Training Timeseries into Training and Validation Set
        
        #Creating Covariates 
        train_cov, cov = self.__ts_covariates(ts=ts, ts_train = ts_train) # Retrieve Covariates

        return ts_train, ts_val, ts_test, train_cov, cov, ts_ttrain # Return Time Series Components
       

    def __ts_normalisation(self, method="minmax", range=(0, 1)):
        """
        Normalisation of Time Series Data Arrays
        :param string method:  method of scaling
               set range: range of scaling
        :return None
        """
        if method =='minmax': # Initialize Type of Scaler based on passed Method
            scale_method = MinMaxScaler(feature_range=range)
        else: 
            scale_method = StandardScaler()
        scaler = Scaler(scaler=scale_method) # Pass Initialized Scaler to Darts Scaler
        scaler.fit(self.ts_ttrain) # Fit Scaler to Training and Validation Set
        self.ts_ttrain = scaler.transform(self.ts_ttrain) # Transform Training and Validation Set using Normalization Scaler
        self.ts_train = scaler.transform(self.ts_train) # Transform Training Set using Normalization Scaler
        self.ts_val = scaler.transform(self.ts_val) # Transform Validation Set using Normalization Scaler
        self.ts_test = scaler.transform(self.ts_test) # Transform Test Set using Normalization Scaler
        covScaler = Scaler(scaler= scale_method) # Pass Initialized Scaler to Darts Scaler
        covScaler.fit(self.train_cov) # Fit Scaler to Training Covariates
        self.f_cov = covScaler.transform(self.cov) # Set f_cov to Transformed Covariates using Normalization Scaler
                

    def __ts_covariates(self, ts, ts_train):
        """
        :param darts.TimeSeries ts:  time indexed of entire DataSet 
               darts.TimeSeries ts_train: time series data array of training data
        :return: darts.TimeSeries train_cov: Training Set of Static Covariates
                 darts.TimeSeries cov: Static Covariates
                 
        """
        l = ts_train.n_timesteps
        cov = datetime_attribute_timeseries(ts, attribute="day", one_hot=False, add_length=l)
        cov = cov.stack(datetime_attribute_timeseries(cov.time_index, attribute="day_of_week"))
        cov = cov.stack(datetime_attribute_timeseries(cov.time_index, attribute="week"))
        cov = cov.stack(datetime_attribute_timeseries(cov.time_index, attribute="month"))
        cov = cov.stack(datetime_attribute_timeseries(cov.time_index, attribute="year"))
        cov = cov.stack(TimeSeries.from_times_and_values(
                                times=cov.time_index, 
                                values=np.arange(len(ts) + l), 
                                columns=["linear_increase"]))
        cov.add_holidays(country_code="US")
        cov = cov.astype(np.float32)

        #Test Split
        train_cov, test_cov = cov.split_before(self.train_split_factor)

        return train_cov, cov 
