
from util.dataset import DatasetInterface
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from binance.client import Client
#from binance import BinanceSocketManager
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler


class BinanceDataset(DatasetInterface):
    
    def __init__(self,  filename="", input_window=10, output_window=1, horizon=0, training_features=[], target_name=[],
                 train_split_factor=0.8, apiKey = "", apiSecurity =""):
        """
        Call to the parent constructor [DatasetInterface] and passing the required parameters:
            :param string: filename
            :param int: input_window
            :param int: output_window
            :param int: horizon
            :param array: training features
            :param array: target name
            :param float: train test split factor
        """
        super().__init__(filename, input_window, output_window, horizon, training_features, target_name, train_split_factor)
        """:param string: api key required for binance api access"""
        self.apiKey = apiKey
        """:param string: api security required for binance api access """
        self.apiSecurity = apiSecurity
        
   
    def create_frame(self, data):
        """
        Builds DataFrame to structure Data
        :param data: list: values returned from BinanceAPI
        :return df: pd.DataFrame: values structured with assigned Columns
        """
        df = pd.DataFrame(data)
        df = df.iloc[:,0:9]
        """columns returned by binance """
        df.columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumTrades']
        """float: open price"""
        df.Open = df.Open.astype(float)
        """float: close price"""
        df.Close = df.Close.astype(float)
        """float: high price"""
        df.High = df.High.astype(float)
        """float: low price"""
        df.Low= df.Low.astype(float)
        """float: volume traded"""
        df.Volume= df.Volume.astype(float)
        """float: asset volume quoted"""
        df.QuoteAssetVolume= df.QuoteAssetVolume.astype(float)
        """int: number of trades"""
        df.NumTrades = df.NumTrades.astype(int)
        """Date: open time"""
        df.OpenTime = pd.to_datetime(df.OpenTime, unit = 'ms')
        """Date: close time"""
        df.CloseTime = pd.to_datetime(df.CloseTime, unit = 'ms')
        return df
    
    
    def get_binance_data(self, sym, start_date, end_date):
        """
        Retrieves Data from Binance API 
        :param string sym: representsthe symbol of the cryptocurrency
        :param string start_data: the start date of collection
        :param string end date: the end data of collection 
        :return pd.DataFrame df_b: DataFrame consisting of columns  ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume']
        """
        try:
                client = Client(self.apiKey, self.apiSecurity)
                print("Logged in")
                interval = Client.KLINE_INTERVAL_1DAY
                klines = client.get_historical_klines(sym, interval, start_date, end_date)
                df = self.create_frame(klines)
                df_b = df.iloc[:, 0:5]
                return df_b
        except: 
                print('Invalid Login')
      
    
    def __save_crypto_df_to_csv(self, df):
        """
        Saves DataFrame to .csv file witin the saved_data folder
        :param pd.DataFrame df: DataFrame of Cryptocurrency Data
        :return None
        """
        df.columns = ['date', 'open', 'high', 'low', 'close', 'timestamp']
        save_name = self.data_path + self.data_file
        df.to_csv(save_name)

    
    def build_crypto_dataset(self, name, start_year, end_year, sym, start_date, end_date):
        """
        Combines Data from Investing.com [stored in Github Repository] with Binance API Data 
        :param string name: name of cryptocurrency pairing - the crypto valued against which currency [BTCUSD]
        :param string start_year: starting year of collection
        :param string end_year: ending year of collection
        :param string sym: symbol of cryptocurrency being collected
        :param string start_date: exact start date of collection
        :param string end_data: exact end date of collection
        :return None
        """
        # get cryptocurrency csv file from github (downloaded from: Investing.com)
        path ='https://github.com/katemurraay/TFT_Data/blob/main/'+name +'_'+start_year+'_'+ end_year +'.csv?raw=true'
        df_git = pd.read_csv(path)
        df_git.Date = pd.to_datetime(df_git.Date, format='%d/%m/%Y')
        # get binance data for cryptocurrency
        df_binance = self.get_binance_data(sym, start_date, end_date)
        #restructure datasets to match one another
        column_names = ["Date", "Open", "High", "Low", "Close"]
        df_git = df_git.reindex(columns=column_names)
        df_binance = df_binance.rename(columns = {'OpenTime':'Date'})
        #get index of binance start date in the github dataframe
        i = df_git.index[df_git['Date'] == df_binance['Date'].iloc[0]].tolist()
        #combine both datasets into a list
        df_a = df_git[:i[0]]
        df_b = df_binance.iloc[:, 0:5]
        list_combine = []
        list_combine.extend(df_a.values)
        list_combine.extend(df_b.values)
        #convert list to a dataframe
        df_combined = pd.DataFrame(list_combine, columns=['date', 'open', 'high', 'low', 'close'])
        df_combined['timestamp'] =  pd.to_datetime(df_combined['date']).view(int) // 10 ** 9
        self.__save_crypto_df_to_csv(df= df_combined)
 

    def inverse_transform_predictions(self, preds, X = 0, method="minmax", scale_range=(0, 1)):             
        """
        Inverts Scaling from the Data
        :param np.array or darts.TimeSeries preds: Scaled predictions from Model
               darts.TimeSeries X: The original value used to fit Scaler
               string method: the normalisation method used in scaling
               set scale_range: the range of values used in scaling
        :return np.array or darts.TimeSeries inverse_preds: Inverse scaled values
        """
    
        if isinstance(X, (int)): # If X is not a Darts Timeseries
            for i in range(self.channels): # Loop over Input Variables
                inverse_preds = self.y_scalers[i].inverse_transform(preds) # Inverse Transformations in Predictions using Saved Scaler Corresponding to the Variable Index
        else: # If X is a Darts Timeseries
            if method =='minmax': # Initialize Scaler based on the defined Method
                scale_method = MinMaxScaler(feature_range=scale_range)
            else: 
                scale_method = StandardScaler()
            
            scaler = Scaler(scaler=scale_method) # Initialize Darts Scaler using Initialized Scaler
            to_invert = TimeSeries.from_values(preds) # Convert Predictions to Darts Timeseries
            scaler.fit(X) # Fit Darts Scaler to Input Variables Darts Timeseries
            inv_preds = scaler.inverse_transform(to_invert) # Inverse Transform Predictions
            inverse_preds = inv_preds.pd_dataframe() # Convert Predictions from Darts Timeseries to DataFrame
            inverse_preds = np.array(inverse_preds.values).reshape(-1, 1) # Convert Values in DataFrame to Numpy Array
        return inverse_preds # Return Inverse Transformed Predictions
    

    def differenced_dataset(self, interval =1):
        """
        Builds Difference Dataset based on Interval
        :param int interval: represents interval of Difference [Default = 1]
        :return pd.DataFrame df: the orginial DataFrame from csv file
                pd.DataFrame diff_df: the DataFrame after Differencing
        """

        df = pd.read_csv(self.data_path + self.data_file) # Read in Original DataFrame
        df.date = pd.to_datetime(df.date, format='%d/%m/%Y') # Convert Date Column to a DateTime Object
        df = df.set_index('date') # Set Date Column as Index
        target = self.target_name[0] # Retrieve Name of Target Column
        diff = list() # Initialize List of Target Variable (= Close Price in this case) Differences
        time_steps = list() # Initialize List of Timestamps
        for i in range(interval, len(df)): # Loop through range from Interval (=1) to Length DataFrame
          value = df[target][i] - df[target][i - 1] # Retrieve Difference of each Timestep by subtracting the Value of Target Variable in the previous Timestep of the Value of the current Timestep
          time_steps.append(df['timestamp'][i]) # Add Timestamp of current Timestep to List of Timestamps
          diff.append(value) # Add Difference of current Timestep to List of Differences
        diff_df = pd.DataFrame(diff, columns=[target]) # Initialize DataFrame containing column with Differences of Target Variable at each Time Step
        diff_df['timestamp'] = time_steps # Add Timestamps as a Column to DataFrame containing Differences
        return df, diff_df # Return Original Dataframe and Differenced Data Frame

    
    def inverse_differenced_dataset(self, df, diff_vals, l = 0, df_start = 0):
        """
        Inverses the Difference on a Dataset
        :param  pd.DataFrame df: the orginial DataFrame from csv file
                list diff_vals:  List of Differenced Values
        :return np.array inverted_values: Array of values with Difference removed
        """
        invert = list() # Initialize List for Inverted Values
        target = self.target_name[0]  # Save Name of Target Variable to Variable
        if df_start == 0:
            if l == 0: # Define from which Point to retrieve non-differenced Values
                df_start = len(df) - len(diff_vals) -1
            else:
                df_start = len(df) - l -1
    
        for i in range(len(diff_vals)): # Loop through Differenced Predictions
            value =  diff_vals[i] + df[target][df_start + i] # Remove Differencing from Predictions by adding previous Actual Value
            invert.append(value) # Add Prediction without Differencing to Inverted Values List
        inverted_values = np.array(invert) # Transformed Inverted Values List to Numpy Array
        return inverted_values # Return Numpy Array containing Non-Differenced
        
        
    def scale_predictions(self, preds, X = 0, method="minmax", scale_range=(0, 1)):             
        """
        Scale Predictions
        :param np.array or darts.TimeSeries preds: Scaled predictions from Model
               darts.TimeSeries X: The original value used to fit Scaler
               string method: the normalisation method used in scaling
               set scale_range: the range of values used in scaling
        :return np.array or darts.TimeSeries scaled_preds: Inverse scaled values
        """   
        if isinstance(preds, (np.ndarray)): # Check if preds is not a Darts Timeseries
            for i in range(self.channels):
                scaled_preds = self.y_scalers[i].transform(preds) # Normalize Predictions using Scaler that was used on the Actual Data
        else: # If preds is a Darts Timeseries
            if method =='minmax': # Initialize Scaler based on the passed Method
                scale_method = MinMaxScaler(feature_range=scale_range)
            else: 
                scale_method = StandardScaler()
            
            scaler = Scaler(scaler=scale_method) # Initialize Darts Scaler using Initialized Scaler
            scaler.fit(X) # Fit Darts Scaler to Input Data
            scaled_preds = scaler.transform(preds) # Transform Predictions using Darts Scaler
        return scaled_preds # Return Normalized Predictions
       
