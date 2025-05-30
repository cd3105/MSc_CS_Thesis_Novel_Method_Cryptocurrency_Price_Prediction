"""
Interface of a predictive DL model with shared functionalities
Inherits from ModelInterface class
"""

from models.model_interface import ModelInterface
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from util import custom_keras
import tensorflow as tf
import talos
import torch


class ModelInterfaceDL(ModelInterface):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super(ModelInterfaceDL, self).__init__(name)

        self.count_save = 0
        """int: counter of the model saving"""
        self.best_val_loss = np.inf
        """float: best validation found so far"""

    def fit(self):
        """
        Training of the model
        :return: None
        """
        save_check = custom_keras.CustomSaveCheckpoint(self) # For Saving the Model
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.p['patience']) # Initializes Early Stopping Object which ensures stopping Training if Loss doesn't improve in certain number of steps
        history = self.temp_model.fit(self.ds.X_train, self.ds.y_train, epochs=self.p['epochs'],
                                      batch_size=self.p['batch_size'], validation_split=0.2, verbose=2, callbacks=[es, save_check]) # Trains temp_model on Training Data
        
        self.model = save_check.dnn.model # Set model to Best Version of Model saved by CustomSaveCheckpoint

        return history, self.model # Return History of Loss across Epochs and Best Model

    def predict(self, X):
        """
        Inference step on the samples X
        :param X: np.array: Input samples to predict
        :return: np.array: predictions: Predictions of the samples X
        """
        if self.model is None:
            print("ERROR: the model needs to be trained before predict")
            return
        predictions = self.model.predict(X) # Perform Predictions on Input Variables
        return predictions # Return Predictions

    def tune(self, X, y):
        """
        Tune the models with new available samples (X, y)
        :param X: nparray: Training features
        :param y: nparray: Training labels
        :return: None
        """
        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.p['patience'])

        history = self.temp_model.fit(X, y, epochs=self.p['epochs'], batch_size=self.p['batch_size'],
                                      validation_split=0.2, verbose=2, callbacks=[es, save_check])

        self.model = save_check.dnn.model

    def evaluate(self):
        """
        Evaluate the model on the training set ds.X_train
        :return: np.array: predictions: predictions of the trained model on the ds.X_train set
        """
        return self.predict(self.ds.X_train)

    def hyperparametrization(self):
        """
        Search the best parameter configuration using talos
        :return: None
        """
        tf.keras.backend.clear_session()
        t = talos.Scan(x=self.ds.X_train,
                   y=self.ds.y_train,
                   model=self.__talos_model,
                   experiment_name='talos/' + self.name,
                   params=self.parameter_list,
                   clear_session=True,
                   print_params=True, round_limit=100)
      
        a = talos.Analyze(t)
        a_table = a.table('val_loss', [], sort_by= 'val_mse', ascending= True)
        print('BEST PARAMS \n{}'.format(a_table.iloc[0]))

    def __talos_model(self, X_train, y_train, x_val, y_val, p):
        """
        Custom fuction for talos optimization
        :return: fit function
        """
        self.create_model()
        return self.fit()

    def save_model(self):
        """
        Save the model into a file in self.saved_models directory
        :return: boolean: 1 if saving operating is successful, 0 otherwise
        """
        if self.model is None:
            print("ERROR: the model must be available before saving it")
            return
        self.model.save(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.keras')
        self.count_save += 1
        return 1

    def load_model(self):
        """
        Load the model from a file
        :return: boolean: 1 if loading operating is successful, 0 otherwise
        """
        count_save = self.count_save - 1
        self.temp_model = tf.keras.models.load_model(self.model_path + self.name + str(count_save).zfill(4) + '_model.keras')
        return 1

    def training(self, p, X_test):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True) # Timers to record Start and End of Operation
        repetitions = 1 # Number of Repetitions
        timings=np.zeros((repetitions,1)) # For Recording Times
        
        for _ in range(10):
            self.p = p # Set Hyperparameters
            self.create_model() # Create Model
            
        with torch.no_grad(): # Seems to have no purpose as fitting is done by Keras / TF
            for rep in range(repetitions): # Loop over Number of Repetitions
                starter.record() # Start Timing
                _, train_model = self.fit() # Fit Model and Save Model
                ender.record() # End Timing
                torch.cuda.synchronize() # Ensures GPU Operations have been completed before Ending Timing
                curr_time = starter.elapsed_time(ender)/1000 # Captures and Converts Recorded Time to Seconds
                timings[rep] = curr_time # Saves Time

        self.train_time = np.sum(timings) / repetitions # Calculate Average Training Time
        for _ in range(10):
            to_predict = X_test # Set to_predict to Input Variable Test Set
        # MEASURE PERFORMANCE
        with torch.no_grad(): # Seems to have no purpose as fitting is done by Keras / TF
            for rep in range(repetitions): # Loop over Number of Repetitions
                starter.record() # Start Timing
                predictions = self.predict(to_predict) # Perform Predictions using Input Variable Test Set
                ender.record() # End Timing
                torch.cuda.synchronize() # Ensures GPU Operations have been completed before Ending Timing
                curr_time = starter.elapsed_time(ender) # Captures Recorded Time
                timings[rep] = curr_time # Saves Time
        mean_time = np.sum(timings) / repetitions # Average Test Time
        self.inference_time = mean_time / len(to_predict) # Average Test Time per Instance
        
        return predictions, train_model # Return Predictions and Trained Model

