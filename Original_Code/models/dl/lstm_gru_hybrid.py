"""
LSTM & GRU hybrid model based on work of Patel et al. (2020)
Inherits from ModelInterfaceDL class
"""

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from models.dl.model_interface_dl import ModelInterfaceDL


class LSTM_GRU(ModelInterfaceDL):

    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)
        self.parameter_list = {
                                'first_lstm_dim': [30, 50, 75],
                                'lstm_activation': ['relu', 'tanh'],
                                'first_dropout_rate': [0.0, 0.05, 0.1],
                                'second_lstm_dim': [30, 50, 75],
                                'first_dense_dim': [16, 32, 64],
                               'dense_activation': ['relu', 'elu', 'selu', 'tanh'],
                               'dense_kernel_init': ['he_normal', 'glorot_uniform'],
                               'gru_dim':[30,50,75],
                                'gru_activation': ['relu', 'tanh'],
                               'second_dropout_rate': [0.0, 0.05, 0.1],
                                'second_dense_dim': [16, 32, 64],
                               'batch_size': [256, 512, 1024],
                               'epochs': [2000],
                               'patience': [50],
                               'optimizer': ['adam', 'nadam', 'rmsprop'],
                               'lr': [1E-3, 1E-4, 1E-5],
                               'momentum': [0.9, 0.99],
                               'decay': [1E-3, 1E-4, 1E-5],
                               }
        """dict: Dictionary of hyperparameters search space"""

        self.p = {
                'first_lstm_dim': 50,
                'lstm_activation': 'relu',
                'first_dropout_rate': 0.05, 
                'second_lstm_dim':  50,
                'first_dense_dim':  32,
                'dense_activation': 'relu',
                'dense_kernel_init': 'he_normal',
                'gru_dim':50,
                'gru_activation':'relu',
                'second_dropout_rate':  0.05,
                'second_dense_dim':  32,
                'batch_size': 256,
                'epochs': 1000,
                'patience': 50,
                'optimizer': 'adam',
                'lr': 1E-4,
                'momentum': 0.9,
                'decay': 1E-4,
                  }
        """dict: Dictionary of hyperparameter configuration of the model"""

    def create_model(self):
        """
        Create an instance of the model. This function contains the definition and the library of$
        :return: None
        """
        

        input = tf.keras.Input(shape=(None, self.ds.X_train.shape[2])) # Shape of each Input


        #LSTM 
        x = tf.keras.layers.LSTM(int(self.p['first_lstm_dim']), 
                                 activation=self.p['lstm_activation'], 
                                 return_sequences=True)(input) # Set up First LSTM Layer with selected Hyper Parameters
        x = tf.keras.layers.Dropout(rate=self.p['first_dropout_rate'])(x) # Set up Dropout Layer with selected Hyper Parameters
        x = tf.keras.layers.LSTM(int(self.p['second_lstm_dim']), 
                                 activation=self.p['lstm_activation'])(x) # Set up Second LSTM Layer with selected Hyper Parameters
        lstm_model = tf.keras.layers.Dense(int(self.p['first_dense_dim']), 
                                           activation=self.p['dense_activation'])(x) # Set up Dense Layer with selected Hyper Parameters


        #GRU
        y = tf.keras.layers.GRU(int(self.p['gru_dim']), 
                                activation=self.p['gru_activation'])(input) # Set up GRU Layer with selected Hyper Parameters
        y = tf.keras.layers.Dropout(rate=self.p['second_dropout_rate'])(y) # Set up Dropout Layer with selected Hyper Parameters
        gru_model = tf.keras.layers.Dense(int(self.p['second_dense_dim']), 
                                          activation=self.p['dense_activation'])(y) # Set up Dense Layer with selected Hyper Parameters


        concatenated = tf.keras.layers.concatenate([lstm_model, gru_model]) # Combine LSTM & GRU Model via Concatenation Layer
        output = tf.keras.layers.Dense(self.ds.y_train.shape[2])(concatenated)  # Set up Output Dense Layer
        self.temp_model = tf.keras.Model(inputs=input, outputs=output) # Combine Everything into Model


        if self.p['optimizer'] == 'adam': # Setup Optimizer corresponding to the passed Hyperparameters
            opt = Adam(learning_rate=self.p['lr'], decay=self.p['decay'])
        elif self.p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=self.p['lr'])
        elif self.p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=self.p['lr'])
        elif self.p['optimizer'] == 'sgd':
                 opt = SGD(learning_rate=self.p['lr'], momentum=self.p['momentum'])
        self.temp_model.compile(loss='mean_squared_error',
                                optimizer=opt,
                                metrics=["mse", "mae"]) # Compile Model using set Optimizer and specified Loss Functions

