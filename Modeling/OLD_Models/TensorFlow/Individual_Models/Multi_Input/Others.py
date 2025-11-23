import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, AdamW, Adadelta, Adagrad, Adamax, Nadam, RMSprop, SGD


class MI_TCN_TF():
    def __init__(self,
                 dataset,):
        self.model = None
        self.dataset = dataset


    def create_model(self,
                     verbose=False):
        input_shape = self.dataset.price_data_X_train.shape[1:]
        output_size = self.dataset.y_train.shape[2]

        

        self.model = Sequential([
            tf.keras.layers.Conv1D(filters=32, 
                                   kernel_size=3,
                                   padding="causal", 
                                   dilation_rate=1,
                                   activation='relu',
                                   input_shape=input_shape), 
            tf.keras.layers.Conv1D(filters=32, 
                                   kernel_size=3,
                                   padding="causal", 
                                   dilation_rate=1,
                                   activation='relu',
                                   input_shape=input_shape),
            tf.keras.layers.Conv1D(filters=32, 
                                   kernel_size=3,
                                   padding="causal", 
                                   dilation_rate=1,
                                   activation='relu',
                                   input_shape=input_shape),
            tf.keras.layers.Conv1D(filters=32, 
                                   kernel_size=3,
                                   padding="causal", 
                                   dilation_rate=1,
                                   activation='relu',
                                   input_shape=input_shape),
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(16, 
                                  activation='relu'), 
            tf.keras.layers.Dense(output_size), 
        ])
        
        if verbose:
            self.model.summary()


    def training(self,
                 
                 n_epochs,
                 batch_size,
                 optimizer,
                 learning_rate,
                 weight_decay,
                 momentum,
                 
                 optimization_mode,
                 verbose):
        
        self.create_model() 

        if optimizer == 'Adam':
            optimizer = Adam(learning_rate=learning_rate,
                             decay=weight_decay)
        elif optimizer == 'AdamW':
            optimizer = AdamW(learning_rate=learning_rate,
                              decay=weight_decay)
        elif optimizer == 'Adadelta':
            optimizer = Adadelta(learning_rate=learning_rate,
                                 decay=weight_decay)
        elif optimizer == 'Adagrad':
            optimizer = Adagrad(learning_rate=learning_rate,
                                decay=weight_decay)
        elif optimizer == 'Adamax':
            optimizer = Adamax(learning_rate=learning_rate,
                               decay=weight_decay)
        elif optimizer == 'Nadam':
            optimizer = Nadam(learning_rate=learning_rate,
                              decay=weight_decay)
        elif optimizer == 'RMSprop':
            optimizer = RMSprop(learning_rate=learning_rate,
                                decay=weight_decay,
                                momentum=momentum)
        else:
            optimizer = SGD(learning_rate=learning_rate,
                            decay=weight_decay,
                            momentum=momentum)

        self.model.compile(loss='mean_squared_error',
                           optimizer=optimizer,
                           metrics=["mse", 
                                    "mae"])


        if optimization_mode:
            history = self.model.fit(self.dataset.price_data_X_train, 
                                     self.dataset.y_train, 
                                     epochs=n_epochs,  
                                     validation_data=(self.dataset.price_data_X_val,
                                                      self.dataset.y_val),
                                     batch_size=batch_size, 
                                     shuffle=False,
                                     verbose=verbose)

        else:
            history = self.model.fit(np.concatenate((self.dataset.price_data_X_train,
                                                     self.dataset.price_data_X_val),
                                                     axis=0), 
                                     np.concatenate((self.dataset.y_train,
                                                     self.dataset.y_val),
                                                     axis=0),
                                     epochs=n_epochs,  
                                     validation_data=(self.dataset.price_data_X_test,
                                                      self.dataset.y_test),
                                     batch_size=batch_size, 
                                     shuffle=False,
                                     verbose=verbose)
    

        learning_curves = {'train_mse':history.history['loss'],
                           'train_mae':history.history['mae'],
                           'val_mse':history.history['val_loss'],
                           'val_mae':history.history['val_mae'],}    

        return learning_curves

    def predict(self,
                X):
        return self.model.predict(X)
    