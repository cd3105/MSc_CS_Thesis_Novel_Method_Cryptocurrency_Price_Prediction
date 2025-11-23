import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, AdamW, Adadelta, Adagrad, Adamax, Nadam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Sequential


class MI_GRU_TF():
    def __init__(self,
                 dataset,):
                #  RNN_units=[75],
                #  Dense_units=[100],
                #  GRU_activations=['relu'],
                #  Dense_activations=['relu']):
        self.model = None
        self.dataset = dataset

    def create_model(self,
                     verbose=False):
                    #  units_GRU=[64, 128],
                    #  units_Dense=[128, 64]):
        input_shape = self.dataset.price_data_X_train.shape[1:]
        output_size = self.dataset.y_train.shape[2]
    
        # self.model = Sequential()

        # if len(units_GRU) == 1:
        #     self.model.add(GRU(units=u,
        #                         input_shape=input_shape))
        # else:
        #     for i, u in enumerate(units_GRU):
        #         if i == 0:
        #             self.model.add(GRU(units=u,
        #                                 return_sequences=True,
        #                                 input_shape=input_shape,))                    
        #         elif i == (len(units_GRU)-1):
        #             self.model.add(GRU(units=u))
        #         else:
        #             self.model.add(GRU(units=u,
        #                                 return_sequences=True))

        # for i, u in enumerate(units_Dense):
        #     self.model.add(Dense(units=u))
        
        # self.model.add(Dense(units=output_size))
        # self.model.compile(optimizer=Adam(learning_rate=1e-3),
        #                    loss='mse')
        
        self.model = Sequential([
            tf.keras.layers.GRU(75, 
                                activation='relu', 
                                input_shape=input_shape), # Set up GRU Layer with selected Hyper Parameters
            tf.keras.layers.Dense(100, 
                                  activation='relu',
                                  kernel_initializer='he_normal'), # Set up Dense Layer after GRU Layer with selected Hyper Parameters
            tf.keras.layers.Dense(output_size), # Set up Ouput Dense Layer resulting in an output matching the dimensions of y
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
                 #patience,
                 
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
        
        # early_stop = EarlyStopping(monitor='val_loss',
        #                            patience=patience)

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
                                     # callbacks=[early_stop])

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
                                     # callbacks=[early_stop])
    

        training_metrics = {'train_mse':history.history['loss'],
                            'train_mae':history.history['mae'],
                            'val_mse':history.history['val_loss'],
                            'val_mae':history.history['val_mae'],}    

        return training_metrics

    def predict(self,
                X):
        return self.model.predict(X)

# ---------------------------------------------------------------

class MI_LSTM_TF():
    def __init__(self,
                 dataset,):
        self.model = None
        self.dataset = dataset
    
    def create_model(self,
                     verbose=False):
        input_shape = self.dataset.price_data_X_train.shape[1:]
        output_size = self.dataset.y_train.shape[2]
        
        self.model = Sequential([
            tf.keras.layers.LSTM(75, 
                                 activation='relu', 
                                 input_shape=input_shape), 
            tf.keras.layers.Dense(100, 
                                  activation='relu',
                                  kernel_initializer='he_normal'),
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

# ---------------------------------------------------------------

class MI_BiGRU_TF():
    def __init__(self,
                 dataset,):
        self.model = None
        self.dataset = dataset
    
    def create_model(self,
                     verbose=False):
        input_shape = self.dataset.price_data_X_train.shape[1:]
        output_size = self.dataset.y_train.shape[2]
    
        self.model = Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(75, 
                                                              activation='relu', 
                                                              input_shape=input_shape)), 
            tf.keras.layers.Dense(100, 
                                  activation='relu',
                                  kernel_initializer='he_normal'),
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

# ---------------------------------------------------------------

class MI_BiLSTM_TF():
    def __init__(self,
                 dataset,):
        self.model = None
        self.dataset = dataset
    
    def create_model(self,
                     verbose=False):
        input_shape = self.dataset.price_data_X_train.shape[1:]
        output_size = self.dataset.y_train.shape[2]
    
        self.model = Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(75, 
                                                               activation='relu', 
                                                               input_shape=input_shape)),
            tf.keras.layers.Dense(100, 
                                  activation='relu',
                                  kernel_initializer='he_normal'), 
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
