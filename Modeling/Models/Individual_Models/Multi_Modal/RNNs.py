import torch.nn as nn
import numpy as np


class MM_GRU(nn.Module):
    def __init__(
            self, 
            dataset,
            selected_IMF_idx,
    ):
        super(MM_GRU, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx
        
        selected_IMF_X_train = np.expand_dims(
            dataset.close_price_data_X_train[...,selected_IMF_idx], 
            axis=-1,
        )
        selected_IMF_y_train = np.expand_dims(
            dataset.y_train[...,selected_IMF_idx], 
            axis=-1,
        )

        input_size = selected_IMF_X_train.shape[2]
        input_window_size = selected_IMF_X_train.shape[1]

        output_size = selected_IMF_y_train.shape[2]
        output_window_size = selected_IMF_y_train.shape[1]

        self.gru = nn.Sequential(
            nn.GRU(
                input_size=input_size,
                hidden_size=64,
                batch_first=True,
            ),
        )

        self.fc = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(
                in_features=64, # 64 * input_window_size
                out_features=32,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=32,
                out_features=output_size * output_window_size,
            )
        )

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        x = close_price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

        out, _ = self.gru(x)  
        out = self.fc(out[:,-1,:]) # self.fc(out)
        
        return out
        

class HPT_MM_GRU(nn.Module):
    def __init__(
            self, 
            dataset,
            selected_IMF_idx,
            
            n_layers=1,
            dims=[75],
            activation='linear',
            
            dense_n_layers=1,
            dense_dims=[100],
            dense_activations=['relu'],
    ):
        super(HPT_MM_GRU, self).__init__()
        
        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(
            dataset.close_price_data_X_train[...,selected_IMF_idx], 
            axis=-1,
        )
        selected_IMF_y_train = np.expand_dims(
            dataset.y_train[...,selected_IMF_idx], 
            axis=-1,
        )

        self.grus = nn.ModuleList()
        
        current_input_size = selected_IMF_X_train.shape[2]
        input_window_size = selected_IMF_X_train.shape[1]

        output_size = selected_IMF_y_train.shape[2]
        output_window_size = selected_IMF_y_train.shape[1]

        for l in range(n_layers):
            self.grus.append(
                nn.GRU(
                    input_size=current_input_size, 
                    hidden_size=dims[l], 
                    batch_first=True,
                )
            )
            current_input_size = dims[l]
        
        if activation == 'elu':
            self.gru_activation_func = nn.ELU()
        elif activation == 'gelu':
            self.gru_activation_func = nn.GELU()
        elif activation == 'leaky_relu':
            self.gru_activation_func = nn.LeakyReLU()
        elif activation == 'relu':
            self.gru_activation_func = nn.ReLU()
        elif activation == 'selu':
            self.gru_activation_func = nn.SELU()
        elif activation == 'sigmoid':
            self.gru_activation_func = nn.Sigmoid()
        elif activation == 'tanh':
            self.gru_activation_func = nn.Tanh()
        else:
            self.gru_activation_func = nn.Identity()

        self.fcs = nn.ModuleList()
        self.fc_activation_funcs = nn.ModuleList()

        for l in range(dense_n_layers):
            self.fcs.append(
                nn.Linear(
                    in_features=current_input_size, 
                    out_features=dense_dims[l],
                )
            )
            current_input_size = dense_dims[l]

            if dense_activations[l] == 'elu':
                self.fc_activation_funcs.append(nn.ELU())
            elif dense_activations[l] == 'gelu':
                self.fc_activation_funcs.append(nn.GELU())
            elif dense_activations[l] == 'leaky_relu':
                self.fc_activation_funcs.append(nn.LeakyReLU())
            elif dense_activations[l] == 'relu':
                self.fc_activation_funcs.append(nn.ReLU())
            elif dense_activations[l] == 'selu':
                self.fc_activation_funcs.append(nn.SELU())
            elif dense_activations[l] == 'sigmoid':
                self.fc_activation_funcs.append(nn.Sigmoid())
            elif dense_activations[l] == 'tanh':
                self.fc_activation_funcs.append(nn.Tanh())
            else:
                self.fc_activation_funcs.append(nn.Identity())

        self.output_fc = nn.Linear(
            in_features=current_input_size, 
            out_features=output_size * output_window_size,
        )

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        x = close_price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

        for gru in self.grus:
            out, _ = gru(x) 
            x = out

        out = x[:, -1, :]

        out = self.gru_activation_func(out)

        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)

        out = self.output_fc(out)
        
        return out


# ---------------------------------------------------


class MM_BiGRU(nn.Module):
    def __init__(
            self, 
            dataset,
            selected_IMF_idx,
    ):
        super(MM_BiGRU, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(
            dataset.close_price_data_X_train[...,selected_IMF_idx], 
            axis=-1,
        )
        selected_IMF_y_train = np.expand_dims(
            dataset.y_train[...,selected_IMF_idx], 
            axis=-1,
        )

        input_size = selected_IMF_X_train.shape[2]
        input_window_size = selected_IMF_X_train.shape[1]

        output_size = selected_IMF_y_train.shape[2]
        output_window_size = selected_IMF_y_train.shape[1]

        self.bigru = nn.Sequential(
            nn.GRU(
                input_size=input_size,
                hidden_size=75,
                batch_first=True,
                bidirectional=True,
            ),
            nn.Dropout(p=0.25),
        )

        self.fc = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(
                in_features=2 * 75, # 2 * 75 * output_window_size
                out_features=100,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=100, 
                out_features=output_size * output_window_size,
            )
        )

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        x = close_price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

        out, _ = self.bigru(x)  
        out = self.fc(out[:,-1,:]) # self.fc(out)
        
        return out


class HPT_MM_BiGRU(nn.Module):
    def __init__(
            self, 
            dataset,
            selected_IMF_idx,
            
            n_layers=1,
            dims=[75],
            activation='linear',
            
            dense_n_layers=1,
            dense_dims=[100],
            dense_activations=['relu'],
    ):
        super(HPT_MM_BiGRU, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(
            dataset.close_price_data_X_train[...,selected_IMF_idx], 
            axis=-1,
        )
        selected_IMF_y_train = np.expand_dims(
            dataset.y_train[...,selected_IMF_idx], 
            axis=-1,
        )

        self.bigrus = nn.ModuleList()

        current_input_size = selected_IMF_X_train.shape[2]
        input_window_size = selected_IMF_X_train.shape[1]

        output_size = selected_IMF_y_train.shape[2]
        output_window_size = selected_IMF_y_train.shape[1]

        for l in range(n_layers):
            self.bigrus.append(
                nn.GRU(
                    input_size=current_input_size, 
                    hidden_size=dims[l], 
                    batch_first=True,
                    bidirectional=True,
                )
            )
            current_input_size = 2 * dims[l]
        
        if activation == 'elu':
            self.bigru_activation_func = nn.ELU()
        elif activation == 'gelu':
            self.bigru_activation_func = nn.GELU()
        elif activation == 'leaky_relu':
            self.bigru_activation_func = nn.LeakyReLU()
        elif activation == 'relu':
            self.bigru_activation_func = nn.ReLU()
        elif activation == 'selu':
            self.bigru_activation_func = nn.SELU()
        elif activation == 'sigmoid':
            self.bigru_activation_func = nn.Sigmoid()
        elif activation == 'tanh':
            self.bigru_activation_func = nn.Tanh()
        else:
            self.bigru_activation_func = nn.Identity()

        self.fcs = nn.ModuleList()
        self.fc_activation_funcs = nn.ModuleList()

        for l in range(dense_n_layers):
            self.fcs.append(
                nn.Linear(
                    in_features=current_input_size, 
                    out_features=dense_dims[l],
                )
            )
            current_input_size = dense_dims[l]

            if dense_activations[l] == 'elu':
                self.fc_activation_funcs.append(nn.ELU())
            elif dense_activations[l] == 'gelu':
                self.fc_activation_funcs.append(nn.GELU())
            elif dense_activations[l] == 'leaky_relu':
                self.fc_activation_funcs.append(nn.LeakyReLU())
            elif dense_activations[l] == 'relu':
                self.fc_activation_funcs.append(nn.ReLU())
            elif dense_activations[l] == 'selu':
                self.fc_activation_funcs.append(nn.SELU())
            elif dense_activations[l] == 'sigmoid':
                self.fc_activation_funcs.append(nn.Sigmoid())
            elif dense_activations[l] == 'tanh':
                self.fc_activation_funcs.append(nn.Tanh())
            else:
                self.fc_activation_funcs.append(nn.Identity())

        self.output_fc = nn.Linear(
            in_features=current_input_size, 
            out_features=output_size * output_window_size,
        )

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        x = close_price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

        for bigru in self.bigrus:
            out, _ = bigru(x) 
            x = out

        out = x[:, -1, :]

        out = self.bigru_activation_func(out)

        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)

        out = self.output_fc(out)
        
        return out


# ---------------------------------------------------


class MM_LSTM(nn.Module):
    def __init__(
            self, 
            dataset,
            selected_IMF_idx,
    ):
        super(MM_LSTM, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(
            dataset.close_price_data_X_train[...,selected_IMF_idx], 
            axis=-1,
        )
        selected_IMF_y_train = np.expand_dims(
            dataset.y_train[...,selected_IMF_idx], 
            axis=-1,
        )

        input_size = selected_IMF_X_train.shape[2]
        input_window_size = selected_IMF_X_train.shape[1]

        output_size = selected_IMF_y_train.shape[2]
        output_window_size = selected_IMF_y_train.shape[1]

        self.lstm = nn.Sequential(
            nn.LSTM(
                input_size=input_size,
                hidden_size=75,
                batch_first=True,
            ),
        )
        
        self.fc_1 = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(
                in_features=75, # 75 * output_window_size
                out_features=100,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=100, 
                out_features=output_size * output_window_size,
            )
        )  

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        x = close_price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

        out, _ = self.lstm(x)  
        out = self.fc(out[:, -1, :]) # self.fc(out)
        
        return out
    

class HPT_MM_LSTM(nn.Module):
    def __init__(
            self, 
            dataset,
            selected_IMF_idx,
            
            n_layers=1,
            dims=[75],
            activation='linear',
            
            dense_n_layers=1,
            dense_dims=[100],
            dense_activations=['relu'],
    ):
        super(HPT_MM_LSTM, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(
            dataset.close_price_data_X_train[...,selected_IMF_idx], 
            axis=-1,
        )
        selected_IMF_y_train = np.expand_dims(
            dataset.y_train[...,selected_IMF_idx], 
            axis=-1,
        )

        self.lstms = nn.ModuleList()

        current_input_size = selected_IMF_X_train.shape[2]
        input_window_size = selected_IMF_X_train.shape[1]

        output_size = selected_IMF_y_train.shape[2]
        output_window_size = selected_IMF_y_train.shape[1]

        for l in range(n_layers):
            self.lstms.append(
                nn.LSTM(
                    input_size=current_input_size, 
                    hidden_size=dims[l], 
                    batch_first=True,
                )
            )
            current_input_size = dims[l]
        
        if activation == 'elu':
            self.lstm_activation_func = nn.ELU()
        elif activation == 'gelu':
            self.lstm_activation_func = nn.GELU()
        elif activation == 'leaky_relu':
            self.lstm_activation_func = nn.LeakyReLU()
        elif activation == 'relu':
            self.lstm_activation_func = nn.ReLU()
        elif activation == 'selu':
            self.lstm_activation_func = nn.SELU()
        elif activation == 'sigmoid':
            self.lstm_activation_func = nn.Sigmoid()
        elif activation == 'tanh':
            self.lstm_activation_func = nn.Tanh()
        else:
            self.lstm_activation_func = nn.Identity()

        self.fcs = nn.ModuleList()
        self.fc_activation_funcs = nn.ModuleList()

        for l in range(dense_n_layers):
            self.fcs.append(
                nn.Linear(
                    in_features=current_input_size, 
                    out_features=dense_dims[l],
                )
            )
            current_input_size = dense_dims[l]

            if dense_activations[l] == 'elu':
                self.fc_activation_funcs.append(nn.ELU())
            elif dense_activations[l] == 'gelu':
                self.fc_activation_funcs.append(nn.GELU())
            elif dense_activations[l] == 'leaky_relu':
                self.fc_activation_funcs.append(nn.LeakyReLU())
            elif dense_activations[l] == 'relu':
                self.fc_activation_funcs.append(nn.ReLU())
            elif dense_activations[l] == 'selu':
                self.fc_activation_funcs.append(nn.SELU())
            elif dense_activations[l] == 'sigmoid':
                self.fc_activation_funcs.append(nn.Sigmoid())
            elif dense_activations[l] == 'tanh':
                self.fc_activation_funcs.append(nn.Tanh())
            else:
                self.fc_activation_funcs.append(nn.Identity())

        self.output_fc = nn.Linear(
            in_features=current_input_size, 
            out_features=output_size * output_window_size,
        )
        
    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        x = close_price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

        for lstm in self.lstms:
            out, _ = lstm(x) 
            x = out

        out = x[:, -1, :] # x

        out = self.lstm_activation_func(out)

        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)

        out = self.output_fc(out)
        
        return out


# ---------------------------------------------------


class MM_BiLSTM(nn.Module):
    def __init__(
            self, 
            dataset,
            selected_IMF_idx,
    ):
        super(MM_BiLSTM, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(
            dataset.close_price_data_X_train[...,selected_IMF_idx], 
            axis=-1,
        )
        selected_IMF_y_train = np.expand_dims(
            dataset.y_train[...,selected_IMF_idx], 
            axis=-1,
        )

        input_size = selected_IMF_X_train.shape[2]
        input_window_size = selected_IMF_X_train.shape[1]

        output_size = selected_IMF_y_train.shape[2]
        output_window_size = selected_IMF_y_train.shape[1]

        self.lstm = nn.Sequential(
            nn.LSTM(
                input_size=input_size,
                hidden_size=75,
                batch_first=True,
                bidirectional=True,
            ),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=2 * 75, # 2 * 75 * input_window_size
                out_features=100,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=100, 
                out_features=output_size * output_window_size,
            ),
        )

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        x = close_price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

        out, _ = self.lstm(x)  
        out = self.fc(out[:, -1, :]) # self.fc(out)
        
        return out
    

class HPT_MM_BiLSTM(nn.Module):
    def __init__(
            self, 
            dataset,
            selected_IMF_idx,
            
            n_layers=1,
            dims=[75],
            activation='linear',
            
            dense_n_layers=1,
            dense_dims=[100],
            dense_activations=['relu'],
    ):
        super(HPT_MM_BiLSTM, self).__init__()

        selected_IMF_X_train = np.expand_dims(
            dataset.close_price_data_X_train[...,selected_IMF_idx], 
            axis=-1,
        )
        selected_IMF_y_train = np.expand_dims(
            dataset.y_train[...,selected_IMF_idx], 
            axis=-1,
        )

        self.bilstms = nn.ModuleList()

        current_input_size = selected_IMF_X_train.shape[2]
        input_window_size = selected_IMF_X_train.shape[1]

        output_size = selected_IMF_y_train.shape[2]
        output_window_size = selected_IMF_y_train.shape[1]

        for l in range(n_layers):
            self.bilstms.append(
                nn.LSTM(
                    input_size=current_input_size, 
                    hidden_size=dims[l], 
                    batch_first=True,
                    bidirectional=True,
                )
            )
            current_input_size = 2 * dims[l]
        
        if activation == 'elu':
            self.bilstm_activation_func = nn.ELU()
        elif activation == 'gelu':
            self.bilstm_activation_func = nn.GELU()
        elif activation == 'leaky_relu':
            self.bilstm_activation_func = nn.LeakyReLU()
        elif activation == 'relu':
            self.bilstm_activation_func = nn.ReLU()
        elif activation == 'selu':
            self.bilstm_activation_func = nn.SELU()
        elif activation == 'sigmoid':
            self.bilstm_activation_func = nn.Sigmoid()
        elif activation == 'tanh':
            self.bilstm_activation_func = nn.Tanh()
        else:
            self.bilstm_activation_func = nn.Identity()

        self.fcs = nn.ModuleList()
        self.fc_activation_funcs = nn.ModuleList()

        for l in range(dense_n_layers):
            self.fcs.append(
                nn.Linear(
                    in_features=current_input_size, 
                    out_features=dense_dims[l],
                )
            )
            current_input_size = dense_dims[l]

            if dense_activations[l] == 'elu':
                self.fc_activation_funcs.append(nn.ELU())
            elif dense_activations[l] == 'gelu':
                self.fc_activation_funcs.append(nn.GELU())
            elif dense_activations[l] == 'leaky_relu':
                self.fc_activation_funcs.append(nn.LeakyReLU())
            elif dense_activations[l] == 'relu':
                self.fc_activation_funcs.append(nn.ReLU())
            elif dense_activations[l] == 'selu':
                self.fc_activation_funcs.append(nn.SELU())
            elif dense_activations[l] == 'sigmoid':
                self.fc_activation_funcs.append(nn.Sigmoid())
            elif dense_activations[l] == 'tanh':
                self.fc_activation_funcs.append(nn.Tanh())
            else:
                self.fc_activation_funcs.append(nn.Identity())

        self.output_fc = nn.Linear(
            in_features=current_input_size, 
            out_features=output_size * output_window_size,
        )
        
    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        x = close_price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)
        
        for bilstm in self.bilstms:
            out, _ = bilstm(x) 
            x = out

        out = x[:, -1, :]

        out = self.bilstm_activation_func(out)

        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)

        out = self.output_fc(out)
        
        return out
