import torch.nn as nn
import numpy as np

class MM_GRU(nn.Module):
    def __init__(self, 
                 dataset,
                 selected_IMF_idx):
        super(MM_GRU, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx
        
        selected_IMF_X_train = np.expand_dims(dataset.price_data_X_train[...,selected_IMF_idx], 
                                              axis=-1)
        selected_IMF_y_train = np.expand_dims(dataset.y_train[...,selected_IMF_idx], 
                                              axis=-1)

        input_size = selected_IMF_X_train.shape[2]
        output_size = selected_IMF_y_train.shape[2]

        self.gru_1 = nn.GRU(input_size=input_size,
                            hidden_size=64,
                            batch_first=True,)

        self.fc_1 = nn.Linear(in_features=64, 
                              out_features=32)
        
        self.output_fc = nn.Linear(in_features=32,
                                   out_features=output_size)
        
        self.relu = nn.ReLU()

    def forward(self, 
                price_data_X):
        x = price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

        out, _ = self.gru_1(x)  

        out = self.fc_1(out[:,-1,:])
        out = self.relu(out)
        out = self.output_fc(out)
        
        return out
        

class HPT_MM_GRU(nn.Module):
    def __init__(self, 
                 dataset,
                 selected_IMF_idx,

                 gru_n_layers=1,
                 gru_dims=[75],
                 gru_activation='linear',

                 dense_n_layers=1,
                 dense_dims=[100],
                 dense_activations=['relu']):
        super(HPT_MM_GRU, self).__init__()
        
        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(dataset.price_data_X_train[...,selected_IMF_idx], 
                                              axis=-1)
        selected_IMF_y_train = np.expand_dims(dataset.y_train[...,selected_IMF_idx], 
                                              axis=-1)

        self.grus = nn.ModuleList()
        current_input_size = selected_IMF_X_train.shape[-1]
        output_size = selected_IMF_y_train.shape[2]

        for l in range(gru_n_layers):
            self.grus.append(nn.GRU(input_size=current_input_size, 
                                    hidden_size=gru_dims[l], 
                                    batch_first=True))
            current_input_size = gru_dims[l]
        
        if gru_activation == 'elu':
            self.gru_activation_func = nn.ELU()
        elif gru_activation == 'gelu':
            self.gru_activation_func = nn.GELU()
        elif gru_activation == 'leaky_relu':
            self.gru_activation_func = nn.LeakyReLU()
        elif gru_activation == 'relu':
            self.gru_activation_func = nn.ReLU()
        elif gru_activation == 'selu':
            self.gru_activation_func = nn.SELU()
        elif gru_activation == 'sigmoid':
            self.gru_activation_func = nn.Sigmoid()
        elif gru_activation == 'tanh':
            self.gru_activation_func = nn.Tanh()
        else:
            self.gru_activation_func = nn.Identity()

        self.fcs = nn.ModuleList()
        self.fc_activation_funcs = nn.ModuleList()

        for l in range(dense_n_layers):
            self.fcs.append(nn.Linear(in_features=current_input_size, 
                                      out_features=dense_dims[l]))
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

        self.output_fc = nn.Linear(in_features=current_input_size, 
                                   out_features=output_size)

    def forward(self, 
                price_data_X):
        x = price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

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
    def __init__(self, 
                 dataset,
                 selected_IMF_idx):
        super(MM_BiGRU, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(dataset.price_data_X_train[...,selected_IMF_idx], 
                                              axis=-1)
        selected_IMF_y_train = np.expand_dims(dataset.y_train[...,selected_IMF_idx], 
                                              axis=-1)

        input_size = selected_IMF_X_train.shape[2]
        output_size = selected_IMF_y_train.shape[2]

        self.bigru_1 = nn.GRU(input_size=input_size,
                              hidden_size=75,
                              batch_first=True,
                              bidirectional=True)
        
        self.bigru_dropout = nn.Dropout(p=0.25)

        self.fc_1 = nn.Linear(in_features=2 * 75,
                              out_features=100)
        
        self.output_fc = nn.Linear(in_features=100, 
                                   out_features=output_size)
        
        self.relu = nn.ReLU()

    def forward(self, 
                price_data_X):
        x = price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

        out, _ = self.bigru_1(x)  
        out = self.bigru_dropout(out[:,-1,:])

        out = self.fc_1(out)
        out = self.relu(out)
        out = self.output_fc(out)
        
        return out


class HPT_MM_BiGRU(nn.Module):
    def __init__(self, 
                 dataset,
                 selected_IMF_idx,

                 bigru_n_layers=1,
                 bigru_dims=[75],
                 bigru_activation='linear',

                 dense_n_layers=1,
                 dense_dims=[100],
                 dense_activations=['relu']):
        super(HPT_MM_BiGRU, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(dataset.price_data_X_train[...,selected_IMF_idx], 
                                              axis=-1)
        selected_IMF_y_train = np.expand_dims(dataset.y_train[...,selected_IMF_idx], 
                                              axis=-1)

        self.bigrus = nn.ModuleList()
        current_input_size = selected_IMF_X_train.shape[2]
        output_size = selected_IMF_y_train.shape[2]

        for l in range(bigru_n_layers):
            self.bigrus.append(nn.GRU(input_size=current_input_size, 
                                      hidden_size=bigru_dims[l], 
                                      batch_first=True,
                                      bidirectional=True))
            current_input_size = 2 * bigru_dims[l]
        
        if bigru_activation == 'elu':
            self.bigru_activation_func = nn.ELU()
        elif bigru_activation == 'gelu':
            self.bigru_activation_func = nn.GELU()
        elif bigru_activation == 'leaky_relu':
            self.bigru_activation_func = nn.LeakyReLU()
        elif bigru_activation == 'relu':
            self.bigru_activation_func = nn.ReLU()
        elif bigru_activation == 'selu':
            self.bigru_activation_func = nn.SELU()
        elif bigru_activation == 'sigmoid':
            self.bigru_activation_func = nn.Sigmoid()
        elif bigru_activation == 'tanh':
            self.bigru_activation_func = nn.Tanh()
        else:
            self.bigru_activation_func = nn.Identity()

        self.fcs = nn.ModuleList()
        self.fc_activation_funcs = nn.ModuleList()

        for l in range(dense_n_layers):
            self.fcs.append(nn.Linear(in_features=current_input_size, 
                                      out_features=dense_dims[l]))
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

        self.output_fc = nn.Linear(in_features=current_input_size, 
                                   out_features=output_size)

    def forward(self, 
                price_data_X):
        x = price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

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
    def __init__(self, 
                 dataset,
                 selected_IMF_idx):
        super(MM_LSTM, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(dataset.price_data_X_train[...,selected_IMF_idx], 
                                              axis=-1)
        selected_IMF_y_train = np.expand_dims(dataset.y_train[...,selected_IMF_idx], 
                                              axis=-1)

        input_size = selected_IMF_X_train.shape[2]
        output_size = selected_IMF_y_train.shape[2]

        self.lstm_1 = nn.LSTM(input_size=input_size,
                              hidden_size=75,
                              batch_first=True,)
        
        self.fc_1 = nn.Linear(in_features=75, 
                              out_features=100)  
        
        self.output_fc = nn.Linear(in_features=100, 
                                   out_features=output_size)
        
        self.relu = nn.ReLU()

    def forward(self, 
                price_data_X):
        x = price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

        out, _ = self.lstm_1(x)  

        out = self.fc_1(out[:, -1, :])
        out = self.relu(out)
        out = self.output_fc(out)
        
        return out
    

class HPT_MM_LSTM(nn.Module):
    def __init__(self, 
                 dataset,
                 selected_IMF_idx,

                 lstm_n_layers=1,
                 lstm_dims=[75],
                 lstm_activation='linear',

                 dense_n_layers=1,
                 dense_dims=[100],
                 dense_activations=['relu']):
        super(HPT_MM_LSTM_PT, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(dataset.price_data_X_train[...,selected_IMF_idx], 
                                              axis=-1)
        selected_IMF_y_train = np.expand_dims(dataset.y_train[...,selected_IMF_idx], 
                                              axis=-1)

        self.lstms = nn.ModuleList()
        current_input_size = selected_IMF_X_train.shape[2]
        output_size = selected_IMF_y_train.shape[2]

        for l in range(lstm_n_layers):
            self.lstms.append(nn.LSTM(input_size=current_input_size, 
                                      hidden_size=lstm_dims[l], 
                                      batch_first=True))
            current_input_size = lstm_dims[l]
        
        if lstm_activation == 'elu':
            self.lstm_activation_func = nn.ELU()
        elif lstm_activation == 'gelu':
            self.lstm_activation_func = nn.GELU()
        elif lstm_activation == 'leaky_relu':
            self.lstm_activation_func = nn.LeakyReLU()
        elif lstm_activation == 'relu':
            self.lstm_activation_func = nn.ReLU()
        elif lstm_activation == 'selu':
            self.lstm_activation_func = nn.SELU()
        elif lstm_activation == 'sigmoid':
            self.lstm_activation_func = nn.Sigmoid()
        elif lstm_activation == 'tanh':
            self.lstm_activation_func = nn.Tanh()
        else:
            self.lstm_activation_func = nn.Identity()

        self.fcs = nn.ModuleList()
        self.fc_activation_funcs = nn.ModuleList()

        for l in range(dense_n_layers):
            self.fcs.append(nn.Linear(in_features=current_input_size, 
                                      out_features=dense_dims[l]))
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

        self.output_fc = nn.Linear(in_features=current_input_size, 
                                   out_features=output_size)
        
    def forward(self, 
                price_data_X):
        x = price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

        for lstm in self.lstms:
            out, _ = lstm(x) 
            x = out

        out = x[:, -1, :]

        out = self.lstm_activation_func(out)

        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)

        out = self.output_fc(out)
        
        return out


# ---------------------------------------------------


class MM_BiLSTM(nn.Module):
    def __init__(self, 
                 dataset,
                 selected_IMF_idx):
        super(MM_BiLSTM, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(dataset.price_data_X_train[...,selected_IMF_idx], 
                                              axis=-1)
        selected_IMF_y_train = np.expand_dims(dataset.y_train[...,selected_IMF_idx], 
                                              axis=-1)

        input_size = selected_IMF_X_train.shape[2]
        output_size = selected_IMF_y_train.shape[2]

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=75,
                            batch_first=True,
                            bidirectional=True)
        
        self.fc1 = nn.Linear(in_features=2 * 75, 
                             out_features=100)
        
        self.output_fc = nn.Linear(in_features=100, 
                                   out_features=output_size)
        
        self.relu = nn.ReLU()

    def forward(self, 
                price_data_X):
        x = price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)

        out, _ = self.lstm(x)  

        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.output_fc(out)
        
        return out
    

class HPT_MM_BiLSTM(nn.Module):
    def __init__(self, 
                 dataset,
                 selected_IMF_idx,

                 bilstm_n_layers=1,
                 bilstm_dims=[75],
                 bilstm_activation='linear',

                 dense_n_layers=1,
                 dense_dims=[100],
                 dense_activations=['relu']):
        super(HPT_MM_BiLSTM, self).__init__()

        selected_IMF_X_train = np.expand_dims(dataset.price_data_X_train[...,selected_IMF_idx], 
                                              axis=-1)
        selected_IMF_y_train = np.expand_dims(dataset.y_train[...,selected_IMF_idx], 
                                              axis=-1)

        self.bilstms = nn.ModuleList()
        current_input_size = selected_IMF_X_train.shape[2]
        output_size = selected_IMF_y_train.shape[2]

        for l in range(bilstm_n_layers):
            self.bilstms.append(nn.LSTM(input_size=current_input_size, 
                                        hidden_size=bilstm_dims[l], 
                                        batch_first=True,
                                        bidirectional=True))
            current_input_size = 2 * bilstm_dims[l]
        
        if bilstm_activation == 'elu':
            self.bilstm_activation_func = nn.ELU()
        elif bilstm_activation == 'gelu':
            self.bilstm_activation_func = nn.GELU()
        elif bilstm_activation == 'leaky_relu':
            self.bilstm_activation_func = nn.LeakyReLU()
        elif bilstm_activation == 'relu':
            self.bilstm_activation_func = nn.ReLU()
        elif bilstm_activation == 'selu':
            self.bilstm_activation_func = nn.SELU()
        elif bilstm_activation == 'sigmoid':
            self.bilstm_activation_func = nn.Sigmoid()
        elif bilstm_activation == 'tanh':
            self.bilstm_activation_func = nn.Tanh()
        else:
            self.bilstm_activation_func = nn.Identity()

        self.fcs = nn.ModuleList()
        self.fc_activation_funcs = nn.ModuleList()

        for l in range(dense_n_layers):
            self.fcs.append(nn.Linear(in_features=current_input_size, 
                                      out_features=dense_dims[l]))
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

        self.output_fc = nn.Linear(in_features=current_input_size, 
                                   out_features=output_size)
        
    def forward(self, 
                price_data_X):
        x = price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)
        
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
