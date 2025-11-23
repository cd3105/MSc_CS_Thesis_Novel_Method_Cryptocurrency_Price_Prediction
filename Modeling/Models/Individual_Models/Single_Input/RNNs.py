import torch
import torch.nn as nn


class Original_GRU(nn.Module):
    def __init__(
            self, 
            dataset,
        ):
        super(Original_GRU, self).__init__()

        self.input_size = dataset.close_price_data_X_train.shape[2]
        self.input_window_size = dataset.close_price_data_X_train.shape[1]

        self.output_size = dataset.y_train.shape[2]
        self.output_window_size = dataset.y_train.shape[1]

        self.gru = nn.Sequential(
            nn.GRU(
                input_size=self.input_size,
                hidden_size=75,
                batch_first=True,
            )
        )
        
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=75, 
                out_features=100,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=100,
                out_features=self.output_size * self.output_window_size,
            )
        )

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        x = close_price_data_X

        batch_size = x.shape[0]

        out, _ = self.gru(x)  
        out = self.fc(out)
        out = out.view(batch_size, -1, self.output_window_size, self.output_size)
        
        return out[:,-1,:,:]
            

class SI_GRU(nn.Module):
    def __init__(
            self, 
            dataset,
            inputs,

            n_layers=1,
            dims=[75],
            dropout_rate=0.0,
            
            fc_n_layers=1,
            fc_dims=[100],
            fc_dropout_rate=0.0,
            fc_activation='relu',
        ):
        super(SI_GRU, self).__init__()

        self.inputs = inputs
        self.input_size = 0
        
        if 'O' in self.inputs:
            self.input_size += dataset.open_price_data_X_train.shape[2]

        if 'H' in self.inputs:
            self.input_size += dataset.high_price_data_X_train.shape[2]

        if 'L' in self.inputs:
            self.input_size += dataset.low_price_data_X_train.shape[2]

        if 'C' in self.inputs:
            self.input_size += dataset.close_price_data_X_train.shape[2]
        
        self.input_window_size = dataset.input_window

        self.output_size = dataset.y_train.shape[2]
        self.output_window_size = dataset.output_window

        self.grus = nn.ModuleList()
        self.gru_dropout = nn.Dropout(p=dropout_rate)

        current_input_size = self.input_size
        
        for l in range(n_layers):
            self.grus.append(
                nn.GRU(
                    input_size=current_input_size, 
                    hidden_size=dims[l], 
                    batch_first=True,
                )
            )
            current_input_size = dims[l]

        self.fcs = nn.ModuleList()

        for l in range(fc_n_layers):
            self.fcs.append(
                nn.Linear(
                    in_features=current_input_size, 
                    out_features=fc_dims[l],
                )
            )

            current_input_size = fc_dims[l]

        if fc_activation == 'elu':
            self.fc_activation = nn.ELU()
        elif fc_activation == 'gelu':
            self.fc_activation = nn.GELU()
        elif fc_activation == 'leaky_relu':
            self.fc_activation = nn.LeakyReLU()
        elif fc_activation == 'selu':
            self.fc_activation = nn.SELU()
        else:
            self.fc_activation = nn.ReLU()

        if fc_activation == 'selu':
            self.fc_dropout = nn.AlphaDropout(p=fc_dropout_rate)
        else:
            self.fc_dropout = nn.Dropout(p=fc_dropout_rate)

        self.output_fc = nn.Linear(
            in_features=current_input_size, 
            out_features=self.output_size * self.output_window_size,
        )

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        X = []

        if 'O' in self.inputs:
            X.append(open_price_data_X)

        if 'H' in self.inputs:
            X.append(high_price_data_X)

        if 'L' in self.inputs:
            X.append(low_price_data_X)

        if 'C' in self.inputs:
            X.append(close_price_data_X)

        X = torch.cat(
            X, 
            dim=-1,
        )

        batch_size = X.shape[0]
        current_X = X

        for gru in self.grus:
            current_X, _ = gru(current_X) 

        out = self.gru_dropout(current_X)

        for fc in self.fcs:
            out = fc(out)
            out = self.fc_activation(out)
            out = self.fc_dropout(out)

        out = self.output_fc(out)
        out = out.view(batch_size, self.input_window_size, self.output_window_size, self.output_size)
        
        return out[:,-1,:,:]


# ---------------------------------------------------


class SI_BiGRU(nn.Module):
    def __init__(
            self, 
            dataset,
            inputs,

            n_layers=1,
            dims=[75],
            dropout_rate=0.0,
            
            fc_n_layers=1,
            fc_dims=[100],
            fc_dropout_rate=0.0,
            fc_activation='relu',
        ):
        super(SI_BiGRU, self).__init__()

        self.inputs = inputs
        self.input_size = 0
        
        if 'O' in self.inputs:
            self.input_size += dataset.open_price_data_X_train.shape[2]

        if 'H' in self.inputs:
            self.input_size += dataset.high_price_data_X_train.shape[2]

        if 'L' in self.inputs:
            self.input_size += dataset.low_price_data_X_train.shape[2]

        if 'C' in self.inputs:
            self.input_size += dataset.close_price_data_X_train.shape[2]
        
        self.input_window_size = dataset.input_window

        self.output_size = dataset.y_train.shape[2]
        self.output_window_size = dataset.output_window

        self.bigrus = nn.ModuleList()
        self.bigru_dropout = nn.Dropout(p=dropout_rate)

        current_input_size = self.input_size

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

        self.fcs = nn.ModuleList()

        for l in range(fc_n_layers):
            self.fcs.append(
                nn.Linear(
                    in_features=current_input_size, 
                    out_features=fc_dims[l],
                )
            )

            current_input_size = fc_dims[l]

        if fc_activation == 'elu':
            self.fc_activation = nn.ELU()
        elif fc_activation == 'gelu':
            self.fc_activation = nn.GELU()
        elif fc_activation == 'leaky_relu':
            self.fc_activation = nn.LeakyReLU()
        elif fc_activation == 'selu':
            self.fc_activation = nn.SELU()
        else:
            self.fc_activation = nn.ReLU()

        if fc_activation == 'selu':
            self.fc_dropout = nn.AlphaDropout(p=fc_dropout_rate)
        else:
            self.fc_dropout = nn.Dropout(p=fc_dropout_rate)

        self.output_fc = nn.Linear(
            in_features=current_input_size, 
            out_features=self.output_size * self.output_window_size,
        )

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        X = []

        if 'O' in self.inputs:
            X.append(open_price_data_X)

        if 'H' in self.inputs:
            X.append(high_price_data_X)

        if 'L' in self.inputs:
            X.append(low_price_data_X)

        if 'C' in self.inputs:
            X.append(close_price_data_X)

        X = torch.cat(
            X, 
            dim=-1,
        )

        batch_size = X.shape[0]
        current_X = X

        for bigru in self.bigrus:
            current_X, _ = bigru(current_X) 

        out = self.bigru_dropout(current_X)

        for fc in self.fcs:
            out = fc(out)
            out = self.fc_activation(out)
            out = self.fc_dropout(out)

        out = self.output_fc(out)
        out = out.view(batch_size, self.input_window_size, self.output_window_size, self.output_size)
        
        return out[:,-1,:,:]


# ---------------------------------------------------


class Original_LSTM(nn.Module):
    def __init__(
            self, 
            dataset,
        ):
        super(Original_LSTM, self).__init__()

        self.input_size = dataset.close_price_data_X_train.shape[2]
        self.input_window_size = dataset.close_price_data_X_train.shape[1]

        self.output_size = dataset.y_train.shape[2]
        self.output_window_size = dataset.y_train.shape[1]

        self.lstm = nn.Sequential(
            nn.LSTM(
                input_size=self.input_size,
                hidden_size=75,
                batch_first=True,
            )
        )
        
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=75, 
                out_features=100,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=100,
                out_features=self.output_size * self.output_window_size,
            )
        )

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        x = close_price_data_X

        batch_size = x.shape[0]

        out, _ = self.lstm(x)  
        out = self.fc(out)
        out = out.view(batch_size, -1, self.output_window_size, self.output_size)
        
        return out[:,-1,:,:]


class SI_LSTM(nn.Module):
    def __init__(
            self, 
            dataset,
            inputs,

            n_layers=1,
            dims=[75],
            dropout_rate=0.0,
            
            fc_n_layers=1,
            fc_dims=[100],
            fc_dropout_rate=0.0,
            fc_activation='relu',
        ):
        super(SI_LSTM, self).__init__()

        self.inputs = inputs
        self.input_size = 0
        
        if 'O' in self.inputs:
            self.input_size += dataset.open_price_data_X_train.shape[2]

        if 'H' in self.inputs:
            self.input_size += dataset.high_price_data_X_train.shape[2]

        if 'L' in self.inputs:
            self.input_size += dataset.low_price_data_X_train.shape[2]

        if 'C' in self.inputs:
            self.input_size += dataset.close_price_data_X_train.shape[2]
        
        self.input_window_size = dataset.input_window

        self.output_size = dataset.y_train.shape[2]
        self.output_window_size = dataset.output_window

        self.lstms = nn.ModuleList()
        self.lstm_dropout = nn.Dropout(p=dropout_rate)

        current_input_size = self.input_size

        for l in range(n_layers):
            self.lstms.append(
                nn.LSTM(
                    input_size=current_input_size, 
                    hidden_size=dims[l], 
                    batch_first=True,
                )
            )
            current_input_size = dims[l]

        self.fcs = nn.ModuleList()

        for l in range(fc_n_layers):
            self.fcs.append(
                nn.Linear(
                    in_features=current_input_size, 
                    out_features=fc_dims[l],
                )
            )

            current_input_size = fc_dims[l]

        if fc_activation == 'elu':
            self.fc_activation = nn.ELU()
        elif fc_activation == 'gelu':
            self.fc_activation = nn.GELU()
        elif fc_activation == 'leaky_relu':
            self.fc_activation = nn.LeakyReLU()
        elif fc_activation == 'selu':
            self.fc_activation = nn.SELU()
        else:
            self.fc_activation = nn.ReLU()

        if fc_activation == 'selu':
            self.fc_dropout = nn.AlphaDropout(p=fc_dropout_rate)
        else:
            self.fc_dropout = nn.Dropout(p=fc_dropout_rate)

        self.output_fc = nn.Linear(
            in_features=current_input_size, 
            out_features=self.output_size * self.output_window_size,
        )

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        X = []

        if 'O' in self.inputs:
            X.append(open_price_data_X)

        if 'H' in self.inputs:
            X.append(high_price_data_X)

        if 'L' in self.inputs:
            X.append(low_price_data_X)

        if 'C' in self.inputs:
            X.append(close_price_data_X)

        X = torch.cat(
            X, 
            dim=-1,
        )

        batch_size = X.shape[0]
        current_X = X

        for lstm in self.lstms:
            current_X, _ = lstm(current_X) 

        out = self.lstm_dropout(current_X)

        for fc in self.fcs:
            out = fc(out)
            out = self.fc_activation(out)
            out = self.fc_dropout(out)

        out = self.output_fc(out)
        out = out.view(batch_size, self.input_window_size, self.output_window_size, self.output_size)
        
        return out[:,-1,:,:]


# ---------------------------------------------------


class SI_BiLSTM(nn.Module):
    def __init__(
            self, 
            dataset,
            inputs,

            n_layers=1,
            dims=[75],
            dropout_rate=0.0,
            
            fc_n_layers=1,
            fc_dims=[100],
            fc_dropout_rate=0.0,
            fc_activation='relu',
        ):
        super(SI_BiLSTM, self).__init__()

        self.inputs = inputs
        self.input_size = 0
        
        if 'O' in self.inputs:
            self.input_size += dataset.open_price_data_X_train.shape[2]

        if 'H' in self.inputs:
            self.input_size += dataset.high_price_data_X_train.shape[2]

        if 'L' in self.inputs:
            self.input_size += dataset.low_price_data_X_train.shape[2]

        if 'C' in self.inputs:
            self.input_size += dataset.close_price_data_X_train.shape[2]
        
        self.input_window_size = dataset.input_window

        self.output_size = dataset.y_train.shape[2]
        self.output_window_size = dataset.output_window

        self.bilstms = nn.ModuleList()
        self.bilstm_dropout = nn.Dropout(p=dropout_rate)

        current_input_size = self.input_size

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

        self.fcs = nn.ModuleList()

        for l in range(fc_n_layers):
            self.fcs.append(
                nn.Linear(
                    in_features=current_input_size, 
                    out_features=fc_dims[l],
                )
            )

            current_input_size = fc_dims[l]

        if fc_activation == 'elu':
            self.fc_activation = nn.ELU()
        elif fc_activation == 'gelu':
            self.fc_activation = nn.GELU()
        elif fc_activation == 'leaky_relu':
            self.fc_activation = nn.LeakyReLU()
        elif fc_activation == 'selu':
            self.fc_activation = nn.SELU()
        else:
            self.fc_activation = nn.ReLU()

        if fc_activation == 'selu':
            self.fc_dropout = nn.AlphaDropout(p=fc_dropout_rate)
        else:
            self.fc_dropout = nn.Dropout(p=fc_dropout_rate)

        self.output_fc = nn.Linear(
            in_features=current_input_size, 
            out_features=self.output_size * self.output_window_size,
        )

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        X = []

        if 'O' in self.inputs:
            X.append(open_price_data_X)

        if 'H' in self.inputs:
            X.append(high_price_data_X)

        if 'L' in self.inputs:
            X.append(low_price_data_X)

        if 'C' in self.inputs:
            X.append(close_price_data_X)

        X = torch.cat(
            X, 
            dim=-1,
        )

        batch_size = X.shape[0]
        current_X = X

        for bilstm in self.bilstms:
            current_X, _ = bilstm(current_X) 

        out = self.bilstm_dropout(current_X)

        for fc in self.fcs:
            out = fc(out)
            out = self.fc_activation(out)
            out = self.fc_dropout(out)

        out = self.output_fc(out)
        out = out.view(batch_size, self.input_window_size, self.output_window_size, self.output_size)
        
        return out[:,-1,:,:]
