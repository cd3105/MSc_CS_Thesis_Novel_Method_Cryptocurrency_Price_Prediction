import torch.nn as nn
import numpy as np 
import torch

# GRU:

class MIC_GRU_Block(nn.Module):
    def __init__(self, 
                 initial_input_size,

                 gru_n_layers=1,
                 gru_dims=[64],
                 gru_activation='linear',
                 
                 dense_n_layers=1,
                 dense_dims=[32],
                 dense_activations=['relu']):
        super(MIC_GRU_Block, self).__init__()

        self.grus = nn.ModuleList()
        current_input_size = initial_input_size

        for l in range(gru_n_layers):
            self.grus.append(
                nn.GRU(
                    input_size=current_input_size, 
                    hidden_size=gru_dims[l], 
                    batch_first=True,
                )
            )
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
    
    def forward(self, x):
        for gru in self.grus:
            out, _ = gru(x) 
            x = out

        out = x[:, -1, :]

        out = self.gru_activation_func(out)

        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)
        
        return out


class MIC_GRU(nn.Module):
    def __init__(self, 
                 dataset):
        super(MIC_GRU, self).__init__()

        self.gru_blocks = nn.ModuleList()
        n_inputs = dataset.price_data_X_train.shape[2]
        output_size = dataset.y_train.shape[2]

        for _ in range(n_inputs):
            self.gru_blocks.append(
                MIC_GRU_Block(
                    initial_input_size=1,
                    gru_n_layers=1,
                    gru_dims=[64],
                    gru_activation='linear',
                    
                    dense_n_layers=1, 
                    dense_dims=[32], 
                    dense_activations=['relu'],
                )
            ) 

        self.fc_1 = nn.Linear(
            in_features=n_inputs * 32, 
            out_features=n_inputs * 16,
        )
        
        self.output_fc = nn.Linear(
            in_features=n_inputs * 16,
            out_features=output_size,
        )
        
        self.relu = nn.ReLU()

    def forward(self, 
                price_data_X):
        outs = []

        for i, gru_block in enumerate(self.gru_blocks):
            current_input = price_data_X[..., i].unsqueeze(-1)
            outs.append(gru_block(current_input))

        out = torch.cat(outs, 
                        dim=-1)
        
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.output_fc(out)
        
        return out
        

class HPT_MIC_GRU(nn.Module):
    def __init__(self, 
                 dataset,

                 block_gru_n_layers=1,
                 block_gru_dims=[64],
                 block_gru_activation='linear',

                 block_dense_n_layers=1,
                 block_dense_dims=[32],
                 block_dense_activations=['relu'],
                 
                 dense_n_layers=1,
                 dense_dim_multipliers=[16],
                 dense_activations=['relu']):
        super(HPT_MIC_GRU, self).__init__()

        self.gru_blocks = nn.ModuleList()
        n_inputs = dataset.price_data_X_train.shape[2]
        output_size = dataset.y_train.shape[2]

        for _ in range(n_inputs):
            self.gru_blocks.append(
                MIC_GRU_Block(
                    initial_input_size=1,
                    gru_n_layers=block_gru_n_layers,
                    gru_dims=block_gru_dims,
                    gru_activation=block_gru_activation,
                    
                    dense_n_layers=block_dense_n_layers, 
                    dense_dims=block_dense_dims, 
                    dense_activations=block_dense_activations,
                )
            ) 

        if dense_n_layers == 0:
            current_input_size = block_gru_dims[-1] * n_inputs
        else:
            current_input_size = block_dense_dims[-1] * n_inputs

        for l in range(dense_n_layers):
            self.fcs.append(
                nn.Linear(
                    in_features=current_input_size, 
                    out_features=n_inputs * dense_dim_multipliers[l],
                )
            )
            current_input_size = n_inputs * dense_dim_multipliers[l]

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
            out_features=output_size,
        )

    def forward(self, x):
        outs = []

        for i, gru_block in enumerate(self.gru_blocks):
            current_input = x[..., i].unsqueeze(-1)
            outs.append(gru_block(current_input))

        out = torch.cat(outs, 
                        dim=-1)
        
        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)

        out = self.output_fc(out)
        
        return out


# ---------------------------------------------------

class MIC_BiGRU_Block(nn.Module):
    def __init__(self, 
                 initial_input_size,

                 bigru_n_layers=1,
                 bigru_dims=[64],
                 bigru_activation='linear',
                 
                 dense_n_layers=1,
                 dense_dims=[32],
                 dense_activations=['relu']):
        super(MIC_BiGRU_Block, self).__init__()

        self.bigrus = nn.ModuleList()
        current_input_size = initial_input_size

        for l in range(bigru_n_layers):
            self.bigrus.append(
                nn.GRU(
                    input_size=current_input_size,
                    hidden_size=bigru_dims[l], 
                    batch_first=True,
                    bidirectional=True,
                )
            )
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
    
    def forward(self, x):
        for bigru in self.bigrus:
            out, _ = bigru(x) 
            x = out

        out = x[:, -1, :]

        out = self.bigru_activation_func(out)

        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)
        
        return out


class MIC_BiGRU(nn.Module):
    def __init__(self, 
                 dataset):
        super(MIC_BiGRU, self).__init__()

        self.bigru_blocks = nn.ModuleList()
        n_inputs = dataset.price_data_X_train.shape[2]
        output_size = dataset.y_train.shape[2]

        for _ in range(n_inputs):
            self.bigru_blocks.append(
                MIC_BiGRU_Block(
                    initial_input_size=1,
                    bigru_n_layers=1,
                    bigru_dims=[64],
                    bigru_activation='linear',
                    
                    dense_n_layers=1, 
                    dense_dims=[32], 
                    dense_activations=['relu'],
                )
            ) 

        self.fc_1 = nn.Linear(
            in_features=n_inputs * 32, 
            out_features=n_inputs * 16,
        )
        
        self.output_fc = nn.Linear(
            in_features=n_inputs * 16,
            out_features=output_size,
        )
        
        self.relu = nn.ReLU()

    def forward(self, 
                price_data_X):
        outs = []

        for i, bigru_block in enumerate(self.bigru_blocks):
            current_input = price_data_X[...,i].unsqueeze(-1)
            outs.append(bigru_block(current_input))

        out = torch.cat(outs, 
                        dim=-1)
        
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.output_fc(out)
        
        return out


class HPT_MIC_BiGRU(nn.Module):
    def __init__(self, 
                 dataset,

                 block_bigru_n_layers=1,
                 block_bigru_dims=[64],
                 block_bigru_activation='linear',

                 block_dense_n_layers=1,
                 block_dense_dims=[32],
                 block_dense_activations=['relu'],
                 
                 dense_n_layers=1,
                 dense_dim_multipliers=[16],
                 dense_activations=['relu']):
        super(HPT_MIC_BiGRU, self).__init__()

        self.bigru_blocks = nn.ModuleList()
        n_inputs = dataset.price_data_X_train.shape[2]
        output_size = dataset.y_train.shape[2]

        for _ in range(n_inputs):
            self.bigru_blocks.append(
                MIC_BiGRU_Block(initial_input_size=1,
                            bigru_n_layers=block_bigru_n_layers,
                            bigru_dims=block_bigru_dims,
                            bigru_activation=block_bigru_activation,
                            
                            dense_n_layers=block_dense_n_layers, 
                            dense_dims=block_dense_dims, 
                            dense_activations=block_dense_activations
                )
            ) 

        if dense_n_layers == 0:
            current_input_size = block_bigru_dims[-1] * n_inputs
        else:
            current_input_size = block_dense_dims[-1] * n_inputs

        for l in range(dense_n_layers):
            self.fcs.append(
                nn.Linear(
                    in_features=current_input_size, 
                    out_features=n_inputs * dense_dim_multipliers[l],
                )
            )
            current_input_size = n_inputs * dense_dim_multipliers[l]

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
            out_features=output_size,
        )

    def forward(self, x):
        outs = []

        for i, bigru_block in enumerate(self.bigru_blocks):
            current_input = x[..., i].unsqueeze(-1)
            outs.append(bigru_block(current_input))

        out = torch.cat(outs, 
                        dim=-1)
        
        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)

        out = self.output_fc(out)
        
        return out

# ---------------------------------------------------

class MIC_LSTM_Block(nn.Module):
    def __init__(self, 
                 initial_input_size,

                 lstm_n_layers=1,
                 lstm_dims=[64],
                 lstm_activation='linear',
                 
                 dense_n_layers=1,
                 dense_dims=[32],
                 dense_activations=['relu']):
        super(MIC_LSTM_Block, self).__init__()

        self.lstms = nn.ModuleList()
        current_input_size = initial_input_size

        for l in range(lstm_n_layers):
            self.lstms.append(
                nn.LSTM(
                    input_size=current_input_size,
                    hidden_size=lstm_dims[l], 
                    batch_first=True,
                )
            )
            current_input_size = 2 * lstm_dims[l]
        
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
    
    def forward(self, x):
        for lstm in self.lstms:
            out, _ = lstm(x) 
            x = out

        out = x[:, -1, :]

        out = self.lstm_activation_func(out)

        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)
        
        return out


class MIC_LSTM(nn.Module):
    def __init__(self, 
                 dataset):
        super(MIC_LSTM, self).__init__()

        self.lstm_blocks = nn.ModuleList()
        n_inputs = dataset.price_data_X_train.shape[2]
        output_size = dataset.y_train.shape[2]

        for _ in range(n_inputs):
            self.lstm_blocks.append(
                MIC_LSTM_Block(
                    initial_input_size=1,
                    lstm_n_layers=1,
                    lstm_dims=[64],
                    lstm_activation='linear',
                    
                    dense_n_layers=1, 
                    dense_dims=[32], 
                    dense_activations=['relu'],
                )
            ) 

        self.fc_1 = nn.Linear(
            in_features=n_inputs * 32, 
            out_features=n_inputs * 16,
        )
        
        self.output_fc = nn.Linear(
            in_features=n_inputs * 16,
            out_features=output_size,
        )
        
        self.relu = nn.ReLU()

    def forward(self, 
                price_data_X):
        outs = []

        for i, lstm_block in enumerate(self.lstm_blocks):
            current_input = price_data_X[..., i].unsqueeze(-1)
            outs.append(lstm_block(current_input))

        out = torch.cat(outs, 
                        dim=-1)
        
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.output_fc(out)
        
        return out
    

class HPT_MIC_LSTM(nn.Module):
    def __init__(self, 
                 dataset,

                 block_lstm_n_layers=1,
                 block_lstm_dims=[64],
                 block_lstm_activation='linear',

                 block_dense_n_layers=1,
                 block_dense_dims=[32],
                 block_dense_activations=['relu'],
                 
                 dense_n_layers=1,
                 dense_dim_multipliers=[16],
                 dense_activations=['relu']):
        super(HPT_MIC_LSTM, self).__init__()

        self.lstm_blocks = nn.ModuleList()
        n_inputs = dataset.price_data_X_train.shape[2]
        output_size = dataset.y_train.shape[2]

        for _ in range(n_inputs):
            self.lstm_blocks.append(
                MIC_LSTM_Block(initial_input_size=1,
                           lstm_n_layers=block_lstm_n_layers,
                           lstm_dims=block_lstm_dims,
                           lstm_activation=block_lstm_activation,
                           
                           dense_n_layers=block_dense_n_layers, 
                           dense_dims=block_dense_dims,
                           dense_activations=block_dense_activations,
                )
            ) 

        if dense_n_layers == 0:
            current_input_size = block_lstm_dims[-1] * n_inputs
        else:
            current_input_size = block_dense_dims[-1] * n_inputs

        for l in range(dense_n_layers):
            self.fcs.append(
                nn.Linear(
                    in_features=current_input_size, 
                    out_features=n_inputs * dense_dim_multipliers[l],
                )
            )
            current_input_size = n_inputs * dense_dim_multipliers[l]

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
            out_features=output_size,
        )

    def forward(self, x):
        outs = []

        for i, lstm_block in enumerate(self.lstm_blocks):
            current_input = x[..., i].unsqueeze(-1)
            outs.append(lstm_block(current_input))

        out = torch.cat(outs, 
                        dim=-1)
        
        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)

        out = self.output_fc(out)
        
        return out


# ---------------------------------------------------

class MIC_BiLSTM_Block(nn.Module):
    def __init__(self, 
                 initial_input_size,

                 bilstm_n_layers=1,
                 bilstm_dims=[64],
                 bilstm_activation='linear',
                 
                 dense_n_layers=1,
                 dense_dims=[32],
                 dense_activations=['relu']):
        super(MIC_BiLSTM_Block, self).__init__()

        self.bilstms = nn.ModuleList()
        current_input_size = initial_input_size

        for l in range(bilstm_n_layers):
            self.bilstms.append(
                nn.LSTM(
                    input_size=current_input_size,
                    hidden_size=bilstm_dims[l], 
                    batch_first=True,
                    bidirectional=True,
                )
            )
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
    
    def forward(self, x):
        for bilstm in self.bilstms:
            out, _ = bilstm(x) 
            x = out

        out = x[:, -1, :]

        out = self.bilstm_activation_func(out)

        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)
        
        return out


class MIC_BiLSTM(nn.Module):
    def __init__(self, 
                 dataset):
        super(MIC_BiLSTM, self).__init__()

        self.bilstm_blocks = nn.ModuleList()
        n_inputs = dataset.price_data_X_train.shape[2]
        output_size = dataset.y_train.shape[2]

        for _ in range(n_inputs):
            self.bilstm_blocks.append(
                MIC_BiLSTM_Block(
                    initial_input_size=1,
                    bilstm_n_layers=1,
                    bilstm_dims=[64],
                    bilstm_activation='linear',
                    
                    dense_n_layers=1, 
                    dense_dims=[32], 
                    dense_activations=['relu'],
                )
            ) 

        self.fc_1 = nn.Linear(
            in_features=n_inputs * 32, 
            out_features=n_inputs * 16,
        )
        
        self.output_fc = nn.Linear(
            in_features=n_inputs * 16,
            out_features=output_size,
        )
        
        self.relu = nn.ReLU()

    def forward(self, 
                price_data_X):
        outs = []

        for i, bilstm_block in enumerate(self.bilstm_blocks):
            current_input = price_data_X[..., i].unsqueeze(-1)
            outs.append(bilstm_block(current_input))

        out = torch.cat(outs, 
                        dim=-1)
        
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.output_fc(out)
        
        return out
    

class HPT_MIC_BiLSTM(nn.Module):
    def __init__(self, 
                 dataset,

                 block_bilstm_n_layers=1,
                 block_bilstm_dims=[64],
                 block_bilstm_activation='linear',

                 block_dense_n_layers=1,
                 block_dense_dims=[32],
                 block_dense_activations=['relu'],
                 
                 dense_n_layers=1,
                 dense_dim_multipliers=[16],
                 dense_activations=['relu']):
        super(HPT_MIC_BiLSTM, self).__init__()

        self.bilstm_blocks = nn.ModuleList()
        n_inputs = dataset.price_data_X_train.shape[2]
        output_size = dataset.y_train.shape[2]

        for _ in range(n_inputs):
            self.bilstm_blocks.append(
                MIC_BiLSTM_Block(
                    initial_input_size=1,
                    bilstm_n_layers=block_bilstm_n_layers,
                    bilstm_dims=block_bilstm_dims,
                    bilstm_activation=block_bilstm_activation,
                    
                    dense_n_layers=block_dense_n_layers, 
                    dense_dims=block_dense_dims, 
                    dense_activations=block_dense_activations,
                )
            ) 

        if dense_n_layers == 0:
            current_input_size = block_bilstm_dims[-1] * n_inputs
        else:
            current_input_size = block_dense_dims[-1] * n_inputs

        for l in range(dense_n_layers):
            self.fcs.append(
                nn.Linear(
                    in_features=current_input_size, 
                    out_features=n_inputs * dense_dim_multipliers[l],
                )
            )
            current_input_size = n_inputs * dense_dim_multipliers[l]

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
            out_features=output_size,
        )

    def forward(self, x):
        outs = []

        for i, bilstm_block in enumerate(self.bilstm_blocks):
            current_input = x[..., i].unsqueeze(-1)
            outs.append(bilstm_block(current_input))

        out = torch.cat(outs, 
                        dim=-1)
        
        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)

        out = self.output_fc(out)
        
        return out
