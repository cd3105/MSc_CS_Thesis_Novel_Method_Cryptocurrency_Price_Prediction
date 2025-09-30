import torch
import torch.nn as nn
import torch.nn.functional as F


class MSRBlock(nn.Module):
    def __init__(self, 
                 msr_scales,
                 msr_conv1d_in_channels, 
                 msr_conv1d_out_channels,
                 msr_conv1d_activation,
                 
                 cross_channel_fusion,
                 cc_conv2d_out_channels,):
        super(MSRBlock, self).__init__()

        self.msr_convs = nn.ModuleList([
            nn.Conv1d(in_channels=msr_conv1d_in_channels, 
                      out_channels=msr_conv1d_out_channels, 
                      kernel_size=s,
                      padding='same') 
            for s in msr_scales
        ])

        if msr_conv1d_activation == 'elu':
            self.msr_conv1d_activation_func = nn.ELU()
        elif msr_conv1d_activation == 'gelu':
            self.msr_conv1d_activation_func = nn.GELU()
        elif msr_conv1d_activation == 'leaky_relu':
            self.msr_conv1d_activation_func = nn.LeakyReLU()
        elif msr_conv1d_activation == 'relu':
            self.msr_conv1d_activation_func = nn.ReLU()
        elif msr_conv1d_activation == 'selu':
            self.msr_conv1d_activation_func = nn.SELU()
        elif msr_conv1d_activation == 'sigmoid':
            self.msr_conv1d_activation_func = nn.Sigmoid()
        elif msr_conv1d_activation == 'tanh':
            self.msr_conv1d_activation_func = nn.Tanh()
        else:
            self.msr_conv1d_activation_func = nn.Identity()
        
        self.cross_channel_fusion = cross_channel_fusion

        self.cc_conv = nn.Conv2d(in_channels=msr_conv1d_out_channels * (len(msr_scales) + 1),
                                 out_channels=cc_conv2d_out_channels,
                                 kernel_size=(1,1))
        

    def forward(self, x):
        msr_conv1d_outputs = [self.msr_conv1d_activation_func(conv(x)) for conv in self.msr_convs]

        out = torch.cat([x] + msr_conv1d_outputs, 
                        dim=1)  
        
        if self.cross_channel_fusion:
            out = out.unsqueeze(2)
                            
            out = self.cc_conv(out)

            out = out.squeeze(2)

        return out


class MSRCNN_LSTM(nn.Module):
    def __init__(self, 
                 dataset,):
        super(MSRCNN_LSTM, self).__init__()

        input_size = dataset.price_data_X_train.shape[-1]
        output_size = dataset.y_train.shape[2]

        self.input_conv = nn.Conv1d(in_channels=input_size, 
                                    out_channels=16, 
                                    kernel_size=1)
        self.relu = nn.ReLU()
        
        self.ms_block = MSRBlock(msr_scales=[1,2,3],
                                 msr_conv1d_in_channels=16,
                                 msr_conv1d_out_channels=16,
                                 msr_conv1d_activation='relu',
                                 
                                 cross_channel_fusion=True,
                                 cc_conv2d_out_channels=32)
    
        self.lstm_1 = nn.LSTM(input_size=32,
                              hidden_size=50,
                              num_layers=1,
                              batch_first=True)
        
        self.output_fc = nn.Linear(in_features=50, 
                                   out_features=output_size)

    def forward(self, 
                price_data_X):
        x = price_data_X.permute(0, 2, 1) 

        out = self.input_conv(x)
        out = self.relu(out)

        out = self.ms_block(out)
        out = self.relu(out)

        out = out.permute(0, 2, 1) 

        out, _ = self.lstm_1(out)

        out = out[:, -1, :]

        out = self.output_fc(out) 
        
        return out


class HPT_MSRCNN_LSTM(nn.Module):
    def __init__(self, 
                 dataset,
                 
                 input_conv_activation,
                 
                 msr_block_conv1d_out_channels,
                 msr_block_conv1d_activation,
                 msr_block_cross_channel_fusion,
                 msr_block_conv2d_out_channels,
                 msr_block_activation,

                 lstm_n_layers=1,
                 lstm_dims=[50],
                 lstm_activation='linear',
                 
                 dense_n_layers=0,
                 dense_dims=[],
                 dense_activations=[]):
        super(HPT_MSRCNN_LSTM, self).__init__()

        current_input_size = dataset.price_data_X_train.shape[-1]
        output_size = dataset.y_train.shape[2]
        
        self.input_conv = nn.Conv1d(in_channels=current_input_size, 
                                    out_channels=msr_block_conv1d_out_channels, 
                                    kernel_size=1)
        
        if input_conv_activation == 'elu':
            self.input_conv_activation_func = nn.ELU()
        elif input_conv_activation == 'gelu':
            self.input_conv_activation_func = nn.GELU()
        elif input_conv_activation == 'leaky_relu':
            self.input_conv_activation_func = nn.LeakyReLU()
        elif input_conv_activation == 'relu':
            self.input_conv_activation_func = nn.ReLU()
        elif input_conv_activation == 'selu':
            self.input_conv_activation_func = nn.SELU()
        elif input_conv_activation == 'sigmoid':
            self.input_conv_activation_func = nn.Sigmoid()
        elif input_conv_activation == 'tanh':
            self.input_conv_activation_func = nn.Tanh()
        else:
            self.input_conv_activation_func = nn.Identity()

        self.msr_block = MSRBlock(msr_scales=[1,2,3],
                                  msr_conv1d_in_channels=msr_block_conv1d_out_channels,
                                  msr_conv1d_out_channels=msr_block_conv1d_out_channels,
                                  msr_conv1d_activation=msr_block_conv1d_activation,

                                  cc_conv2d_out_channels=msr_block_conv2d_out_channels,
                                  cross_channel_fusion=msr_block_cross_channel_fusion)
        
        if msr_block_activation == 'elu':
            self.msr_activation_func = nn.ELU()
        elif msr_block_activation == 'gelu':
            self.msr_activation_func = nn.GELU()
        elif msr_block_activation == 'leaky_relu':
            self.msr_activation_func = nn.LeakyReLU()
        elif msr_block_activation == 'relu':
            self.msr_activation_func = nn.ReLU()
        elif msr_block_activation == 'selu':
            self.msr_activation_func = nn.SELU()
        elif msr_block_activation == 'sigmoid':
            self.msr_activation_func = nn.Sigmoid()
        elif msr_block_activation == 'tanh':
            self.msr_activation_func = nn.Tanh()
        else:
            self.msr_activation_func = nn.Identity()
        
        self.lstms = nn.ModuleList()

        if msr_block_cross_channel_fusion:
            current_input_size = msr_block_conv2d_out_channels
        else:
            current_input_size = msr_block_conv1d_out_channels * (len([1,2,3]) + 1)

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
            self.fcs.append(nn.Linear(current_input_size, 
                                      dense_dims[l]))
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
        x = price_data_X.permute(0, 2, 1) 

        out = self.input_conv(x)
        out = self.input_conv_activation_func(out)

        out = self.msr_block(out)
        out = self.msr_activation_func(out)

        out = out.permute(0, 2, 1) 

        for lstm in self.lstms:
            out, _ = lstm(out) 

        out = out[:, -1, :]

        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)

        out = self.output_fc(out)
        
        return out
