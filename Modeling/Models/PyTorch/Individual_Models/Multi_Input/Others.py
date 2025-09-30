import torch.nn as nn
import torch.nn.functional as F


class MI_TCN(nn.Module):
    def __init__(self, 
                 dataset,):
        super(MI_TCN, self).__init__()

        self.conv_1 = nn.Conv1d(in_channels=dataset.price_data_X_train.shape[-1], 
                                out_channels=32,
                                kernel_size=3, 
                                dilation=1)
        self.conv_2 = nn.Conv1d(in_channels=32, 
                                out_channels=32,
                                kernel_size=3, 
                                dilation=1)
        self.conv_3 = nn.Conv1d(in_channels=32, 
                                out_channels=32,
                                kernel_size=3, 
                                dilation=1)
        self.conv_4 = nn.Conv1d(in_channels=32, 
                                out_channels=32,
                                kernel_size=3, 
                                dilation=1)

        self.conv_dropout = nn.Dropout(p=0.5)

        self.fc_1 = nn.Linear(32 * dataset.price_data_X_train.shape[1], 
                              16)  
        self.output_fc = nn.Linear(16, 
                                   dataset.y_train.shape[2])

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)

        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            pad = (conv.kernel_size[0]-1) * conv.dilation[0]
            x = F.pad(x, (pad, 0)) 
            x = self.relu(conv(x))

        out = x.view(x.size(0), -1)
        out = self.conv_dropout(out)

        out = self.fc_1(out)
        out = self.relu(out)
        out = self.output_fc(out)

        return out
    

class HPT_MI_TCN(nn.Module):
    def __init__(self, 
                 dataset,
                 
                 conv_n_layers=4,
                 conv_out_channels=32,
                 conv_kernel_size=3,
                 conv_dilation=1,
                 conv_activation='relu',
                 conv_dropout_rate=0.5,

                 dense_n_layers=1,
                 dense_dims=[16],
                 dense_activations=['relu']):
        super(HPT_MI_TCN, self).__init__()

        self.convs = nn.ModuleList()
        input_size = dataset.price_data_X_train.shape[-1]

        for l in range(conv_n_layers):
            self.convs.append(nn.Conv1d(in_channels=input_size, 
                                        out_channels=conv_out_channels,
                                        kernel_size=conv_kernel_size, 
                                        dilation=conv_dilation))
            input_size = conv_out_channels

        if conv_activation == 'elu':
            self.conv_activation_func = nn.ELU()
        elif conv_activation == 'gelu':
            self.conv_activation_func = nn.GELU()
        elif conv_activation == 'leaky_relu':
            self.conv_activation_func = nn.LeakyReLU()
        elif conv_activation == 'relu':
            self.conv_activation_func = nn.ReLU()
        elif conv_activation == 'selu':
            self.conv_activation_func = nn.SELU()
        elif conv_activation == 'sigmoid':
            self.conv_activation_func = nn.Sigmoid()
        elif conv_activation == 'tanh':
            self.conv_activation_func = nn.Tanh()
        else:
            self.conv_activation_func = nn.Identity()

        self.conv_dropout = nn.Dropout(p=conv_dropout_rate)

        self.fcs = nn.ModuleList()
        self.fc_activation_funcs = nn.ModuleList()

        for l in range(dense_n_layers):
            self.fcs.append(nn.Linear(input_size, 
                                      dense_dims[l]))
            input_size = dense_dims[l]

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
        
        self.output_fc = nn.Linear(input_size, 
                                   dataset.y_train.shape[2])

    def forward(self, x):
        x = x.permute(0, 2, 1)

        for conv in self.convs:
            pad = (conv.kernel_size[0]-1) * conv.dilation[0]
            x = F.pad(x, (pad, 0)) 
            x = self.conv_activation_func(conv(x))

        out = x.view(x.size(0), -1)
        out = self.conv_dropout(out)

        for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
            out = fc(out)
            out = fc_activation_func(out)

        out = self.output_fc(out)

        return out
    