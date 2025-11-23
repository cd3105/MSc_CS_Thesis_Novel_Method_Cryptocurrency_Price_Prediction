import torch
import torch.nn as nn
import torch.nn.functional as F


class Original_TCN(nn.Module):
    def __init__(
            self, 
            dataset,
        ):
        super(Original_TCN, self).__init__()

        input_size = dataset.price_data_X_train.shape[2]
        output_size = dataset.y_train.shape[2]
        window_size = dataset.price_data_X_train.shape[1]

        self.conv_1 = nn.Conv1d(
            in_channels=input_size, 
            out_channels=32,
            kernel_size=16, 
            dilation=8,
        )

        self.conv_2 = nn.Conv1d(
            in_channels=32, 
            out_channels=32,
            kernel_size=16, 
            dilation=8,
        )

        self.conv_3 = nn.Conv1d(
            in_channels=32, 
            out_channels=32,
            kernel_size=16, 
            dilation=8,
        )

        self.conv_4 = nn.Conv1d(
            in_channels=32, 
            out_channels=32,
            kernel_size=16, 
            dilation=8,
        )

        self.conv_list = nn.ModuleList([self.conv_1, self.conv_2, self.conv_3, self.conv_4])
        self.conv_dropout = nn.Dropout(p=0.05)

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=32 * window_size, 
                out_features=64
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64, 
                out_features=output_size
            )
        )

    def forward(self, price_data_X):
        x = price_data_X.permute(0, 2, 1)

        for conv in self.conv_list:
            pad = (conv.kernel_size[0]-1) * conv.dilation[0]
            x = F.pad(x, (pad, 0)) 
            x = self.relu(conv(x))

        out = x.view(x.size(0), -1)
        out = self.conv_dropout(out)
        out = self.fc(out)

        return out


# ---------------------------------------------------------------------------


class Residual_Block(nn.Module):
    def __init__(
            self, 
            input_size,
            #output_size,

            kernel_size,
            n_filters,
            dilation_base,
            dropout_rate,
            
            n_blocks_before,
            n_total_blocks,
        ):
        super(Residual_Block, self).__init__()

        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.n_blocks_before = n_blocks_before
        self.n_total_blocks = n_total_blocks

        input_dim = input_size if n_blocks_before == 0 else n_filters
        output_dim = n_filters
        # output_dim = output_size if n_blocks_before == n_total_blocks - 1 else n_filters

        self.conv_1 = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=n_filters,
            kernel_size=kernel_size, 
            dilation=dilation_base**n_blocks_before
        )
        self.conv_2 = nn.Conv1d(
            in_channels=n_filters, 
            out_channels=output_dim,
            kernel_size=kernel_size, 
            dilation=dilation_base**n_blocks_before
        )

        self.conv_1, self.conv_2 = (
            nn.utils.parametrizations.weight_norm(self.conv_1),
            nn.utils.parametrizations.weight_norm(self.conv_2),
        )

        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.dropout_2 = nn.Dropout(p=dropout_rate)

        if input_dim != output_dim:
            self.conv_3 = nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim, 
                kernel_size=1
            )
    
    def forward(self, x):
        res = x
        left_padding = (self.dilation_base ** self.n_blocks_before) * (self.kernel_size - 1)

        x = F.pad(x, (left_padding, 0))
        out = self.dropout_1(F.relu(self.conv_1(x)))

        out = F.pad(out, (left_padding, 0))
        out = self.conv_2(out)

        if self.n_blocks_before < self.n_total_blocks - 1:
            out = F.relu(out)

        out = self.dropout_2(out)

        if self.conv_1.in_channels != self.conv_2.out_channels:
            res = self.conv_3(res)

        out = out + res

        return out


class SI_TCN(nn.Module):
    def __init__(
            self, 
            dataset,
        ):
        super(SI_TCN, self).__init__()

        input_size = dataset.price_data_X_train.shape[2]
        input_window_size = dataset.price_data_X_train.shape[1]

        output_size = dataset.y_train.shape[2]
        output_window_size = dataset.y_train.shape[1]

        n_total_blocks = 4

        n_filters = 64
        kernel_size = 3
        dilation_base = 2

        self.res_blocks = nn.ModuleList()

        for l in range(n_total_blocks):
            res_block = Residual_Block(
                input_size=input_size,

                kernel_size=kernel_size,
                n_filters=n_filters,
                dilation_base=dilation_base,
                dropout_rate=0.1,

                n_blocks_before=l,
                n_total_blocks=n_total_blocks,
            )
            self.res_blocks.append(res_block)

        self.fc = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(
                in_features=n_filters, # input_window_size * n_filters
                out_features=64,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(
                in_features=64, 
                out_features=output_window_size * output_size,
            ),
        )

    def forward(self, price_data_X):
        price_data_X_in = price_data_X.transpose(1, 2)

        for res_block in self.res_blocks:
            price_data_X_in = res_block(price_data_X_in)

        out = price_data_X_in.transpose(1, 2)
        #out = self.flatten(out)  
        #out = self.fc(out) 
        out = self.fc(out[:, -1, :]) 

        # out = out.view(-1, self.output_chunk_length, self.output_size)

        return out


class SIMH_TCN(nn.Module):
    def __init__(
            self, 
            dataset,
        ):
        super(SIMH_TCN, self).__init__()

        input_size = dataset.price_data_X_train.shape[2]
        input_window_size = dataset.price_data_X_train.shape[1]

        output_size = dataset.y_train.shape[2]
        output_window_size = dataset.y_train.shape[1]

        self.target_IMFs = dataset.target_IMFs

        n_total_blocks = 4

        n_filters = 64
        kernel_size = 3
        dilation_base = 2

        self.res_blocks = nn.ModuleList()

        for l in range(n_total_blocks):
            res_block = Residual_Block(
                input_size=input_size,

                kernel_size=kernel_size,
                n_filters=n_filters,
                dilation_base=dilation_base,
                dropout_rate=0.1,

                n_blocks_before=l,
                n_total_blocks=n_total_blocks,
            )
            self.res_blocks.append(res_block)
        
        self.heads = nn.ModuleList()
        n_inputs = input_size

        for _ in range(n_inputs):
            head = nn.Sequential(
                # nn.Flatten(),
                nn.Linear(
                    in_features=n_filters, # input_window_size * n_filters
                    out_features=32,
                ),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(
                    in_features=32, 
                    out_features=output_window_size * 1,
                ),
            )

            self.heads.append(head)
        
    def forward(self, price_data_X):
        current_price_data_X_in = price_data_X.transpose(1, 2)

        for res_block in self.res_blocks:
            current_price_data_X_in = res_block(current_price_data_X_in)

        out = current_price_data_X_in.transpose(1, 2)
        out = out[:, -1, :] #out

        outs = []

        for head in self.heads:
            current_out = head(out)
            outs.append(current_out)
        
        out = torch.cat(
            outs, 
            dim=-1
        )

        if not self.target_IMFs:
            out = out.sum(dim=1, 
                          keepdim=True)
            
        # out = out.view(-1, self.output_chunk_length, self.output_size)

        return out


# class HPT_TCN(nn.Module):
#     def __init__(self, 
#                  dataset,
                 
#                  conv_n_layers=4,
#                  conv_n_filters=32,
#                  conv_kernel_size=3,
#                  conv_dilation=1,
#                  conv_activation='relu',
#                  conv_dropout_rate=0.5,

#                  dense_n_layers=1,
#                  dense_dims=[16],
#                  dense_activations=['relu']):
#         super(HPT_TCN, self).__init__()

#         self.convs = nn.ModuleList()
#         current_input_size = dataset.price_data_X_train.shape[2]
#         output_size = dataset.y_train.shape[2]
#         window_size = dataset.price_data_X_train.shape[1]

#         for l in range(conv_n_layers):
#             self.convs.append(
#                 nn.Conv1d(
#                     in_channels=current_input_size, 
#                     out_channels=conv_n_filters,
#                     kernel_size=conv_kernel_size, 
#                     dilation=conv_dilation
#                 )
#             )
#             current_input_size = conv_n_filters

#         if conv_activation == 'elu':
#             self.conv_activation_func = nn.ELU()
#         elif conv_activation == 'gelu':
#             self.conv_activation_func = nn.GELU()
#         elif conv_activation == 'leaky_relu':
#             self.conv_activation_func = nn.LeakyReLU()
#         elif conv_activation == 'relu':
#             self.conv_activation_func = nn.ReLU()
#         elif conv_activation == 'selu':
#             self.conv_activation_func = nn.SELU()
#         elif conv_activation == 'sigmoid':
#             self.conv_activation_func = nn.Sigmoid()
#         elif conv_activation == 'tanh':
#             self.conv_activation_func = nn.Tanh()
#         else:
#             self.conv_activation_func = nn.Identity()

#         self.conv_dropout = nn.Dropout(p=conv_dropout_rate)

#         self.fcs = nn.ModuleList()
#         self.fc_activation_funcs = nn.ModuleList()
#         current_input_size = conv_n_filters * window_size

#         for l in range(dense_n_layers):
#             self.fcs.append(
#                 nn.Linear(in_features=current_input_size, 
#                           out_features=dense_dims[l]
#                 )
#             )
#             current_input_size = dense_dims[l]

#             if dense_activations[l] == 'elu':
#                 self.fc_activation_funcs.append(nn.ELU())
#             elif dense_activations[l] == 'gelu':
#                 self.fc_activation_funcs.append(nn.GELU())
#             elif dense_activations[l] == 'leaky_relu':
#                 self.fc_activation_funcs.append(nn.LeakyReLU())
#             elif dense_activations[l] == 'relu':
#                 self.fc_activation_funcs.append(nn.ReLU())
#             elif dense_activations[l] == 'selu':
#                 self.fc_activation_funcs.append(nn.SELU())
#             elif dense_activations[l] == 'sigmoid':
#                 self.fc_activation_funcs.append(nn.Sigmoid())
#             elif dense_activations[l] == 'tanh':
#                 self.fc_activation_funcs.append(nn.Tanh())
#             else:
#                 self.fc_activation_funcs.append(nn.Identity())
        
#         self.output_fc = nn.Linear(
#             in_features=current_input_size, 
#             out_features=output_size
#         )

#     def forward(self, 
#                 price_data_X):
#         x = price_data_X.permute(0, 2, 1)

#         for conv in self.convs:
#             pad = (conv.kernel_size[0]-1) * conv.dilation[0]
#             x = F.pad(x, (pad, 0)) 
#             x = self.conv_activation_func(conv(x))

#         out = x.view(x.size(0), -1)
#         out = self.conv_dropout(out)

#         for fc, fc_activation_func in zip(self.fcs, self.fc_activation_funcs):
#             out = fc(out)
#             out = fc_activation_func(out)

#         out = self.output_fc(out)

#         return out
