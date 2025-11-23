import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_Block(nn.Module):
    def __init__(self, 
                 input_size,
                 #output_size,

                 kernel_size,
                 n_filters,
                 dilation_base,
                 dropout_rate,

                 n_blocks_before,
                 n_total_blocks):
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
    
    def forward(self, 
                price_data_X):
        
        res = price_data_X
        left_padding = (self.dilation_base ** self.n_blocks_before) * (self.kernel_size - 1)

        price_data_X = F.pad(price_data_X, (left_padding, 0))
        out = self.dropout_1(F.relu(self.conv_1(price_data_X)))

        out = F.pad(out, (left_padding, 0))
        out = self.conv_2(out)

        if self.n_blocks_before < self.n_total_blocks - 1:
            out = F.relu(out)

        out = self.dropout_2(out)

        if self.conv_1.in_channels != self.conv_2.out_channels:
            res = self.conv_3(res)

        out = out + res

        return out


class MIC_TCN_Block(nn.Module):
    def __init__(self, 
                 initial_input_size,
                 input_window_size,
                 
                 tcn_n_total_blocks,
                 tcn_n_filters,
                 tcn_kernel_size,
                 tcn_dilation_base,
                 tcn_dropout_rate,
                 
                 dense_n_layers,
                 dense_dims,
                 dense_activations,
                 dense_dropout_rates):
        super(MIC_TCN_Block, self).__init__()

        self.res_blocks = nn.ModuleList()

        for l in range(tcn_n_total_blocks):
            res_block = Residual_Block(
                input_size=initial_input_size,

                kernel_size=tcn_kernel_size,
                n_filters=tcn_n_filters,
                dilation_base=tcn_dilation_base,
                dropout_rate=tcn_dropout_rate,

                n_blocks_before=l,
                n_total_blocks=tcn_n_total_blocks,
            )
            self.res_blocks.append(res_block)

        #self.flatten = nn.Flatten()
        current_input_size = tcn_n_filters # tcn_n_filters * input_window_size

        self.fcs = nn.ModuleList()
        self.fc_activation_funcs = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()

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
            
            self.fc_dropouts.append(nn.Dropout(p=dense_dropout_rates[l]))

    def forward(self, 
                price_data_X):
        current_price_data_X_in = price_data_X.transpose(1, 2)

        for res_block in self.res_blocks:
            current_price_data_X_in = res_block(current_price_data_X_in)

        out = current_price_data_X_in.transpose(1, 2)
        out = out[:, -1, :] # self.flatten(out)  

        # for fc, fc_activation_func, fc_dropout in zip(self.fcs, self.fc_activation_funcs, self.fc_dropouts):
        #     out = fc(out)
        #     out = fc_activation_func(out)
        #     out = fc_dropout(out)

        return out
    

class MIC_TCN(nn.Module):
    def __init__(self, 
                 dataset):
        super(MIC_TCN, self).__init__()

        self.tcn_blocks = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        self.fcs = nn.ModuleList()

        n_inputs = dataset.price_data_X_train.shape[2]
        input_window_size = dataset.price_data_X_train.shape[1]

        n_outputs = dataset.y_test.shape[2]
        output_window_size = dataset.y_test.shape[1]

        for _ in range(n_inputs):
            self.tcn_blocks.append(
                MIC_TCN_Block(
                    initial_input_size=1,
                    input_window_size=input_window_size,
                    
                    tcn_n_total_blocks=4,
                    tcn_n_filters=64,
                    tcn_kernel_size=3,
                    tcn_dilation_base=2,
                    tcn_dropout_rate=0.2,
                 
                    dense_n_layers=2,
                    dense_dims=[64, 32],
                    dense_activations=['relu', 'relu'],
                    dense_dropout_rates=[0.2, 0.2]
                )
            ) 

        # self.fc = (
        #     nn.Sequential(
        #         nn.Linear(
        #             in_features=n_inputs * 32, 
        #             out_features=n_inputs * 16,
        #         ),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.1),
        #         nn.Linear(
        #             in_features=n_inputs * 16, 
        #             out_features=output_size * output_window_size,
        #         )
        #     )
        # )

        # for l in range(4):
        #     res_block = Residual_Block(
        #         input_size=n_inputs * 64,

        #         kernel_size=3,
        #         n_filters=n_inputs * 32,
        #         dilation_base=2,
        #         dropout_rate=0.25,

        #         n_blocks_before=l,
        #         n_total_blocks=4,
        #     )
        #     self.res_blocks.append(res_block)

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=n_inputs * 64, 
                out_features=n_inputs * 16,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            # nn.Linear(
            #     in_features=n_inputs * 16, 
            #     out_features=n_inputs * 8,
            # ),
            # nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Linear(
                in_features=n_inputs * 16, 
                out_features=n_outputs * output_window_size,
            )
        )

    def forward(self, 
                price_data_X):
        tcn_outs = []

        for i, tcn_block in enumerate(self.tcn_blocks):
            current_input = price_data_X[..., i].unsqueeze(-1)
            current_out = tcn_block(current_input)
            tcn_outs.append(current_out)

        out = torch.cat(
            tcn_outs, 
            dim=-1
        )

        # out = out.transpose(1, 2)

        # for res_block in self.res_blocks:
        #     out = res_block(out)

        # out = out.transpose(1, 2)

        out = self.fc(out)

        return out


class MIS_TCN_Block(nn.Module):
    def __init__(self, 
                 initial_input_size,
                 input_window_size,
                 output_size,
                 output_window_size,
                 
                 tcn_n_total_blocks,
                 tcn_n_filters,
                 tcn_kernel_size,
                 tcn_dilation_base,
                 tcn_dropout_rate,
                 
                 dense_n_layers,
                 dense_dims,
                 dense_activations,
                 dense_dropout_rates):
        super(MIS_TCN_Block, self).__init__()

        self.res_blocks = nn.ModuleList()

        for l in range(tcn_n_total_blocks):
            res_block = Residual_Block(
                input_size=initial_input_size,

                kernel_size=tcn_kernel_size,
                n_filters=tcn_n_filters,
                dilation_base=tcn_dilation_base,
                dropout_rate=tcn_dropout_rate,

                n_blocks_before=l,
                n_total_blocks=tcn_n_total_blocks,
            )
            self.res_blocks.append(res_block)

        #self.flatten = nn.Flatten()
        current_input_size = tcn_n_filters # tcn_n_filters * input_window_size

        self.fcs = nn.ModuleList()
        self.fc_activation_funcs = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()

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
            
            self.fc_dropouts.append(nn.Dropout(p=dense_dropout_rates[l]))
        
        self.output_fc = nn.Linear(
            in_features=dense_dims[-1], 
            out_features=output_size * output_window_size,
        )

    def forward(self, 
                price_data_X):
        current_price_data_X_in = price_data_X.transpose(1, 2)

        for res_block in self.res_blocks:
            current_price_data_X_in = res_block(current_price_data_X_in)

        out = current_price_data_X_in.transpose(1, 2)
        out = out[:, -1, :] # self.flatten(out)  

        for fc, fc_activation_func, fc_dropout in zip(self.fcs, self.fc_activation_funcs, self.fc_dropouts):
            out = fc(out)
            out = fc_activation_func(out)
            out = fc_dropout(out)

        out = self.output_fc(out)

        return out
    

class MIS_TCN(nn.Module):
    def __init__(self, 
                 dataset):
        super(MIS_TCN, self).__init__()

        self.blocks = nn.ModuleList()

        n_inputs = dataset.price_data_X_train.shape[2]
        input_window_size = dataset.price_data_X_train.shape[1]
        output_window_size = dataset.y_test.shape[1]

        self.target_IMFs = dataset.target_IMFs

        for _ in range(n_inputs):
            self.blocks.append(
                MIS_TCN_Block(
                    initial_input_size=1,
                    input_window_size=input_window_size,
                    output_size=1,
                    output_window_size=output_window_size,
                    
                    tcn_n_total_blocks=4,
                    tcn_n_filters=64,
                    tcn_kernel_size=3,
                    tcn_dilation_base=2,
                    tcn_dropout_rate=0.1,
                 
                    dense_n_layers=1,
                    dense_dims=[64],
                    dense_activations=['relu'],
                    dense_dropout_rates=[0],
                )
            ) 

    def forward(self, 
                price_data_X):
        outs = []

        for i, block in enumerate(self.blocks):
            current_input = price_data_X[..., i].unsqueeze(-1)
            current_out = block(current_input)
            outs.append(current_out)

        out = torch.cat(outs, 
                        dim=-1)
        
        if not self.target_IMFs:
            out = out.sum(dim=1, 
                          keepdim=True)
        
        return out


# class HPT_MI_TCN(nn.Module):
#     def __init__(self, 
#                  dataset,
                 
#                  conv_n_layers=4,
#                  conv_out_channels=32,
#                  conv_kernel_size=3,
#                  conv_dilation=1,
#                  conv_activation='relu',
#                  conv_dropout_rate=0.5,

#                  dense_n_layers=1,
#                  dense_dims=[16],
#                  dense_activations=['relu']):
#         super(HPT_MI_TCN, self).__init__()

#         self.convs = nn.ModuleList()
#         input_size = dataset.price_data_X_train.shape[-1]

#         for l in range(conv_n_layers):
#             self.convs.append(nn.Conv1d(in_channels=input_size, 
#                                         out_channels=conv_out_channels,
#                                         kernel_size=conv_kernel_size, 
#                                         dilation=conv_dilation))
#             input_size = conv_out_channels

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

#         for l in range(dense_n_layers):
#             self.fcs.append(nn.Linear(input_size, 
#                                       dense_dims[l]))
#             input_size = dense_dims[l]

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
        
#         self.output_fc = nn.Linear(input_size, 
#                                    dataset.y_train.shape[2])

#     def forward(self, x):
#         x = x.permute(0, 2, 1)

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
    