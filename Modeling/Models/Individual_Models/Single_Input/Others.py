import torch
import torch.nn as nn
import torch.nn.functional as F


class Original_TCN(nn.Module):
    def __init__(
            self, 
            dataset,
            inputs,
    ):
        super(Original_TCN, self).__init__()

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

        self.conv_1 = nn.Conv1d(
            in_channels=self.input_size, 
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
                in_features=32 * self.input_window_size, 
                out_features=64
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64, 
                out_features=self.output_size * self.output_window_size
            )
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

        X = x = torch.cat(
            X, 
            dim=-1,
        )

        batch_size = X.shape[0]

        X = X.permute(0, 2, 1)
        current_X = X

        for conv in self.conv_list:
            pad = (conv.kernel_size[0]-1) * conv.dilation[0]
            current_X = F.pad(current_X, (pad, 0)) 
            current_X = self.relu(conv(current_X))

        out = current_X.view(current_X.size(0), -1)
        out = self.conv_dropout(out)
        out = self.fc(out)
        out = out.view(batch_size, -1, self.output_window_size, self.output_size)

        return out[:,-1,:,:]


# ---------------------------------------------------------------------------


class Residual_Block(nn.Module):
    def __init__(
            self, 
            input_size,
            output_size,

            kernel_size,
            n_filters,
            dilation_base,
            activation,
            dropout_rate,
            
            n_blocks_before,
            n_total_blocks,
            fc_layers_after,
    ):
        super(Residual_Block, self).__init__()

        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.n_blocks_before = n_blocks_before
        self.n_total_blocks = n_total_blocks

        input_dim = input_size if n_blocks_before == 0 else n_filters
        output_dim = output_size if (n_blocks_before == (n_total_blocks - 1)) and not fc_layers_after else n_filters

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

        if activation == 'SiLU':
            self.activation = nn.SiLU()
        elif activation == 'GELU':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.dropout_2 = nn.Dropout(p=dropout_rate)

        if input_dim != output_dim:
            self.conv_3 = nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim, 
                kernel_size=1
            )
    
    def forward(
            self, 
            X,
    ):
        res = X
        left_padding = (self.dilation_base ** self.n_blocks_before) * (self.kernel_size - 1)
        X = F.pad(X, (left_padding, 0))

        out = self.conv_1(X)
        out = self.activation(out)
        out = self.dropout_1(out)

        out = F.pad(out, (left_padding, 0))
        out = self.conv_2(out)

        if self.n_blocks_before < self.n_total_blocks - 1:
            out = self.activation(out)

        out = self.dropout_2(out)

        if self.conv_1.in_channels != self.conv_2.out_channels:
            res = self.conv_3(res)

        out = out + res

        return out


class SI_TCN(nn.Module):
    def __init__(
            self, 
            dataset,
            inputs,

            n_blocks=4,
            n_filters=128,
            kernel_size=3,
            dilation_base=2,
            dropout_rate=0.1,
            activation='relu',

            fc_n_layers=1,
            fc_dims=[32],
            fc_dropout_rate=0.1,
            fc_activation='relu',
    ):
        super(SI_TCN, self).__init__()

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

        self.res_blocks = nn.ModuleList()
        self.fc_layers_after_res_blocks = fc_n_layers > 0

        for l in range(n_blocks):
            res_block = Residual_Block(
                input_size=self.input_size,
                output_size=self.output_size * self.output_window_size,

                kernel_size=kernel_size,
                n_filters=n_filters,
                dilation_base=dilation_base,
                dropout_rate=dropout_rate,
                activation=activation,

                n_blocks_before=l,
                n_total_blocks=n_blocks,
                fc_layers_after=self.fc_layers_after_res_blocks,
            )
            self.res_blocks.append(res_block)
        
        if fc_n_layers > 0:
            self.fcs = nn.ModuleList()
            current_input_size = n_filters

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
                out_features=self.output_size,
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
        
        current_X = X.transpose(1, 2)

        for res_block in self.res_blocks:
            current_X = res_block(current_X)

        out = current_X.transpose(1, 2)

        if self.fc_layers_after_res_blocks:
            for fc in self.fcs:
                out = fc(out)
                out = self.fc_activation(out)
                out = self.fc_dropout(out)

            out = self.output_fc(out)
        
        return out[:, -self.output_window_size:, :]


# class SIMH_TCN(nn.Module):
#     def __init__(
#             self, 
#             dataset,
#     ):
#         super(SIMH_TCN, self).__init__()

#         input_size = dataset.close_price_data_X_train.shape[2]
#         input_window_size = dataset.close_price_data_X_train.shape[1]

#         output_size = dataset.y_train.shape[2]
#         output_window_size = dataset.y_train.shape[1]

#         self.target_IMFs = dataset.target_IMFs

#         n_total_blocks = 4

#         n_filters = 64
#         kernel_size = 3
#         dilation_base = 2

#         self.res_blocks = nn.ModuleList()

#         for l in range(n_total_blocks):
#             res_block = Residual_Block(
#                 input_size=input_size,

#                 kernel_size=kernel_size,
#                 n_filters=n_filters,
#                 dilation_base=dilation_base,
#                 dropout_rate=0.1,

#                 n_blocks_before=l,
#                 n_total_blocks=n_total_blocks,
#             )
#             self.res_blocks.append(res_block)
        
#         self.heads = nn.ModuleList()
#         n_inputs = input_size

#         for _ in range(n_inputs):
#             head = nn.Sequential(
#                 # nn.Flatten(),
#                 nn.Linear(
#                     in_features=n_filters, # input_window_size * n_filters
#                     out_features=32,
#                 ),
#                 nn.ReLU(),
#                 nn.Dropout(p=0.1),
#                 nn.Linear(
#                     in_features=32, 
#                     out_features=output_window_size * 1,
#                 ),
#             )

#             self.heads.append(head)
        
#     def forward(
#             self, 
#             open_price_data_X,
#             low_price_data_X,
#             high_price_data_X,
#             close_price_data_X,
#     ):
#         current_price_data_X_in = close_price_data_X.transpose(1, 2)

#         for res_block in self.res_blocks:
#             current_price_data_X_in = res_block(current_price_data_X_in)

#         out = current_price_data_X_in.transpose(1, 2)
#         out = out[:, -1, :] #out

#         outs = []

#         for head in self.heads:
#             current_out = head(out)
#             outs.append(current_out)
        
#         out = torch.cat(
#             outs, 
#             dim=-1
#         )

#         if not self.target_IMFs:
#             out = out.sum(dim=1, 
#                           keepdim=True)
            
#         # out = out.view(-1, self.output_chunk_length, self.output_size)

#         return out
