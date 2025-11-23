import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    

class MM_TCN(nn.Module):
    def __init__(
            self, 
            dataset,
            inputs,
            selected_IMF_idx,

            n_blocks=4,
            n_filters=128,
            kernel_size=3,
            dilation_base=2,
            dropout_rate=0.1,
            activation='relu',

            fc_n_layers=2,
            fc_dims=[64, 16],
            fc_dropout_rate=0.1,
            fc_activation='relu',
    ):
        super(MM_TCN, self).__init__()

        self.inputs = inputs
        self.MVMD_bool = dataset.apply_MVMD
        self.selected_IMF_idx = selected_IMF_idx

        self.input_size = int('O' in self.inputs) + int('H' in self.inputs) + int('L' in self.inputs) + int('C' in self.inputs)
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

        if self.MVMD_bool:
            if 'O' in self.inputs:
                X.append(open_price_data_X[..., self.selected_IMF_idx].unsqueeze(-1))

            if 'H' in self.inputs:
                X.append(high_price_data_X[..., self.selected_IMF_idx].unsqueeze(-1))

            if 'L' in self.inputs:
                X.append(low_price_data_X[..., self.selected_IMF_idx].unsqueeze(-1))
        else:
            if 'O' in self.inputs:
                X.append(open_price_data_X)

            if 'H' in self.inputs:
                X.append(high_price_data_X)

            if 'L' in self.inputs:
                X.append(low_price_data_X)
        
        if 'C' in self.inputs:
                X.append(close_price_data_X[..., self.selected_IMF_idx].unsqueeze(-1))

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

    