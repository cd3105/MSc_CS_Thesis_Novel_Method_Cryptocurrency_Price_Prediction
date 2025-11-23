import torch
import torch.nn as nn
import torch.nn.functional as F


class MIC_Residual_Block(nn.Module):
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
            fc_layers_after_res_blocks,
            fc_layers_after_tcn_blocks,
    ):
        super(MIC_Residual_Block, self).__init__()

        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.n_blocks_before = n_blocks_before
        self.n_total_blocks = n_total_blocks

        input_dim = input_size if n_blocks_before == 0 else n_filters
        output_dim = output_size if (n_blocks_before == (n_total_blocks - 1)) and not fc_layers_after_res_blocks and not fc_layers_after_tcn_blocks else n_filters

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


class MIC_TCN_Block(nn.Module):
    def __init__(
            self, 
            initial_input_size,
            input_window_size,

            n_blocks,
            n_filters,
            kernel_size,
            dilation_base,
            dropout_rate,
            activation,

            fc_n_layers,
            fc_dims,
            fc_dropout_rate,
            fc_activation,

            fc_layers_after_tcn_blocks,
    ):
        super(MIC_TCN_Block, self).__init__()

        self.input_size = initial_input_size
        self.input_window_size = input_window_size

        self.res_blocks = nn.ModuleList()
        self.fc_layers_after_res_blocks = fc_n_layers > 0
        self.fc_layers_after_tcn_blocks = fc_layers_after_tcn_blocks

        for l in range(n_blocks):
            res_block = MIC_Residual_Block(
                input_size=self.input_size,
                output_size=1,

                kernel_size=kernel_size,
                n_filters=n_filters,
                dilation_base=dilation_base,
                dropout_rate=dropout_rate,
                activation=activation,

                n_blocks_before=l,
                n_total_blocks=n_blocks,
                fc_layers_after_res_blocks=self.fc_layers_after_res_blocks,
                fc_layers_after_tcn_blocks=self.fc_layers_after_tcn_blocks,
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
            
            if not self.fc_layers_after_tcn_blocks:
                self.output_fc = nn.Linear(
                    in_features=current_input_size, 
                    out_features=1,
                )
        

    def forward(
            self, 
            X,
    ):
        current_X = X.transpose(1, 2)

        for res_block in self.res_blocks:
            current_X = res_block(current_X)

        out = current_X.transpose(1, 2)

        if self.fc_layers_after_res_blocks:
            for fc in self.fcs:
                out = fc(out)
                out = self.fc_activation(out)
                out = self.fc_dropout(out)

                if not self.fc_layers_after_tcn_blocks:
                    out = self.output_fc(out)
        
        return out


class MIC_TCN(nn.Module):
    def __init__(
            self, 
            dataset,
            inputs,

            tcn_n_blocks=4,
            tcn_n_filters=256,
            tcn_kernel_size=3,
            tcn_dilation_base=2,
            tcn_dropout_rate=0.1,
            tcn_activation='relu',

            tcn_fc_n_layers=0,
            tcn_fc_dims=[128, 64],
            tcn_fc_dropout_rate=0.1,
            tcn_fc_activation='relu',

            fc_n_layers=3,
            fc_dim_multipliers=[32, 16, 8],
            fc_dropout_rate=0.1,
            fc_activation='relu',
    ):
        super(MIC_TCN, self).__init__()

        self.inputs = inputs
        self.MVMD_bool = dataset.apply_MVMD
        self.n_tcn_blocks = dataset.close_price_data_X_train.shape[2]

        self.input_size_per_block = int('O' in self.inputs) + int('H' in self.inputs) + int('L' in self.inputs) + int('C' in self.inputs)
        self.input_window_size = dataset.input_window

        self.output_size = dataset.y_train.shape[2]
        self.output_window_size = dataset.output_window

        self.tcn_blocks = nn.ModuleList()
        self.fc_layers_after_tcn_blocks = fc_n_layers > 0

        for _ in range(self.n_tcn_blocks):
            self.tcn_blocks.append(
                MIC_TCN_Block(
                    initial_input_size=self.input_size_per_block,
                    input_window_size=self.input_window_size,
                    
                    n_blocks=tcn_n_blocks,
                    n_filters=tcn_n_filters,
                    kernel_size=tcn_kernel_size,
                    dilation_base=tcn_dilation_base,
                    dropout_rate=tcn_dropout_rate,
                    activation=tcn_activation,
                 
                    fc_n_layers=tcn_fc_n_layers,
                    fc_dims=tcn_fc_dims,
                    fc_dropout_rate=tcn_fc_dropout_rate,
                    fc_activation=tcn_fc_activation,
                    fc_layers_after_tcn_blocks=self.fc_layers_after_tcn_blocks,
                )
            )

        if fc_n_layers > 0:
            if tcn_fc_n_layers > 0:
                current_input_size = self.n_tcn_blocks * tcn_fc_dims[-1]
            else:
                current_input_size = self.n_tcn_blocks * tcn_n_filters
        else:
            current_input_size = self.n_tcn_blocks * 1

        if self.fc_layers_after_tcn_blocks:
            self.fcs = nn.ModuleList()

            for l in range(fc_n_layers):
                self.fcs.append(
                    nn.Linear(
                        in_features=current_input_size, 
                        out_features=self.n_tcn_blocks * fc_dim_multipliers[l],
                    )
                )

                current_input_size = self.n_tcn_blocks * fc_dim_multipliers[l]

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
        tcn_outs = []

        for i, tcn_block in enumerate(self.tcn_blocks):
            current_X = []

            if self.MVMD_bool:
                if 'O' in self.inputs:
                    current_X.append(open_price_data_X[..., i].unsqueeze(-1))

                if 'H' in self.inputs:
                    current_X.append(high_price_data_X[..., i].unsqueeze(-1))

                if 'L' in self.inputs:
                    current_X.append(low_price_data_X[..., i].unsqueeze(-1))
            else:
                if 'O' in self.inputs:
                    current_X.append(open_price_data_X)

                if 'H' in self.inputs:
                    current_X.append(high_price_data_X)

                if 'L' in self.inputs:
                    current_X.append(low_price_data_X)

            if 'C' in self.inputs:
                current_X.append(close_price_data_X[..., i].unsqueeze(-1))

            current_X = torch.cat(
                current_X, 
                dim=-1,
            )
            
            current_out = tcn_block(current_X)
            tcn_outs.append(current_out)

        out = torch.cat(
            tcn_outs, 
            dim=-1
        )

        if self.fc_layers_after_tcn_blocks:
            for fc in self.fcs:
                out = fc(out)
                out = self.fc_activation(out)
                out = self.fc_dropout(out)

            out = self.output_fc(out)
        else:
            out = out.sum(
                dim=-1, 
                keepdim=True,
            )

        return out[:, -self.output_window_size:, :]


#############################################################


class MIS_Residual_Block(nn.Module):
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
            fc_layers_after_res_blocks,
    ):
        super(MIS_Residual_Block, self).__init__()

        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.n_blocks_before = n_blocks_before
        self.n_total_blocks = n_total_blocks

        input_dim = input_size if n_blocks_before == 0 else n_filters
        output_dim = output_size if (n_blocks_before == (n_total_blocks - 1)) and not fc_layers_after_res_blocks else n_filters

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


class MIS_TCN_Block(nn.Module):
    def __init__(
            self, 
            initial_input_size,
            input_window_size,

            n_blocks,
            n_filters,
            kernel_size,
            dilation_base,
            dropout_rate,
            activation,

            fc_n_layers,
            fc_dims,
            fc_dropout_rate,
            fc_activation,
    ):
        super(MIS_TCN_Block, self).__init__()

        self.input_size = initial_input_size
        self.input_window_size = input_window_size

        self.res_blocks = nn.ModuleList()
        self.fc_layers_after_res_blocks = fc_n_layers > 0

        for l in range(n_blocks):
            res_block = MIS_Residual_Block(
                input_size=self.input_size,
                output_size=1,

                kernel_size=kernel_size,
                n_filters=n_filters,
                dilation_base=dilation_base,
                dropout_rate=dropout_rate,
                activation=activation,

                n_blocks_before=l,
                n_total_blocks=n_blocks,
                fc_layers_after_res_blocks=self.fc_layers_after_res_blocks,
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
                out_features=1,
            )
        
    def forward(
            self, 
            X,
    ):
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
        
        return out


class MIS_TCN(nn.Module):
    def __init__(
            self, 
            dataset,
            inputs,

            tcn_n_blocks=4,
            tcn_n_filters=128,
            tcn_kernel_size=3,
            tcn_dilation_base=2,
            tcn_dropout_rate=0.1,
            tcn_activation='relu',

            tcn_fc_n_layers=2,
            tcn_fc_dims=[64, 16],
            tcn_fc_dropout_rate=0.1,
            tcn_fc_activation='relu',
    ):
        super(MIS_TCN, self).__init__()

        self.inputs = inputs
        self.MVMD_bool = dataset.apply_MVMD
        self.n_tcn_blocks = dataset.close_price_data_X_train.shape[2]

        self.input_size_per_block = int('O' in self.inputs) + int('H' in self.inputs) + int('L' in self.inputs) + int('C' in self.inputs)
        self.input_window_size = dataset.input_window

        self.output_size = dataset.y_train.shape[2]
        self.output_window_size = dataset.output_window

        self.tcn_blocks = nn.ModuleList()

        for _ in range(self.n_tcn_blocks):
            self.tcn_blocks.append(
                MIS_TCN_Block(
                    initial_input_size=self.input_size_per_block,
                    input_window_size=self.input_window_size,
                    
                    n_blocks=tcn_n_blocks,
                    n_filters=tcn_n_filters,
                    kernel_size=tcn_kernel_size,
                    dilation_base=tcn_dilation_base,
                    dropout_rate=tcn_dropout_rate,
                    activation=tcn_activation,
                 
                    fc_n_layers=tcn_fc_n_layers,
                    fc_dims=tcn_fc_dims,
                    fc_dropout_rate=tcn_fc_dropout_rate,
                    fc_activation=tcn_fc_activation,
                )
            )


    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        tcn_outs = []

        for i, tcn_block in enumerate(self.tcn_blocks):
            current_X = []

            if self.MVMD_bool:
                if 'O' in self.inputs:
                    current_X.append(open_price_data_X[..., i].unsqueeze(-1))

                if 'H' in self.inputs:
                    current_X.append(high_price_data_X[..., i].unsqueeze(-1))

                if 'L' in self.inputs:
                    current_X.append(low_price_data_X[..., i].unsqueeze(-1))
            else:
                if 'O' in self.inputs:
                    current_X.append(open_price_data_X)

                if 'H' in self.inputs:
                    current_X.append(high_price_data_X)

                if 'L' in self.inputs:
                    current_X.append(low_price_data_X)

            if 'C' in self.inputs:
                current_X.append(close_price_data_X[..., i].unsqueeze(-1))

            current_X = torch.cat(
                current_X, 
                dim=-1,
            )
            
            current_out = tcn_block(current_X)
            tcn_outs.append(current_out)

        out = torch.cat(
            tcn_outs, 
            dim=-1
        )

        out = out.sum(
            dim=-1, 
            keepdim=True,
        )

        return out[:, -self.output_window_size:, :]
