import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Single_Head_Temporal_Attention(nn.Module):
    def __init__(
            self, 
            feature_dim,
    ):
        super(Single_Head_Temporal_Attention, self).__init__()

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        #self.scale = torch.sqrt(torch.FloatTensor([feature_dim]))

    def forward(
            self, 
            X,
    ):
        # Compute query, key, and value matrices
        Q = self.query(X)  # [batch_size, num_frames, feature_dim]
        K = self.key(X)    # [batch_size, num_frames, feature_dim]
        V = self.value(X)  # [batch_size, num_frames, feature_dim]

        # Compute attention scores
        scale = torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float, device=Q.device))

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / scale #/ self.scale  # [batch_size, num_frames, num_frames]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_frames, num_frames]

        # Apply the attention weights to the value matrix
        attended_features = torch.bmm(attention_weights, V)  # [batch_size, num_frames, feature_dim]

        return attended_features, attention_weights


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
            n_total_blocks
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
    
    def forward(
            self, 
            X,
    ):
        res = X
        left_padding = (self.dilation_base ** self.n_blocks_before) * (self.kernel_size - 1)

        X = F.pad(X, (left_padding, 0))
        out = self.dropout_1(F.relu(self.conv_1(X)))

        out = F.pad(out, (left_padding, 0))
        out = self.conv_2(out)

        if self.n_blocks_before < self.n_total_blocks - 1:
            out = F.relu(out)

        out = self.dropout_2(out)

        if self.conv_1.in_channels != self.conv_2.out_channels:
            res = self.conv_3(res)

        out = out + res

        return out
    

class MM_ATCN(nn.Module):
    def __init__(
            self, 
            dataset,
            selected_IMF_idx,
    ):
        super(MM_ATCN, self).__init__()

        self.selected_IMF_idx = selected_IMF_idx

        selected_IMF_X_train = np.expand_dims(
            dataset.close_price_data_X_train[...,selected_IMF_idx], 
            axis=-1
        )
        selected_IMF_y_train = np.expand_dims(
            dataset.y_train[...,selected_IMF_idx], 
            axis=-1
        )

        input_size = selected_IMF_X_train.shape[2]
        input_window_size = selected_IMF_X_train.shape[1]

        output_size = selected_IMF_y_train.shape[2]
        output_window_size = selected_IMF_y_train.shape[1]

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

        self.temporal_attention = Single_Head_Temporal_Attention(n_filters)

        self.fc = nn.Sequential(
            # nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=n_filters, # n_filters * input_window_size / n_filters
                out_features=64,
            ),
            nn.ReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(
                in_features=64, 
                out_features=output_size * output_window_size,
            ),
        )

    def forward(
            self, 
            open_price_data_X,
            low_price_data_X,
            high_price_data_X,
            close_price_data_X,
    ):
        current_close_price_data_X_in = close_price_data_X[...,self.selected_IMF_idx].unsqueeze(-1)
        current_close_price_data_X_in = current_close_price_data_X_in.transpose(1, 2)

        for res_block in self.res_blocks:
            current_close_price_data_X_in = res_block(current_close_price_data_X_in)

        out = current_close_price_data_X_in.transpose(1, 2)
        out, attention_weights = self.temporal_attention(out)
        out = self.fc(out.mean(dim=1))#(out[:, -1, :]) # self.fc(out) / self.fc(out[:, -1, :])
        
        # out = out.view(-1, self.output_chunk_length, self.output_size)

        return out
    