# pytorch_ms_tcn_gru.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleTCNBlock(nn.Module):
    """
    Parallel 1D convs with different kernel sizes/dilations (causal),
    concatenate outputs along the channel axis, then residual add.
    Input: (batch, channels, seq_len)
    Output: (batch, out_channels * n_scales, seq_len)
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=(3,5,7), dilations=(1,2,4), dropout=0.0):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        self.scales = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            # causal padding: pad left with (k-1)*d
            padding = (k - 1) * d
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=k,
                             dilation=d, padding=padding)
            self.scales.append(nn.Sequential(
                conv,
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        self.out_channels = out_channels * len(kernel_sizes)
        # residual mapping
        if in_channels != self.out_channels:
            self.res_conv = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, C, T)
        outs = []
        for block in self.scales:
            y = block(x)
            # because Conv1d padding=(k-1)*d produces output same length but padded on both ends,
            # we need to trim right side to ensure causality: keep last T elements.
            # Padding strategy above pads left only if we manually pad; here Conv1d symmetric padding was used.
            # To enforce causal output, slice to the original seq length from the right:
            if y.size(-1) > x.size(-1):
                y = y[:, :, :x.size(-1)]
            outs.append(y)
        out = torch.cat(outs, dim=1)        # (B, out_channels * n_scales, T)
        res = self.res_conv(x)              # map residual to same channels
        return self.relu(out + res)


class MS_TCN_GRU(nn.Module):
    def __init__(self, input_features, conv_channels=16, kernel_sizes=(3,5,7),
                 dilations=(1,2,4), gru_hidden=64, gru_layers=1, output_size=1,
                 dropout=0.0):
        super().__init__()
        # input conv to lift features -> channels
        self.input_conv = nn.Conv1d(input_features, conv_channels, kernel_size=1)
        # multi-scale TCN block
        self.ms_block = MultiScaleTCNBlock(conv_channels, conv_channels,
                                           kernel_sizes=kernel_sizes,
                                           dilations=dilations,
                                           dropout=dropout)
        # GRU expects input (B, T, C)
        self.gru = nn.GRU(input_size=self.ms_block.out_channels,
                          hidden_size=gru_hidden,
                          num_layers=gru_layers,
                          batch_first=True,
                          bidirectional=False)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # x: (B, T, features)
        x = x.permute(0, 2, 1)          # -> (B, features, T)
        x = self.input_conv(x)         # -> (B, conv_channels, T)
        x = self.ms_block(x)           # -> (B, conv_channels * n_scales, T)
        x = x.permute(0, 2, 1)         # -> (B, T, channels_for_gru)
        out, _ = self.gru(x)           # out: (B, T, gru_hidden)
        out = out[:, -1, :]            # last timestep
        return self.head(out)


if __name__ == "__main__":
    B = 8
    T = 50
    F_in = 5
    model = MS_TCN_GRU(input_features=F_in)
    x = torch.randn(B, T, F_in)
    y = model(x)
    print("PyTorch output shape:", y.shape)  # (B, output_size)
