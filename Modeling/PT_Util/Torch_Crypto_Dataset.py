import torch

class Torch_Crypto_Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        open_price_X, 
        high_price_X, 
        low_price_X,
        close_price_X, 
        y,
):
        self.open_price_X = open_price_X
        self.high_price_X = high_price_X
        self.low_price_X = low_price_X
        self.close_price_X = close_price_X
        self.y = y

    def __len__(self):
        return len(self.close_price_X)

    def __getitem__(
            self, 
            idx,
    ):
        return (
            self.open_price_X[idx], 
            self.high_price_X[idx], 
            self.low_price_X[idx], 
            self.close_price_X[idx], 
            self.y[idx]
        )
