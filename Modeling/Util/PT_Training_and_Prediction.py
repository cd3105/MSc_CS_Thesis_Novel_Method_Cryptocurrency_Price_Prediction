import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class EarlyStopping:
    def __init__(self, 
                 patience, 
                 delta, 
                 model_base_path,
                 model_fn,
                 verbose):
        self.patience = patience
        self.delta = delta
        self.model_base_path = model_base_path
        self.model_fn = model_fn
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
        self.verbose = verbose

        if not os.path.exists(model_base_path):
            os.makedirs(model_base_path)
    
    def check_early_stop(self, 
                         val_loss,
                         model):
        if (self.best_loss is None) or (val_loss < (self.best_loss - self.delta)):
            self.best_loss = val_loss
            self.no_improvement_count = 0
            self.save_checkpoint(model)
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")
    
    def save_checkpoint(self, 
                        model):
        torch.save(obj=model.state_dict(), 
                   f=f"{self.model_base_path}{self.model_fn}")
        
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {self.model_base_path}{self.model_fn}.")


def training(model, 
             dataset,

             n_epochs=100, 
             batch_size=32,
             optimizer='Adam',
             learning_rate=1e-3, 
             weight_decay=1e-3, 
             momentum=0.9,

             early_stopping_patience=5,
             early_stopping_delta=0,
             early_stopping_model_base_save_path="Modeling/Model_Checkpoints/", 
             early_stopping_model_fn="Best_Model.pt", 
             
             targeted_IMF_idx=0,
             multi_modal=False,
             optimization_mode=False,
             verbose=1,
             device="cuda"):
    model = model.to(device)

    early_stopping = EarlyStopping(patience=early_stopping_patience, 
                                   delta=early_stopping_delta, 
                                   model_base_path=early_stopping_model_base_save_path,
                                   model_fn=early_stopping_model_fn,
                                   verbose=verbose,)

    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), 
                               lr=learning_rate, 
                               weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), 
                                lr=learning_rate, 
                                weight_decay=weight_decay)
    elif optimizer == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=weight_decay)
    elif optimizer == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), 
                                  lr=learning_rate, 
                                  weight_decay=weight_decay)
    elif optimizer == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), 
                                 lr=learning_rate, 
                                 weight_decay=weight_decay)
    elif optimizer == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), 
                                lr=learning_rate, 
                                weight_decay=weight_decay)
    elif optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), 
                                  lr=learning_rate, 
                                  weight_decay=weight_decay,
                                  momentum=momentum)
    else:
        optimizer = optim.SGD(model.parameters(), 
                              lr=learning_rate, 
                              weight_decay=weight_decay,
                              momentum=momentum)

    criterion = nn.MSELoss()

    if optimization_mode:
        price_data_train_dataset = TensorDataset(torch.tensor(dataset.price_data_X_train, 
                                                              dtype=torch.float32), 
                                                 torch.tensor(dataset.y_train, 
                                                              dtype=torch.float32))
        price_data_train_loader = DataLoader(price_data_train_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False)
        price_data_val_dataset = TensorDataset(torch.tensor(dataset.price_data_X_val, 
                                                            dtype=torch.float32), 
                                               torch.tensor(dataset.y_val, 
                                                            dtype=torch.float32))
        price_data_val_loader = DataLoader(price_data_val_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False)
    else:
        price_data_train_dataset = TensorDataset(torch.tensor(np.concatenate((dataset.price_data_X_train, 
                                                                              dataset.price_data_X_val),
                                                                              axis=0), 
                                                                              dtype=torch.float32), 
                                                 torch.tensor(np.concatenate((dataset.y_train,
                                                                              dataset.y_val),
                                                                              axis=0), 
                                                                              dtype=torch.float32))
        price_data_train_loader = DataLoader(price_data_train_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False)
        
        price_data_val_dataset = TensorDataset(torch.tensor(dataset.price_data_X_test, 
                                                            dtype=torch.float32), 
                                               torch.tensor(dataset.y_test, 
                                                            dtype=torch.float32))
        price_data_val_loader = DataLoader(price_data_val_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False)


    training_metrics = {'train_mse':[],
                        'train_mae':[],
                        'val_mse':[],
                        'val_mae':[],}

    for epoch in range(n_epochs):
        # ---- Training ----

        model.train()
        train_loss = 0.0
        train_mae = 0.0

        for price_data_X_train, y_train in price_data_train_loader:
            price_data_X_train, y_train = price_data_X_train.to(device), y_train.to(device)

            # Forward pass
            y_train_pred = model(price_data_X=price_data_X_train)

            if multi_modal:
                y_train_true = y_train[...,targeted_IMF_idx]
            else:
                y_train_true = y_train

            loss = criterion(y_train_pred, y_train_true)

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * price_data_X_train.size(0)
            train_mae += torch.mean(torch.abs(y_train_pred - y_train_true)).item() * price_data_X_train.size(0)

        train_loss /= len(price_data_train_loader.dataset)
        train_mae /= len(price_data_train_loader.dataset)

        # ---- Validation ----
        model.eval()

        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for price_data_X_val, y_val in price_data_val_loader:
                price_data_X_val, y_val = price_data_X_val.to(device), y_val.to(device)

                y_val_pred = model(price_data_X=price_data_X_val)
                
                if multi_modal:
                    y_val_true = y_val[...,targeted_IMF_idx]
                else:
                    y_val_true = y_val

                loss = criterion(y_val_pred, y_val_true)

                val_loss += loss.item() * price_data_X_val.size(0)
                val_mae += torch.mean(torch.abs(y_val_pred - y_val_true)).item() * price_data_X_val.size(0)

        val_loss /= len(price_data_val_loader.dataset)
        val_mae /= len(price_data_val_loader.dataset)

        if verbose:
            print(f"Epoch [{epoch+1}/{n_epochs}]  Train Loss (MSE): {train_loss:.4f}  Val Loss (MSE): {val_loss:.4f} Train MAE: {train_mae:.4f}  Val MAE: {val_mae:.4f}")
        
        training_metrics['train_mse'].append(train_loss)
        training_metrics['train_mae'].append(train_mae)
        training_metrics['val_mse'].append(val_loss)
        training_metrics['val_mae'].append(val_mae)

        early_stopping.check_early_stop(model=model,
                                        val_loss=val_loss,)

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch+1}!")
            break
    
    model.load_state_dict(torch.load(f"{early_stopping_model_base_save_path}{early_stopping_model_fn}"))

    return model, training_metrics


def predict(model,
            price_data_X,
            device="cuda"):

    model.to(device)
    price_data_X = torch.tensor(price_data_X, 
                                dtype=torch.float32).to(device)

    model.eval() 

    with torch.no_grad():
        y_preds = model(price_data_X=price_data_X)

    return y_preds.detach().cpu().numpy()
