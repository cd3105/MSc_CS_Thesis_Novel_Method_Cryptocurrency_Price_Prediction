import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from PT_Util.Early_Stopping import Early_Stopping
from PT_Util.Torch_Crypto_Dataset import Torch_Crypto_Dataset


def uni_modal_model_training(
        model, 
        dataset,
        
        n_epochs=500, 
        batch_size=64,
        optimizer_type='Adam',
        learning_rate=1e-4, 
        weight_decay=1e-3, 
        
        early_stopping_patience=150,
        early_stopping_delta=0,
        early_stopping_model_base_save_path="Modeling/Model_Checkpoints/", 
        early_stopping_model_fn="Best_Model.pt", 
        
        #targeted_IMF_idx=0,
        #multi_modal=False,
        VMD_optimization_mode=False,
        model_optimization_mode=False,
        verbose=1,
        device="cuda",
):
    model = model.to(device)

    train_dataset = Torch_Crypto_Dataset(
        open_price_X=torch.tensor(
            dataset.open_price_data_X_train,
            dtype=torch.float32,
        ),
        high_price_X=torch.tensor(
            dataset.high_price_data_X_train,
            dtype=torch.float32,
        ),
        low_price_X=torch.tensor(
            dataset.low_price_data_X_train,
            dtype=torch.float32,
        ),
        close_price_X=torch.tensor(
            dataset.close_price_data_X_train,
            dtype=torch.float32,
        ),
        y=torch.tensor(
            dataset.y_train,
            dtype=torch.float32,
        ),
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )

    val_dataset = Torch_Crypto_Dataset(
        open_price_X=torch.tensor(
            dataset.open_price_data_X_test,
            dtype=torch.float32,
        ),
        high_price_X=torch.tensor(
            dataset.high_price_data_X_test,
            dtype=torch.float32,
        ),
        low_price_X=torch.tensor(
            dataset.low_price_data_X_test,
            dtype=torch.float32,
        ),
        close_price_X=torch.tensor(
            dataset.close_price_data_X_test,
            dtype=torch.float32,
        ),
        y=torch.tensor(
            dataset.y_test,
            dtype=torch.float32,
        ),
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )

    early_stopping = Early_Stopping(
        patience=early_stopping_patience, 
        delta=early_stopping_delta, 
        model_base_path=early_stopping_model_base_save_path,
        model_fn=early_stopping_model_fn,
        verbose=verbose,
    )
    
    if optimizer_type == 'Adam':
        optimizer_type = optim.Adam
    elif optimizer_type == 'AdamW':
        optimizer_type = optim.AdamW
    elif optimizer_type == 'RMSprop':
        optimizer_type = optim.RMSprop
    elif optimizer_type == 'SGD':
        optimizer_type = optim.SGD
    else:
        print("ERROR: Invalid Optimizer")

    criterion = nn.MSELoss()

    training_metrics = {'train_mse':[],
                        'train_mae':[],
                        'val_mse':[],
                        'val_mae':[],}
    
    optimizer = optimizer_type(
        params=model.parameters(), 
        lr=learning_rate, 
    )

    for epoch in range(n_epochs):
        # ---- Training ----

        model.train()
        train_loss = 0.0
        train_mae = 0.0

        for open_price_data_X_train, high_price_data_X_train, low_price_data_X_train, close_price_data_X_train, y_train in train_loader:
            open_price_data_X_train, high_price_data_X_train, low_price_data_X_train, close_price_data_X_train, y_train = (
                open_price_data_X_train.to(device), 
                high_price_data_X_train.to(device), 
                low_price_data_X_train.to(device), 
                close_price_data_X_train.to(device), 
                y_train.to(device),
            )

            # Forward pass

            # if multi_modal:
            #     y_train_pred = model(
            #         open_price_data_X=open_price_data_X_train,
            #         high_price_data_X=high_price_data_X_train,
            #         low_price_data_X=low_price_data_X_train,
            #         close_price_data_X=close_price_data_X_train,
            #     )
            #     y_train_true = y_train.clone()[...,targeted_IMF_idx]
            # else:
            y_train_pred = model(
                open_price_data_X=open_price_data_X_train,
                high_price_data_X=high_price_data_X_train,
                low_price_data_X=low_price_data_X_train,
                close_price_data_X=close_price_data_X_train,
            )
            y_train_true = y_train.clone()

            loss = criterion(y_train_pred, y_train_true)

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * close_price_data_X_train.size(0)
            train_mae += torch.mean(torch.abs(y_train_pred - y_train_true)).item() * close_price_data_X_train.size(0)

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        # ---- Validation ----
        model.eval()

        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for open_price_data_X_val, high_price_data_X_val, low_price_data_X_val, close_price_data_X_val, y_val in val_loader:
                open_price_data_X_val, high_price_data_X_val, low_price_data_X_val, close_price_data_X_val, y_val = (
                    open_price_data_X_val.to(device), 
                    high_price_data_X_val.to(device), 
                    low_price_data_X_val.to(device), 
                    close_price_data_X_val.to(device), 
                    y_val.to(device)
                )
                
                # if multi_modal:
                #     y_val_pred = model(
                #         open_price_data_X=open_price_data_X_val,
                #         high_price_data_X=high_price_data_X_val,
                #         low_price_data_X=low_price_data_X_val,
                #         close_price_data_X=close_price_data_X_val,
                #     )
                #     y_val_true = y_val.clone()[...,targeted_IMF_idx]
                # else:
                y_val_pred = model(
                    open_price_data_X=open_price_data_X_val,
                    high_price_data_X=high_price_data_X_val,
                    low_price_data_X=low_price_data_X_val,
                    close_price_data_X=close_price_data_X_val,
                )
                y_val_true = y_val.clone()

                loss = criterion(y_val_pred, y_val_true)

                val_loss += loss.item() * close_price_data_X_val.size(0)
                val_mae += torch.mean(torch.abs(y_val_pred - y_val_true)).item() * close_price_data_X_val.size(0)

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        if verbose:
            print(f"Epoch [{epoch+1}/{n_epochs}]  Train Loss (MSE): {train_loss}  Val Loss (MSE): {val_loss} Train MAE: {train_mae}  Val MAE: {val_mae}")
        
        training_metrics['train_mse'].append(train_loss)
        training_metrics['train_mae'].append(train_mae)
        training_metrics['val_mse'].append(val_loss)
        training_metrics['val_mae'].append(val_mae)

        early_stopping.check_early_stop(
            model=model,
            val_loss=val_loss,
        )

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch+1}!")
            break
    
    model.load_state_dict(torch.load(f"{early_stopping_model_base_save_path}{early_stopping_model_fn}"))

    os.remove(f"{early_stopping_model_base_save_path}{early_stopping_model_fn}")

    return model, training_metrics


def multi_modal_model_training(
        models, 
        dataset,
        
        n_epochs=500, 
        batch_size=64,
        optimizer_type='Adam',
        learning_rate=1e-4, 
        weight_decay=1e-3, 
        
        early_stopping_patience=150,
        early_stopping_delta=0,
        early_stopping_model_base_save_path="Modeling/Model_Checkpoints/", 
        early_stopping_model_fn="Best_Model.pt", 
        
        VMD_optimization_mode=False,
        model_optimization_mode=False,
        verbose=1,
        device="cuda",
):
    train_dataset = Torch_Crypto_Dataset(
        open_price_X=torch.tensor(
            dataset.open_price_data_X_train,
            dtype=torch.float32,
        ),
        high_price_X=torch.tensor(
            dataset.high_price_data_X_train,
            dtype=torch.float32,
        ),
        low_price_X=torch.tensor(
            dataset.low_price_data_X_train,
            dtype=torch.float32,
        ),
        close_price_X=torch.tensor(
            dataset.close_price_data_X_train,
            dtype=torch.float32,
        ),
        y=torch.tensor(
            dataset.y_train,
            dtype=torch.float32,
        ),
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )

    val_dataset = Torch_Crypto_Dataset(
        open_price_X=torch.tensor(
            dataset.open_price_data_X_test,
            dtype=torch.float32,
        ),
        high_price_X=torch.tensor(
            dataset.high_price_data_X_test,
            dtype=torch.float32,
        ),
        low_price_X=torch.tensor(
            dataset.low_price_data_X_test,
            dtype=torch.float32,
        ),
        close_price_X=torch.tensor(
            dataset.close_price_data_X_test,
            dtype=torch.float32,
        ),
        y=torch.tensor(
            dataset.y_test,
            dtype=torch.float32,
        ),
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )

    es_per_model = []
    optim_per_model = []

    if optimizer_type == 'Adam':
        optimizer_type = optim.Adam
    elif optimizer_type == 'AdamW':
        optimizer_type = optim.AdamW
    elif optimizer_type == 'RMSprop':
        optimizer_type = optim.RMSprop
    elif optimizer_type == 'SGD':
        optimizer_type = optim.SGD
    else:
        print("ERROR: Invalid Optimizer")

    for i, model in enumerate(models):
        models[i] = model.to(device)

        early_stopping = Early_Stopping(
            patience=early_stopping_patience, 
            delta=early_stopping_delta, 
            model_base_path=f'{early_stopping_model_base_save_path}{i}/',
            model_fn=early_stopping_model_fn,
            verbose=verbose,
        )

        optimizer = optimizer_type(
            params=model.parameters(), 
            lr=learning_rate, 
        )

        es_per_model.append(early_stopping)
        optim_per_model.append(optimizer)

    criterion = nn.MSELoss()

    training_metrics = {'train_mse':[],
                        'train_mae':[],
                        'val_mse':[],
                        'val_mae':[],}

    for epoch in range(n_epochs):
        # ---- Training ----

        for model in models:
            model.train()

        train_loss = 0.0
        train_mae = 0.0

        for open_price_data_X_train, high_price_data_X_train, low_price_data_X_train, close_price_data_X_train, y_train in train_loader:
            open_price_data_X_train, high_price_data_X_train, low_price_data_X_train, close_price_data_X_train, y_train = (
                open_price_data_X_train.to(device), 
                high_price_data_X_train.to(device), 
                low_price_data_X_train.to(device), 
                close_price_data_X_train.to(device), 
                y_train.to(device),
            )

            # Forward pass
            
            all_y_train_preds = []

            for model in models:
                current_y_train_pred = model(
                    open_price_data_X=open_price_data_X_train,
                    high_price_data_X=high_price_data_X_train,
                    low_price_data_X=low_price_data_X_train,
                    close_price_data_X=close_price_data_X_train,
                )
                all_y_train_preds.append(current_y_train_pred)
            
            y_train_pred = torch.cat(
                all_y_train_preds,
                dim=-1,
            )
            y_train_pred = y_train_pred.sum(
                dim=-1, 
                keepdim=True,
            )
            y_train_true = y_train.clone()

            loss = criterion(y_train_pred, y_train_true)

            # Backward + optimize
            for i, model in enumerate(models):
                optim_per_model[i].zero_grad()

            loss.backward()
            
            for i, model in enumerate(models):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim_per_model[i].step()

            train_loss += loss.item() * close_price_data_X_train.size(0)
            train_mae += torch.mean(torch.abs(y_train_pred - y_train_true)).item() * close_price_data_X_train.size(0)

        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        # ---- Validation ----

        for model in models:
            model.eval()

        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for open_price_data_X_val, high_price_data_X_val, low_price_data_X_val, close_price_data_X_val, y_val in val_loader:
                open_price_data_X_val, high_price_data_X_val, low_price_data_X_val, close_price_data_X_val, y_val = (
                    open_price_data_X_val.to(device), 
                    high_price_data_X_val.to(device), 
                    low_price_data_X_val.to(device), 
                    close_price_data_X_val.to(device), 
                    y_val.to(device)
                )
                
                all_y_val_preds = []

                for model in models:
                    current_y_val_pred = model(
                        open_price_data_X=open_price_data_X_val,
                        high_price_data_X=high_price_data_X_val,
                        low_price_data_X=low_price_data_X_val,
                        close_price_data_X=close_price_data_X_val,
                    )
                    all_y_val_preds.append(current_y_val_pred)
                
                y_val_pred = torch.cat(
                    all_y_val_preds,
                    dim=-1,
                )
                y_val_pred = y_val_pred.sum(
                    dim=-1, 
                    keepdim=True,
                )

                y_val_true = y_val.clone()

                loss = criterion(y_val_pred, y_val_true)

                val_loss += loss.item() * close_price_data_X_val.size(0)
                val_mae += torch.mean(torch.abs(y_val_pred - y_val_true)).item() * close_price_data_X_val.size(0)

        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        if verbose:
            print(f"Epoch [{epoch+1}/{n_epochs}]  Train Loss (MSE): {train_loss}  Val Loss (MSE): {val_loss} Train MAE: {train_mae}  Val MAE: {val_mae}")
        
        training_metrics['train_mse'].append(train_loss)
        training_metrics['train_mae'].append(train_mae)
        training_metrics['val_mse'].append(val_loss)
        training_metrics['val_mae'].append(val_mae)
        
        for i, model in enumerate(models):
            es_per_model[i].check_early_stop(
                model=model,
                val_loss=val_loss,
            )

            if es_per_model[i].stop_training:
                print(f"Early stopping at epoch {epoch+1}!")
                break
    
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(f"{early_stopping_model_base_save_path}/{i}/{early_stopping_model_fn}"))

        os.remove(f"{early_stopping_model_base_save_path}/{i}/{early_stopping_model_fn}")

    return models, training_metrics


def model_predict(
        model,

        open_price_data_X,
        high_price_data_X,
        low_price_data_X,
        close_price_data_X,
        
        device="cuda",
):

    model.to(device)

    open_price_data_X = torch.tensor(
        open_price_data_X, 
        dtype=torch.float32
    ).to(device)
    high_price_data_X = torch.tensor(
        high_price_data_X, 
        dtype=torch.float32
    ).to(device)
    low_price_data_X = torch.tensor(
        low_price_data_X, 
        dtype=torch.float32
    ).to(device)
    close_price_data_X = torch.tensor(
        close_price_data_X, 
        dtype=torch.float32
    ).to(device)

    model.eval() 

    with torch.no_grad():
        y_preds = model(
            open_price_data_X=open_price_data_X,
            high_price_data_X=high_price_data_X,
            low_price_data_X=low_price_data_X,
            close_price_data_X=close_price_data_X,
        )

    y_preds = y_preds.detach().cpu().numpy()

    return y_preds
