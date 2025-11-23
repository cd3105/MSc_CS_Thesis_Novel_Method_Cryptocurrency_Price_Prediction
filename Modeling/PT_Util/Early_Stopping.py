import os
import torch


class Early_Stopping:
    def __init__(
            self, 
            patience, 
            delta, 
            model_base_path,
            model_fn,
            verbose,
    ):
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
    
    def check_early_stop(
            self, 
            val_loss,
            model,
    ):
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
    
    def save_checkpoint(
            self, 
            model,
    ):
        torch.save(
            obj=model.state_dict(), 
            f=f"{self.model_base_path}{self.model_fn}",
        )
        
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {self.model_base_path}{self.model_fn}.")
