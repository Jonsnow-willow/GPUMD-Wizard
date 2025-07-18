import torch
import torch.nn as nn
import os
from importlib import util

has_wandb = util.find_spec("wandb") is not None
if has_wandb:
    import wandb

class Trainer:
    def __init__(
        self, model, training_set, optimizer=None, loss_fn=None,
        device=None, save_path="nep_model.pt", early_stopping_patience=10, 
        use_wandb=False, wandb_project="NEP"):
        self.model = model
        self.training_set = training_set
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=1e-3)
        self.loss_fn = loss_fn if loss_fn is not None else nn.L1Loss()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience = 0
        self.model.to(self.device)
        
        self.use_wandb = use_wandb and has_wandb
        if self.use_wandb:
            wandb.init(project=wandb_project, config={"early_stopping_patience": early_stopping_patience})

    def compute_loss(self, prediction, batch, weights=None):
        if weights is None:
            weights = {'energy': 1.0, 'forces': 1.0, 'virial': 0.1}
        loss = 0.0
        loss_dict = {}

        if 'energy' in prediction and 'energy' in batch and 'is_energy' in batch:
            mask = batch['is_energy']
            pred = prediction['energy'][mask]  
            target = batch['energy'][mask]       

            n_atoms_per_structure = batch['n_atoms_per_structure'][mask]
            pred = pred / n_atoms_per_structure.float()
            target = target / n_atoms_per_structure.float()

            energy_loss = self.loss_fn(pred, target)
            loss += weights.get('energy', 1.0) * energy_loss
            loss_dict['energy_loss'] = energy_loss.item()

        if 'forces' in prediction and 'forces' in batch:
            pred = prediction['forces']
            target = batch['forces']
            forces_loss = self.loss_fn(pred, target)
            loss += weights.get('forces', 1.0) * forces_loss
            loss_dict['forces_loss'] = forces_loss.item()

        if 'virial' in prediction and 'virial' in batch and 'is_virial' in batch:
            mask = batch['is_virial']
            pred = prediction['virial'][mask]
            target = batch['virial'][mask]

            n_atoms_per_structure = batch['n_atoms_per_structure'][mask]
            pred = pred / n_atoms_per_structure.float().unsqueeze(-1)  
            target = target / n_atoms_per_structure.float().unsqueeze(-1)

            virial_loss = self.loss_fn(pred, target)
            loss += weights.get('virial', 1.0) * virial_loss
            loss_dict['virial_loss'] = virial_loss.item()
        return loss, loss_dict

    def train_epoch(self, weights=None, epoch=None):
        self.model.train()
        total_loss = 0
        num_batches = 0
        energy_loss_sum = 0
        forces_loss_sum = 0
        virial_loss_sum = 0
        for batch_idx, batch in enumerate(self.training_set, 1):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            self.optimizer.zero_grad()
            prediction = self.model(batch)
            loss, loss_dict = self.compute_loss(prediction, batch, weights)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            energy_loss_sum += loss_dict.get('energy_loss', 0.0)
            forces_loss_sum += loss_dict.get('forces_loss', 0.0)
            virial_loss_sum += loss_dict.get('virial_loss', 0.0)
            num_batches += 1
            print(f"  [Batch {batch_idx}] Loss = {loss.item():.6f} | Energy = {loss_dict.get('energy_loss', 0.0):.6f} | Forces = {loss_dict.get('forces_loss', 0.0):.6f} | Virial = {loss_dict.get('virial_loss', 0.0):.6f}")
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_energy_loss = energy_loss_sum / num_batches if num_batches > 0 else 0.0
        avg_forces_loss = forces_loss_sum / num_batches if num_batches > 0 else 0.0
        avg_virial_loss = virial_loss_sum / num_batches if num_batches > 0 else 0.0

        if epoch is not None and self.use_wandb:
            wandb.log({
                'Loss/train_total': avg_loss,
                'Loss/train_energy': avg_energy_loss,
                'Loss/train_forces': avg_forces_loss,
                'Loss/train_virial': avg_virial_loss,
                'epoch': epoch
            })
        return avg_loss, avg_energy_loss, avg_forces_loss, avg_virial_loss

    def save(self, path=None):
        save_path = path if path is not None else self.save_path
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'para': self.model.para,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epoch': getattr(self, 'current_epoch', 0)
        }, save_path)

    def fit(self, epochs=100):
        print(f"Number of Epoch: {epochs}")
        print(f"Devices: {self.device}")
        print(f"Number of Parameters: {sum(p.numel() for p in self.model.parameters())}")
        print("-" * 50)
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_loss, train_energy_loss, train_forces_loss, train_virial_loss = self.train_epoch(epoch=epoch)
            print(f"Epoch {epoch:3d}/{epochs}: "
                f"Train Loss = {train_loss:.6f} | Energy = {train_energy_loss:.6f} | "
                f"Forces = {train_forces_loss:.6f} | Virial = {train_virial_loss:.6f}")
            self.save()        
        if self.use_wandb:
            wandb.finish()
            
        print("\ncompleted!")