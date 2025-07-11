import torch
import torch.nn as nn
import os

class Alchemy:
    def __init__(
        self, model, training_set, val_set=None, optimizer=None, loss_fn=None,
        device=None, save_path="nep_model.pt", early_stopping_patience=10
    ):
        self.model = model
        self.training_set = training_set
        self.val_set = val_set
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=1e-3)
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience = 0
        self.model.to(self.device)

    def compute_loss(self, prediction, batch, weights=None):
        if weights is None:
            weights = {'energy': 1.0, 'forces': 1.0, 'virial': 0.1}
        loss = 0.0
        loss_dict = {}

        if 'energy' in prediction and 'energy' in batch and 'is_energy' in batch:
            mask = batch['is_energy']
            pred = prediction['energy_total'][mask]
            target = batch['energy'][mask]
            energy_loss = self.loss_fn(pred, target)
            loss += weights.get('energy', 1.0) * energy_loss
            loss_dict['energy_loss'] = energy_loss.item()

        if 'forces' in prediction and 'forces' in batch and 'is_forces' in batch:
            mask = batch['is_forces']
            pred = prediction['forces'][mask]
            target = batch['forces'][mask]
            forces_loss = self.loss_fn(pred, target)
            loss += weights.get('forces', 1.0) * forces_loss
            loss_dict['forces_loss'] = forces_loss.item()

        if 'virial' in prediction and 'virial' in batch and 'is_virial' in batch:
            mask = batch['is_virial']
            pred = prediction['virial'][mask]
            target = batch['virial'][mask]
            virial_loss = self.loss_fn(pred, target)
            loss += weights.get('virial', 1.0) * virial_loss
            loss_dict['virial_loss'] = virial_loss.item()
        return loss, loss_dict

    def train_epoch(self, weights=None):
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
        return avg_loss, avg_energy_loss, avg_forces_loss, avg_virial_loss

    def validate(self, weights=None):
        if self.val_set is None:
            return None, None, None, None
        self.model.eval()
        val_loss = 0
        num_batches = 0
        energy_loss_sum = 0
        forces_loss_sum = 0
        virial_loss_sum = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_set, 1):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                prediction = self.model(batch)
                loss, loss_dict = self.compute_loss(prediction, batch, weights)
                val_loss += loss.item()
                energy_loss_sum += loss_dict.get('energy_loss', 0.0)
                forces_loss_sum += loss_dict.get('forces_loss', 0.0)
                virial_loss_sum += loss_dict.get('virial_loss', 0.0)
                num_batches += 1
                print(f"  [Val Batch {batch_idx}] Loss = {loss.item():.6f} | Energy = {loss_dict.get('energy_loss', 0.0):.6f} | Forces = {loss_dict.get('forces_loss', 0.0):.6f} | Virial = {loss_dict.get('virial_loss', 0.0):.6f}")
        avg_loss = val_loss / num_batches if num_batches > 0 else 0.0
        avg_energy_loss = energy_loss_sum / num_batches if num_batches > 0 else 0.0
        avg_forces_loss = forces_loss_sum / num_batches if num_batches > 0 else 0.0
        avg_virial_loss = virial_loss_sum / num_batches if num_batches > 0 else 0.0
        return avg_loss, avg_energy_loss, avg_forces_loss, avg_virial_loss

    def save(self, path=None):
        save_path = path if path is not None else self.save_path
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epoch': getattr(self, 'current_epoch', 0)
        }, save_path)

    def fit(self, epochs=100):
        print(f"开始训练，总共 {epochs} 轮")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        print("-" * 50)
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            train_loss, train_energy_loss, train_forces_loss, train_virial_loss = self.train_epoch()
            print(f"Epoch {epoch:3d}/{epochs}: Train Loss = {train_loss:.6f} | Energy = {train_energy_loss:.6f} | Forces = {train_forces_loss:.6f} | Virial = {train_virial_loss:.6f}")
            
            val_loss, val_energy_loss, val_forces_loss, val_virial_loss = self.validate()
            if val_loss is not None:
                print(f"{'':16} Val Loss   = {val_loss:.6f} | Energy = {val_energy_loss:.6f} | Forces = {val_forces_loss:.6f} | Virial = {val_virial_loss:.6f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience = 0
                    self.save()
                    print(f"{'':16} ✓ 模型已保存 (best val loss: {val_loss:.6f})")
                else:
                    self.patience += 1
                    print(f"{'':16} Patience: {self.patience}/{self.early_stopping_patience}")
                    
                    if self.patience >= self.early_stopping_patience:
                        print(f"\n Early stopping triggered after {epoch} epochs")
                        break
            else:
                self.save()
                
        print("\n训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
    
    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点文件不存在: {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"✓ 已加载检查点: {path}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"✓ 已加载模型权重: {path}")