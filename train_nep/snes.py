import torch
import torch.nn as nn
import numpy as np
import os

class SNES:
    def __init__(
        self, 
        model, 
        training_set, 
        loss_fn=None,
        device=None, 
        save_path="nep_model.pt",
        population_size=50,
        learning_rate=0.01,
        sigma_init=0.1,
        patience=10
    ):
        self.model = model
        self.training_set = training_set
        self.loss_fn = loss_fn if loss_fn is not None else nn.L1Loss()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.sigma_init = sigma_init
        self.patience = patience
        
        self.model.to(self.device)
        self.param_shapes = []
        self.param_sizes = []
        self.total_params = 0
        
        for param in self.model.parameters():
            self.param_shapes.append(param.shape)
            size = param.numel()
            self.param_sizes.append(size)
            self.total_params += size

        self.theta = self._get_flat_params() 
        self.sigma = np.full(self.total_params, sigma_init)  
        
        self.best_loss = float('inf')
        self.best_params = self.theta.copy()
        self.stagnation_count = 0

    def _get_flat_params(self) -> np.ndarray:
        params = []
        for param in self.model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def _set_flat_params(self, flat_params: np.ndarray):
        with torch.no_grad():
            idx = 0
            for param, size in zip(self.model.parameters(), self.param_sizes):
                param_data = flat_params[idx:idx+size].reshape(param.shape)
                param.data = torch.from_numpy(param_data).float().to(self.device)
                idx += size

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
        
        if hasattr(loss, "item"):
            loss = loss.item()
        return loss, loss_dict

    def evaluate_individual(self, params: np.ndarray) -> float:
        self._set_flat_params(params)
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        for batch in self.training_set:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
            prediction = self.model(batch)
            loss, _ = self.compute_loss(prediction, batch)
            total_loss += loss if isinstance(loss, float) else float(loss)
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else float('inf')

    def snes_step(self):
        candidates = []
        noises = []
        
        for _ in range(self.population_size):
            noise = np.random.randn(self.total_params)
            candidate = self.theta + self.sigma * noise
            candidates.append(candidate)
            noises.append(noise)
        
        fitnesses = []
        for candidate in candidates:
            fitness = self.evaluate_individual(candidate)
            fitnesses.append(fitness)
        
        fitnesses = np.array(fitnesses)
        ranks = np.argsort(fitnesses)
        utilities = np.zeros(self.population_size)
        for i, rank in enumerate(ranks):
            utilities[rank] = max(0, np.log(self.population_size / 2 + 1) - np.log(i + 1))
        utilities = utilities / utilities.sum() - 1.0 / self.population_size
        
        gradient_theta = np.zeros(self.total_params)
        for i in range(self.population_size):
            gradient_theta += utilities[i] * noises[i]
        self.theta += self.learning_rate * self.sigma * gradient_theta
        
        gradient_sigma = np.zeros(self.total_params)
        for i in range(self.population_size):
            gradient_sigma += utilities[i] * (noises[i]**2 - 1) / self.sigma
        grad_clip = np.clip(self.learning_rate / 2 * gradient_sigma, -5, 5)
        self.sigma *= np.exp(grad_clip)
        self.sigma = np.clip(self.sigma, 1e-8, 1.0)
        
        best_idx = np.argmin(fitnesses)
        best_fitness = fitnesses[best_idx]
        if best_fitness < self.best_loss:
            self.best_loss = best_fitness
            self.best_params = candidates[best_idx].copy()
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        return best_fitness, fitnesses.mean(), fitnesses.std()

    def fit(self, generations=1000):
        print(f"Generation {generations}")
        print(f"Devices: {self.device}")
        print(f"Number of Parameters: {self.total_params}")
        print(f"Population Size: {self.population_size}")
        print("-" * 50)
        
        for generation in range(1, generations + 1):
            best_fitness, mean_fitness, std_fitness = self.snes_step()
            self._set_flat_params(self.best_params)
            print(f"Gen {generation:4d}/{generations}: "
                  f"Best = {best_fitness:.6f} | "
                  f"Mean = {mean_fitness:.6f} | "
                  f"Std = {std_fitness:.6f} | "
                  f"Sigma_mean = {self.sigma.mean():.6f}")
            
            if generation % 10 == 0 or best_fitness == self.best_loss:
                self.save()
            if self.stagnation_count >= self.patience:
                print(f"Early stopping at gen {generation} due to no improvement for {self.patience} generations.")
                break

        self._set_flat_params(self.best_params)
        self.save()
        print("\nFinish!")
        print(f"Best Loss: {self.best_loss:.6f}")

    def save(self, path=None):
        save_path = path if path is not None else self.save_path
        dir_name = os.path.dirname(save_path)
        if dir_name != '':
            os.makedirs(dir_name, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            #'best_loss': self.best_loss,
            #'theta': self.theta,
            #'sigma': self.sigma,
            #'best_params': self.best_params,
            'para': self.model.para
        }, save_path)