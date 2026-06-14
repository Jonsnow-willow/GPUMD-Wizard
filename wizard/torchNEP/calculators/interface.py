from ase.calculators.calculator import Calculator, all_changes
from wizard.torchNEP.datasets.dataset import StructureDataset, collate_fn
from wizard.torchNEP.nep.model import NEP
import numpy as np
import torch

class NEPCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_path, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        model_path = str(model_path)
        if model_path.endswith(".txt"):
            self.model = NEP.from_nep_txt(model_path, device=self.device)
        else:
            self.model = NEP.from_checkpoint(model_path, device=self.device)
        self.para = self.model.para
        self.model.eval()    

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        batch = self.ase_atoms_to_batch(atoms)
        batch = self._batch_to_device(batch)
        batch["compute_virial"] = "stress" in properties
        batch['positions'].requires_grad_(True)
        pred = self.model(batch)

        if 'energy' in pred:
            self.results['energy'] = float(pred['energy'].cpu().detach().numpy()[0])
        if 'forces' in pred:
            self.results['forces'] = pred['forces'].cpu().detach().numpy()
        if 'virial' in pred and "stress" in properties:
            volume = atoms.get_volume()
            if volume <= 0:
                raise ValueError("Stress calculation requires a positive cell volume.")
            virial = pred['virial'].cpu().detach().numpy()[0]
            virial = np.array(virial).reshape(3, 3)
            self.results['stress'] = - virial / volume
    
    def ase_atoms_to_batch(self, atoms):
        item = StructureDataset([atoms], self.para, require_forces=False).data
        batch = collate_fn(item)
        return batch

    def _batch_to_device(self, batch):
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
