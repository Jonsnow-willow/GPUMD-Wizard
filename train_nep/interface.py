from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers
import torch
from .config import get_nep_config
from .dataset import StructureDataset, collate_fn

class NEPCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model, device='cpu', para=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        if para is None:
            para = get_nep_config()
        self.para = para
        elements = para["elements"]
        element_atomic_numbers = [atomic_numbers[element] for element in elements]
        self.z2id = {z: idx for idx, z in enumerate(element_atomic_numbers)}

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        batch = self.ase_atoms_to_batch(atoms)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        with torch.no_grad():
            pred = self.model(batch)
        if 'energy' in pred:
            self.results['energy'] = float(pred['energy'].cpu().numpy()[0])
        if 'forces' in pred:
            self.results['forces'] = pred['forces'].cpu().numpy()[0]
        if 'virial' in pred:
            self.results['stress'] = pred['virial'].cpu().numpy()[0]
    
    def ase_atoms_to_batch(self, atoms):
        item = StructureDataset.process(self, atoms)
        batch = collate_fn([item])
        return batch
