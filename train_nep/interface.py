from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers
import torch
from .dataset import StructureDataset, collate_fn
from .model import NEP

class NEPCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_path, para=None, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.para = para
        self.model = NEP(para)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        batch = self.ase_atoms_to_batch(atoms)
        batch['positions'].requires_grad_(True)
        pred = self.model(batch)

        if 'energy' in pred:
            self.results['energy'] = float(pred['energy'].cpu().detach().numpy()[0])
        if 'forces' in pred:
            self.results['forces'] = pred['forces'].cpu().detach().numpy()
        if 'virial' in pred:
            self.results['stress'] = pred['virial'].cpu().detach().numpy()[0]
    
    def ase_atoms_to_batch(self, atoms):
        item = StructureDataset.process(self, atoms)
        batch = collate_fn([item])
        return batch
