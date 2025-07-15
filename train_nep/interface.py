from ase.calculators.calculator import Calculator, all_changes
from .dataset import StructureDataset, collate_fn
from .model import NEP

class NEPCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_path, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.model = NEP.from_checkpoint(model_path, device=device)
        self.para = self.model.para
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
        item = StructureDataset([atoms], self.para).data
        batch = collate_fn(item)
        return batch
