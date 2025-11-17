from wizard.torchNEP.interface import NEPCalculator
from wizard.tools import plot_force_results
from calorine.calculators import CPUNEP
from wizard.io import read_xyz

frames = read_xyz("PbTe/train.xyz")
calc1 = NEPCalculator("PbTe/nep_model.pt")
calc2 = CPUNEP("PbTe/nep.txt")
calcs = [calc1, calc2]
plot_force_results(frames, calcs, ['NEP-torch', 'NEP-CPU'])