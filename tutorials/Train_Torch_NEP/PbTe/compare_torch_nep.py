from wizard.torchNEP.interface import NEPCalculator
from wizard.tools import plot_force_results
from calorine.calculators import CPUNEP
from wizard.io import read_xyz

frames = read_xyz("train.xyz")
# Use the same nep.txt for both Torch and NEP_CPU to verify alignment
calc1 = NEPCalculator("nep.txt")
calc2 = NEPCalculator("nep_model.pt")
calc3 = CPUNEP("nep.txt")
calcs = [calc1, calc2, calc3]
plot_force_results(frames, calcs, ['NEP-torch-txt', 'NEP-torch','NEP-CPU'])
