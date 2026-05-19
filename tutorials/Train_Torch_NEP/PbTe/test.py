from wizard.torchNEP.interface import NEPCalculator
from wizard.utils.tools import plot_force_results
from calorine.calculators import CPUNEP
from wizard.utils.io import read_xyz

frames = read_xyz("train.xyz")
calc1 = NEPCalculator("exports/nep.txt")
calc2 = NEPCalculator("checkpoints/last.pt")
calc3 = CPUNEP("exports/nep.txt")
calcs = [calc1, calc2, calc3]
plot_force_results(frames, calcs, ['NEP-torch-txt', 'NEP-torch','NEP-CPU'])