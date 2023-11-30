from pynep.io import load_nep, dump_nep
from pynep.calculate import NEP
from wizard.frames import MultiMol
from wizard.io import plot_force_results

frames = load_nep('train.xyz', ftype='exyz')
calc = NEP('nep.txt')
for atoms in frames:
    atoms.calc = calc

select_set, split_set = MultiMol(frames).select_by_error(0.3, 0.5)
dump_nep('select.xyz', select_set, ftype='exyz')
dump_nep('split.xyz', split_set, ftype='exyz')
plot_force_results(select_set, [calc], ['error_[0.3,0.5]'])

random_set, _ = MultiMol(select_set).select_random(100)
dump_nep('random.xyz', random_set, ftype='exyz')





