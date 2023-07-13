import numpy as np

def dump_dimer(symbol1, symbol2, distance):
    f = open('model.xyz', 'a')
    Out_string = ""
    Out_string += "2\n"
    Out_string += "pbc=\"T T T\" "
    Out_string += "Lattice=\"30.0 0.0 0.0 0.0 18.0 0.0 0.0 0.0 17.0\" "
    Out_string += "Properties=species:S:1:pos:R:3\n"
    Out_string += '{:2} 0 0 0\n'.format(symbol1)
    Out_string += '{:2} {} 0 0\n'.format(symbol2, distance)
    f.write(Out_string)
    f.close()

start = 1.2
end = 2.0
step = 0.1
symbol1 = "Fe"
symbol2 = "Fe"

sequence = np.arange(start, end + step, step)

for distance in sequence:
    dump_dimer(symbol1=symbol1, symbol2=symbol2, distance=distance)
    
