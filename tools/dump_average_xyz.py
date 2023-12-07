from wizard.io import read_xyz, dump_xyz
from ase import Atoms

num_beads = 32
num_frames = 9

pimd = []
for i in range(num_beads):
    pimd.append(read_xyz('beads_dump_%d.xyz' % i))

frames = []

for i in range(num_frames):
    symbols = pimd[0][i].get_chemical_symbols()

    cell = []
    for j in range(3):
        a = 0
        b = 0
        c = 0
        for k in range(num_beads):
            a += pimd[k][i].cell[j][0]
            b += pimd[k][i].cell[j][1]
            c += pimd[k][i].cell[j][2]
        a /= num_beads
        b /= num_beads
        c /= num_beads
        cell.append([a, b, c])

    positions = []
    for j in range(len(symbols)):
        x = 0
        y = 0
        z = 0
        for k in range(num_beads):
            x += pimd[k][i].positions[j][0]
            y += pimd[k][i].positions[j][1]
            z += pimd[k][i].positions[j][2]
        x /= num_beads
        y /= num_beads
        z /= num_beads
        positions.append([x, y, z])

    forces = []
    for j in range(len(symbols)):
        fx = 0
        fy = 0
        fz = 0
        for k in range(num_beads):
            fx += pimd[k][i].info['forces'][j][0]
            fy += pimd[k][i].info['forces'][j][1]
            fz += pimd[k][i].info['forces'][j][2]
        fx /= num_beads
        fy /= num_beads
        fz /= num_beads
        forces.append([fx, fy, fz])

    velocities = []
    for j in range(len(symbols)):
        vx = 0
        vy = 0
        vz = 0
        for k in range(num_beads):
            vx += pimd[k][i].info['velocities'][j][0]
            vy += pimd[k][i].info['velocities'][j][1]
            vz += pimd[k][i].info['velocities'][j][2]
        vx /= num_beads
        vy /= num_beads
        vz /= num_beads
        velocities.append([vx, vy, vz])

    frames.append(Atoms(symbols=symbols, positions=positions, cell=cell, pbc="T T T", info={'forces': forces, 'velocities': velocities}))

for atoms in frames:
    dump_xyz('average.xyz', atoms)
