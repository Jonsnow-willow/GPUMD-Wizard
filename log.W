LAMMPS (23 Jun 2022 - Update 4)
units metal
atom_style atomic
atom_modify map array sort 0 0
boundary p p p
box tilt large
region cell prism    0 3.185 0 3.185 0 3.185     0.0 0.0 0.0     units box
create_box 1 cell
Created triclinic box = (0 0 0) to (3.185 3.185 3.185) with tilt (0 0 0)
  1 by 1 by 1 MPI processor grid
echo none
Created 2 atoms
  using lattice units in triclinic box = (0 0 0) to (3.185 3.185 3.185) with tilt (0 0 0)
  create_atoms CPU = 0.000 seconds
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
pair_style eam/fs
pair_coeff * * WRe.eam.fs W Re
ERROR: Incorrect args for pair coefficients (src/src/MANYBODY/pair_eam_fs.cpp:52)
Last command: pair_coeff * * WRe.eam.fs W Re
Total wall time: 0:00:00
