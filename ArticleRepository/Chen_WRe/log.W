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
pair_coeff * * WRe.eam.fs W
mass 1 183.839999952708211594654130749404
variable pxx equal pxx
variable pyy equal pyy
variable pzz equal pzz
variable pxy equal pxy
variable pxz equal pxz
variable pyz equal pyz
thermo_style custom pe pxx emol ecoul
variable fx atom fx
variable fy atom fy
variable fz atom fz
variable pe equal pe
neigh_modify delay 0 every 1 check yes
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Neighbor list info ...
  update every 1 steps, delay 0 steps, check yes
  max neighbors/atom: 2000, page size: 100000
  master list distance cutoff = 7.4
  ghost atom cutoff = 7.4
  binsize = 3.7, bins = 1 1 1
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair eam/fs, perpetual
      attributes: half, newton on
      pair build: half/bin/newton/tri
      stencil: half/bin/3d/tri
      bin: standard
Per MPI rank memory allocation (min/avg/max) = 3.242 | 3.242 | 3.242 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.789322     -56075.709      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 2 atoms

290.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.375e-06  |            |       |100.00

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            339 ave         339 max         339 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            112 ave         112 max         112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 112
Ave neighs/atom = 56
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 2.8444521820857 y final 0 2.8444521820857 z final 0 2.8444521820857      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8444522 3.185 3.185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8444522 2.8444522 3.185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8444522 2.8444522 2.8444522) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8444522 2.8444522 2.8444522) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8444522 2.8444522 2.8444522) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8444522 2.8444522 2.8444522) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.208583      2628301.9      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 2 atoms

126.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.92e-07   |            |       |100.00

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            557 ave         557 max         557 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            168 ave         168 max         168 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 168
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.1619864034940255 y final 0 3.1619864034940255 z final 0 3.1619864034940255      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1619864 2.8444522 2.8444522) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1619864 3.1619864 2.8444522) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1619864 3.1619864 3.1619864) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1619864 3.1619864 3.1619864) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1619864 3.1619864 3.1619864) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1619864 3.1619864 3.1619864) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.799669      9500.9918      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 2 atoms

218.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.17e-07   |            |       |100.00

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            339 ave         339 max         339 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            112 ave         112 max         112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 112
Ave neighs/atom = 56
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.165248975367204 y final 0 3.165248975367204 z final 0 3.165248975367204      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.165249 3.1619864 3.1619864) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.165249 3.165249 3.1619864) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.165249 3.165249 3.165249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.165249 3.165249 3.165249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.165249 3.165249 3.165249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.165249 3.165249 3.165249) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.799954     -143.35241      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 2 atoms

120.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.33e-07   |            |       |100.00

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            339 ave         339 max         339 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            112 ave         112 max         112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 112
Ave neighs/atom = 56
Neighbor list builds = 0
Dangerous builds = 0
change_box all boundary s s s
Changing box ...
WARNING: Resetting image flags for non-periodic dimensions (src/src/domain.cpp:1928)
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 0.0 y final 0 0.0 z final 0 0.0      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
ERROR on proc 0: Box bounds are invalid or missing (src/src/domain.cpp:210)
Last command: change_box all     x final 0 0.0 y final 0 0.0 z final 0 0.0      xy final 0.0 xz final 0.0 yz final 0.0 units box
Total wall time: 0:00:00
