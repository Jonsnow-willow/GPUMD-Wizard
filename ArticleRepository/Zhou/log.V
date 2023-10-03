LAMMPS (23 Jun 2022 - Update 4)
units metal
atom_style atomic
atom_modify map array sort 0 0
boundary p p p
box tilt large
region cell prism    0 2.997 0 2.997 0 2.997     0.0 0.0 0.0     units box
create_box 1 cell
Created triclinic box = (0 0 0) to (2.997 2.997 2.997) with tilt (0 0 0)
  1 by 1 by 1 MPI processor grid
echo none
Created 2 atoms
  using lattice units in triclinic box = (0 0 0) to (2.997 2.997 2.997) with tilt (0 0 0)
  create_atoms CPU = 0.000 seconds
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
pair_style eam/alloy
pair_coeff * * potentials/V.eam.alloy V
Reading eam/alloy potential file potentials/V.eam.alloy with DATE: 2018-03-30
mass 1 50.941499986895586005175573518500
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
  master list distance cutoff = 7.8018406
  ghost atom cutoff = 7.8018406
  binsize = 3.9009203, bins = 1 1 1
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair eam/alloy, perpetual
      attributes: half, newton on
      pair build: half/bin/newton/tri
      stencil: half/bin/3d/tri
      bin: standard
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.607154      50917.658      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 2 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

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
change_box all     x final 0 3.264678999141974 y final 0 3.264678999141974 z final 0 3.264678999141974      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.264679 2.997 2.997) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.264679 3.264679 2.997) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.264679 3.264679 3.264679) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.264679 3.264679 3.264679) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.264679 3.264679 3.264679) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.264679 3.264679 3.264679) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-9.9991319     -261162.72      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 2 atoms

114.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

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
change_box all     x final 0 3.031543713883322 y final 0 3.031543713883322 z final 0 3.031543713883322      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0315437 3.264679 3.264679) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0315437 3.0315437 3.264679) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0315437 3.0315437 3.0315437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0315437 3.0315437 3.0315437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0315437 3.0315437 3.0315437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0315437 3.0315437 3.0315437) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.619877     -6017.8738      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 2 atoms

327.2% CPU use with 1 MPI tasks x no OpenMP threads

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
Nghost:            557 ave         557 max         557 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            136 ave         136 max         136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 136
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.0277595293175072 y final 0 3.0277595293175072 z final 0 3.0277595293175072      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0277595 3.0315437 3.0315437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0277595 3.0277595 3.0315437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0277595 3.0277595 3.0277595) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0277595 3.0277595 3.0277595) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0277595 3.0277595 3.0277595) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0277595 3.0277595 3.0277595) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.620079     -180.34479      0              0            
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
Nghost:            557 ave         557 max         557 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            136 ave         136 max         136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 136
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 2.7249835763857564 y final 0 2.7249835763857564 z final 0 2.7249835763857564      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.7249836 3.0277595 3.0277595) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7249836 2.7249836 3.0277595) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7249836 2.7249836 2.7249836) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7249836 2.7249836 2.7249836) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7249836 2.7249836 2.7249836) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7249836 2.7249836 2.7249836) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-9.393421       588084.56      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 2 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            557 ave         557 max         557 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            180 ave         180 max         180 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 180
Ave neighs/atom = 90
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 2.7552611716789315 y final 0 2.7552611716789315 z final 0 2.7552611716789315      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.7552612 2.7249836 2.7249836) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7552612 2.7552612 2.7249836) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7552612 2.7552612 2.7552612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7552612 2.7552612 2.7552612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7552612 2.7552612 2.7552612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7552612 2.7552612 2.7552612) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-9.6266316      509109.21      0              0            
Loop time of 6.25e-07 on 1 procs for 0 steps with 2 atoms

160.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            557 ave         557 max         557 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            180 ave         180 max         180 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 180
Ave neighs/atom = 90
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 2.785538766972107 y final 0 2.785538766972107 z final 0 2.785538766972107      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.7855388 2.7552612 2.7552612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7855388 2.7855388 2.7552612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7855388 2.7855388 2.7855388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7855388 2.7855388 2.7855388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7855388 2.7855388 2.7855388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7855388 2.7855388 2.7855388) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-9.8320962      436495.97      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 2 atoms

109.1% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 2.815816362265282 y final 0 2.815816362265282 z final 0 2.815816362265282      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8158164 2.7855388 2.7855388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8158164 2.8158164 2.7855388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8158164 2.8158164 2.8158164) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8158164 2.8158164 2.8158164) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8158164 2.8158164 2.8158164) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8158164 2.8158164 2.8158164) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.011023      369320.48      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 2 atoms

104.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.59e-07   |            |       |100.00

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
change_box all     x final 0 2.846093957558457 y final 0 2.846093957558457 z final 0 2.846093957558457      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.846094 2.8158164 2.8158164) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.846094 2.846094 2.8158164) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.846094 2.846094 2.846094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.846094 2.846094 2.846094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.846094 2.846094 2.846094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.846094 2.846094 2.846094) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.164457      307068.27      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 2 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 2.8763715528516323 y final 0 2.8763715528516323 z final 0 2.8763715528516323      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8763716 2.846094 2.846094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8763716 2.8763716 2.846094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8763716 2.8763716 2.8763716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8763716 2.8763716 2.8763716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8763716 2.8763716 2.8763716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8763716 2.8763716 2.8763716) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.293362      249309.51      0              0            
Loop time of 6.67e-07 on 1 procs for 0 steps with 2 atoms

149.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.67e-07   |            |       |100.00

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
change_box all     x final 0 2.9066491481448073 y final 0 2.9066491481448073 z final 0 2.9066491481448073      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9066491 2.8763716 2.8763716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9066491 2.9066491 2.8763716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9066491 2.9066491 2.9066491) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9066491 2.9066491 2.9066491) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9066491 2.9066491 2.9066491) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9066491 2.9066491 2.9066491) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.399243      253618.09      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 2 atoms

208.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

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
change_box all     x final 0 2.9369267434379824 y final 0 2.9369267434379824 z final 0 2.9369267434379824      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9369267 2.9066491 2.9066491) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9369267 2.9369267 2.9066491) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9369267 2.9369267 2.9369267) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9369267 2.9369267 2.9369267) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9369267 2.9369267 2.9369267) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9369267 2.9369267 2.9369267) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.499821      171752.39      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 2 atoms

109.1% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 2.9672043387311575 y final 0 2.9672043387311575 z final 0 2.9672043387311575      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9672043 2.9369267 2.9369267) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9672043 2.9672043 2.9369267) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9672043 2.9672043 2.9672043) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9672043 2.9672043 2.9672043) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9672043 2.9672043 2.9672043) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9672043 2.9672043 2.9672043) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.568298      107052.85      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 2 atoms

342.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

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
change_box all     x final 0 2.9974819340243326 y final 0 2.9974819340243326 z final 0 2.9974819340243326      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9974819 2.9672043 2.9672043) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9974819 2.9974819 2.9672043) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9974819 2.9974819 2.9974819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9974819 2.9974819 2.9974819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9974819 2.9974819 2.9974819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9974819 2.9974819 2.9974819) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.607563      50065.364      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 2 atoms

133.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

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
change_box all     x final 0 3.0277595293175072 y final 0 3.0277595293175072 z final 0 3.0277595293175072      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0277595 2.9974819 2.9974819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0277595 3.0277595 2.9974819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0277595 3.0277595 3.0277595) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0277595 3.0277595 3.0277595) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0277595 3.0277595 3.0277595) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0277595 3.0277595 3.0277595) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.620079     -180.34479      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 2 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            557 ave         557 max         557 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            136 ave         136 max         136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 136
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.058037124610683 y final 0 3.058037124610683 z final 0 3.058037124610683      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0580371 3.0277595 3.0277595) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0580371 3.0580371 3.0277595) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0580371 3.0580371 3.0580371) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0580371 3.0580371 3.0580371) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0580371 3.0580371 3.0580371) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0580371 3.0580371 3.0580371) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.608118     -44243.622      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 2 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            557 ave         557 max         557 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            136 ave         136 max         136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 136
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.0883147199038574 y final 0 3.0883147199038574 z final 0 3.0883147199038574      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0883147 3.0580371 3.0580371) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0883147 3.0883147 3.0580371) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0883147 3.0883147 3.0883147) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0883147 3.0883147 3.0883147) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0883147 3.0883147 3.0883147) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0883147 3.0883147 3.0883147) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.57528      -71269.506      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 2 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

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
Nghost:            557 ave         557 max         557 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            136 ave         136 max         136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 136
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.1185923151970334 y final 0 3.1185923151970334 z final 0 3.1185923151970334      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1185923 3.0883147 3.0883147) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1185923 3.1185923 3.0883147) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1185923 3.1185923 3.1185923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1185923 3.1185923 3.1185923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1185923 3.1185923 3.1185923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1185923 3.1185923 3.1185923) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.526841     -106357.48      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 2 atoms

119.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.34e-07   |            |       |100.00

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            557 ave         557 max         557 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            136 ave         136 max         136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 136
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.1488699104902076 y final 0 3.1488699104902076 z final 0 3.1488699104902076      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1488699 3.1185923 3.1185923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1488699 3.1488699 3.1185923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1488699 3.1488699 3.1488699) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1488699 3.1488699 3.1488699) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1488699 3.1488699 3.1488699) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1488699 3.1488699 3.1488699) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.457606     -142178.01      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 2 atoms

218.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.16e-07   |            |       |100.00

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            339 ave         339 max         339 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            136 ave         136 max         136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 136
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.1791475057833836 y final 0 3.1791475057833836 z final 0 3.1791475057833836      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1791475 3.1488699 3.1488699) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1791475 3.1791475 3.1488699) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1791475 3.1791475 3.1791475) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1791475 3.1791475 3.1791475) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1791475 3.1791475 3.1791475) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1791475 3.1791475 3.1791475) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.366954     -176849.13      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 2 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            339 ave         339 max         339 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            136 ave         136 max         136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 136
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.2094251010765578 y final 0 3.2094251010765578 z final 0 3.2094251010765578      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2094251 3.1791475 3.1791475) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2094251 3.2094251 3.1791475) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2094251 3.2094251 3.2094251) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2094251 3.2094251 3.2094251) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2094251 3.2094251 3.2094251) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2094251 3.2094251 3.2094251) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.255127     -209239.79      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 2 atoms

400.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

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
change_box all     x final 0 3.2397026963697337 y final 0 3.2397026963697337 z final 0 3.2397026963697337      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2397027 3.2094251 3.2094251) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2397027 3.2397027 3.2094251) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2397027 3.2397027 3.2397027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2397027 3.2397027 3.2397027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2397027 3.2397027 3.2397027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2397027 3.2397027 3.2397027) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.122886     -238849.95      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 2 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

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
change_box all     x final 0 3.269980291662908 y final 0 3.269980291662908 z final 0 3.269980291662908      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2699803 3.2397027 3.2397027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2699803 3.2699803 3.2397027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2699803 3.2699803 3.2699803) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2699803 3.2699803 3.2699803) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2699803 3.2699803 3.2699803) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2699803 3.2699803 3.2699803) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-9.9712173     -265667.53      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 2 atoms

171.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.166e-06  |            |       |100.00

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
change_box all     x final 0 3.300257886956084 y final 0 3.300257886956084 z final 0 3.300257886956084      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3002579 3.2699803 3.2699803) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3002579 3.3002579 3.2699803) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3002579 3.3002579 3.3002579) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3002579 3.3002579 3.3002579) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3002579 3.3002579 3.3002579) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3002579 3.3002579 3.3002579) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-9.8010859     -290015.98      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 2 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

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
change_box all     x final 0 3.330535482249258 y final 0 3.330535482249258 z final 0 3.330535482249258      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3305355 3.3002579 3.3002579) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3305355 3.3305355 3.3002579) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3305355 3.3305355 3.3305355) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3305355 3.3305355 3.3305355) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3305355 3.3305355 3.3305355) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3305355 3.3305355 3.3305355) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-9.6132724     -312404.57      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 2 atoms

80.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.25e-06   |            |       |100.00

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
change_box all     x final 0 9.083278587952522 y final 0 9.083278587952522 z final 0 9.083278587952522      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.0832786 3.3305355 3.3305355) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 3.3305355) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
echo none
Created 52 atoms
  using lattice units in triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
  create_atoms CPU = 0.000 seconds
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-286.74214     -180.34479      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 54 atoms

154.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.292e-06  |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3672 ave        3672 max        3672 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3672
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 8.90161301619347 y final 0 9.083278587952522 z final 0 9.083278587952522      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.901613 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.901613 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.901613 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.901613 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.901613 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.901613 9.0832786 9.0832786) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-286.53423      43687.425      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 54 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.59e-07   |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3888 ave        3888 max        3888 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3888
Ave neighs/atom = 72
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 8.992445802072996 y final 0 9.083278587952522 z final 0 9.083278587952522      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9924458 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9924458 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9924458 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9924458 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9924458 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9924458 9.0832786 9.0832786) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-286.68954      22462.665      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 54 atoms

114.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3888 ave        3888 max        3888 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3888
Ave neighs/atom = 72
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.083278587952522 y final 0 9.083278587952522 z final 0 9.083278587952522      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-286.74214     -180.34479      0              0            
Loop time of 5.42e-07 on 1 procs for 0 steps with 54 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.42e-07   |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3672 ave        3672 max        3672 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3672
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.174111373832048 y final 0 9.083278587952522 z final 0 9.083278587952522      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.1741114 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.1741114 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.1741114 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.1741114 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.1741114 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.1741114 9.0832786 9.0832786) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-286.68627     -23829.554      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 54 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3672 ave        3672 max        3672 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3672
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.264944159711574 y final 0 9.083278587952522 z final 0 9.083278587952522      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.2649442 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.2649442 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.2649442 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.2649442 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.2649442 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.2649442 9.0832786 9.0832786) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-286.51866     -47833.169      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 54 atoms

80.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.25e-06   |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3672 ave        3672 max        3672 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3672
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.083278587952522 y final 0 9.083296754491531 z final 0 9.083260421449847      xy final 0.0 xz final 0.0 yz final 0.018166520842899696 units box
Changing box ...
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832968 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832968 9.0832604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832968 9.0832604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832968 9.0832604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0832786 9.0832968 9.0832604) with tilt (0 0 0.018166521)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-286.74171     -173.60504      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 54 atoms

88.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3672 ave        3672 max        3672 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3672
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.083278587952522 y final 0 9.083470470185958 z final 0 9.08308670977247      xy final 0.0 xz final 0.0 yz final 0.05904006361352106 units box
Changing box ...
  triclinic box = (0 0 0) to (9.0832786 9.0832968 9.0832604) with tilt (0 0 0.018166521)
  triclinic box = (0 0 0) to (9.0832786 9.0834705 9.0832604) with tilt (0 0 0.018166521)
  triclinic box = (0 0 0) to (9.0832786 9.0834705 9.0830867) with tilt (0 0 0.018166521)
  triclinic box = (0 0 0) to (9.0832786 9.0834705 9.0830867) with tilt (0 0 0.018166521)
  triclinic box = (0 0 0) to (9.0832786 9.0834705 9.0830867) with tilt (0 0 0.018166521)
  triclinic box = (0 0 0) to (9.0832786 9.0834705 9.0830867) with tilt (0 0 0.059040064)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-286.73761     -109.15074      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 54 atoms

104.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3672 ave        3672 max        3672 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3672
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.083278587952522 y final 0 9.083828109684564 z final 0 9.08272909946353      xy final 0.0 xz final 0.0 yz final 0.09991002009409884 units box
Changing box ...
  triclinic box = (0 0 0) to (9.0832786 9.0834705 9.0830867) with tilt (0 0 0.059040064)
  triclinic box = (0 0 0) to (9.0832786 9.0838281 9.0830867) with tilt (0 0 0.059040064)
  triclinic box = (0 0 0) to (9.0832786 9.0838281 9.0827291) with tilt (0 0 0.059040064)
  triclinic box = (0 0 0) to (9.0832786 9.0838281 9.0827291) with tilt (0 0 0.059040064)
  triclinic box = (0 0 0) to (9.0832786 9.0838281 9.0827291) with tilt (0 0 0.059040064)
  triclinic box = (0 0 0) to (9.0832786 9.0838281 9.0827291) with tilt (0 0 0.09991002)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-286.72915      23.536027      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 54 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.16e-07   |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3672 ave        3672 max        3672 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3672
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.083278587952522 y final 0 9.084369651264845 z final 0 9.082187655680569      xy final 0.0 xz final 0.0 yz final 0.14077390866304884 units box
Changing box ...
  triclinic box = (0 0 0) to (9.0832786 9.0838281 9.0827291) with tilt (0 0 0.09991002)
  triclinic box = (0 0 0) to (9.0832786 9.0843697 9.0827291) with tilt (0 0 0.09991002)
  triclinic box = (0 0 0) to (9.0832786 9.0843697 9.0821877) with tilt (0 0 0.09991002)
  triclinic box = (0 0 0) to (9.0832786 9.0843697 9.0821877) with tilt (0 0 0.09991002)
  triclinic box = (0 0 0) to (9.0832786 9.0843697 9.0821877) with tilt (0 0 0.09991002)
  triclinic box = (0 0 0) to (9.0832786 9.0843697 9.0821877) with tilt (0 0 0.14077391)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-286.71635      224.43322      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 54 atoms

92.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3672 ave        3672 max        3672 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3672
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.083278587952522 y final 0 9.085095062040864 z final 0 9.081462477050044      xy final 0.0 xz final 0.0 yz final 0.1816292495410009 units box
Changing box ...
  triclinic box = (0 0 0) to (9.0832786 9.0843697 9.0821877) with tilt (0 0 0.14077391)
  triclinic box = (0 0 0) to (9.0832786 9.0850951 9.0821877) with tilt (0 0 0.14077391)
  triclinic box = (0 0 0) to (9.0832786 9.0850951 9.0814625) with tilt (0 0 0.14077391)
  triclinic box = (0 0 0) to (9.0832786 9.0850951 9.0814625) with tilt (0 0 0.14077391)
  triclinic box = (0 0 0) to (9.0832786 9.0850951 9.0814625) with tilt (0 0 0.14077391)
  triclinic box = (0 0 0) to (9.0832786 9.0850951 9.0814625) with tilt (0 0 0.18162925)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-286.69922      493.50433      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 54 atoms

109.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.16e-07   |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3672 ave        3672 max        3672 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3672
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.083278587952522 y final 0 9.083278587952522 z final 0 9.083278587952522      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.0832786 9.0850951 9.0814625) with tilt (0 0 0.18162925)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0814625) with tilt (0 0 0.18162925)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0.18162925)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0.18162925)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0.18162925)
  triclinic box = (0 0 0) to (9.0832786 9.0832786 9.0832786) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-286.74185     -182.93382      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 54 atoms

177.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3672 ave        3672 max        3672 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3672
Ave neighs/atom = 68
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 12.111038117270029 y final 0 15.138797646587536 z final 0 18.166557175905044      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.111038 9.0832786 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 9.0832786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
echo none
Created 185 atoms
  using lattice units in triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  create_atoms CPU = 0.000 seconds
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.927      -1401.5965      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 239 atoms

85.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.167e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 9.915679358308633 y final 0 12.394599197885793 z final 0 14.873519037462952      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.9156794 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-711.65521      1990001.1      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 239 atoms

88.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2789 ave        2789 max        2789 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          30702 ave       30702 max       30702 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 30702
Ave neighs/atom = 128.46025
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.10566010569326 y final 0 15.132075132116576 z final 0 18.158490158539895      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.10566 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10566 15.132075 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10566 15.132075 18.15849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10566 15.132075 18.15849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10566 15.132075 18.15849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10566 15.132075 18.15849) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.928       654.43548      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 239 atoms

177.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.10566010569326 y final 0 15.132075132116578 z final 0 18.158490158539895      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.10566 15.132075 18.15849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10566 15.132075 18.15849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10566 15.132075 18.15849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10566 15.132075 18.15849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10566 15.132075 18.15849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10566 15.132075 18.15849) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.928       654.43548      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 239 atoms

184.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.107110778595155 y final 0 15.133888473243966 z final 0 18.160666167892746      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.107111 15.132075 18.15849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107111 15.133888 18.15849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107111 15.133888 18.160666) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107111 15.133888 18.160666) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107111 15.133888 18.160666) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107111 15.133888 18.160666) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9457      3.6621782      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 239 atoms

141.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.08e-07   |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.109164086409319 y final 0 15.136455108011694 z final 0 18.163746129614005      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.109164 15.133888 18.160666) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.109164 15.136455 18.160666) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.109164 15.136455 18.163746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.109164 15.136455 18.163746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.109164 15.136455 18.163746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.109164 15.136455 18.163746) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9548     -931.95707      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 239 atoms

141.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.08e-07   |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.109164086409319 y final 0 15.136455108011695 z final 0 18.163746129614005      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.109164 15.136455 18.163746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.109164 15.136455 18.163746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.109164 15.136455 18.163746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.109164 15.136455 18.163746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.109164 15.136455 18.163746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.109164 15.136455 18.163746) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9548     -931.95707      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 239 atoms

363.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.375e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.107056363546306 y final 0 15.133820454432938 z final 0 18.160584545319487      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.107056 15.136455 18.163746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107056 15.13382 18.163746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107056 15.13382 18.160585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107056 15.13382 18.160585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107056 15.13382 18.160585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107056 15.13382 18.160585) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9584     -181.48967      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 239 atoms

228.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.107056363546306 y final 0 15.13382045443294 z final 0 18.160584545319487      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.107056 15.13382 18.160585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107056 15.13382 18.160585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107056 15.13382 18.160585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107056 15.13382 18.160585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107056 15.13382 18.160585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107056 15.13382 18.160585) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9584     -181.48967      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 239 atoms

232.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.292e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.106351086206603 y final 0 15.132938857758315 z final 0 18.159526629309937      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106351 15.13382 18.160585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106351 15.132939 18.160585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106351 15.132939 18.159527) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106351 15.132939 18.159527) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106351 15.132939 18.159527) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106351 15.132939 18.159527) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9592      68.779017      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 239 atoms

88.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.106351086206603 y final 0 15.132938857758317 z final 0 18.159526629309937      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106351 15.132939 18.159527) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106351 15.132939 18.159527) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106351 15.132939 18.159527) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106351 15.132939 18.159527) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106351 15.132939 18.159527) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106351 15.132939 18.159527) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9592      68.779017      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 239 atoms

171.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.167e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.106387293636173 y final 0 15.13298411704528 z final 0 18.15958094045429      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106387 15.132939 18.159527) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106387 15.132984 18.159527) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106387 15.132984 18.159581) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106387 15.132984 18.159581) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106387 15.132984 18.159581) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106387 15.132984 18.159581) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9593      56.050143      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 239 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.08e-07   |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.106387293636173 y final 0 15.132984117045282 z final 0 18.15958094045429      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106387 15.132984 18.159581) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106387 15.132984 18.159581) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106387 15.132984 18.159581) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106387 15.132984 18.159581) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106387 15.132984 18.159581) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106387 15.132984 18.159581) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9593      56.050143      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 239 atoms

360.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.33e-07   |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.10651563471372 y final 0 15.133144543392214 z final 0 18.159773452070606      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106516 15.132984 18.159581) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106516 15.133145 18.159581) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106516 15.133145 18.159773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106516 15.133145 18.159773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106516 15.133145 18.159773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106516 15.133145 18.159773) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9594      10.07348       0              0            
Loop time of 1.417e-06 on 1 procs for 0 steps with 239 atoms

211.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.417e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.10651563471372 y final 0 15.133144543392216 z final 0 18.159773452070606      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106516 15.133145 18.159773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106516 15.133145 18.159773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106516 15.133145 18.159773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106516 15.133145 18.159773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106516 15.133145 18.159773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106516 15.133145 18.159773) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9594      10.07348       0              0            
Loop time of 1.333e-06 on 1 procs for 0 steps with 239 atoms

225.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.333e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.106553938167469 y final 0 15.133192422709405 z final 0 18.15983090725123      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106554 15.133145 18.159773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106554 15.133192 18.159773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106554 15.133192 18.159831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106554 15.133192 18.159831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106554 15.133192 18.159831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106554 15.133192 18.159831) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9594     -4.2933544      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 239 atoms

104.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.106553938167469 y final 0 15.133192422709406 z final 0 18.15983090725123      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106554 15.133192 18.159831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106554 15.133192 18.159831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106554 15.133192 18.159831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106554 15.133192 18.159831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106554 15.133192 18.159831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106554 15.133192 18.159831) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9594     -4.2933544      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 239 atoms

192.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.041e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.10655110150497 y final 0 15.133188876881286 z final 0 18.15982665225748      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106551 15.133192 18.159831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106551 15.133189 18.159831) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106551 15.133189 18.159827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106551 15.133189 18.159827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106551 15.133189 18.159827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106551 15.133189 18.159827) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9594     -4.9181539      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 239 atoms

104.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.10655110150497 y final 0 15.133188876881288 z final 0 18.15982665225748      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106551 15.133189 18.159827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106551 15.133189 18.159827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106551 15.133189 18.159827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106551 15.133189 18.159827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106551 15.133189 18.159827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106551 15.133189 18.159827) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9594     -4.9181539      0              0            
Loop time of 1.458e-06 on 1 procs for 0 steps with 239 atoms

274.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.458e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.106535171263953 y final 0 15.133168964080019 z final 0 18.15980275689595      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106535 15.133189 18.159827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106535 15.133169 18.159827) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106535 15.133169 18.159803) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106535 15.133169 18.159803) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106535 15.133169 18.159803) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106535 15.133169 18.159803) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1266.9594     -0.3885897      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 239 atoms

300.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16184 ave       16184 max       16184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16184
Ave neighs/atom = 67.715481
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.111038117270029 y final 0 15.138797646587536 z final 0 18.166557175905044      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.111038 15.133169 18.159803) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.159803) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
echo none
1 atoms in group delatoms
Deleted 1 atoms, new total = 238
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7284     -2312.6764      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 238 atoms

252.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.92e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 9.915679358308633 y final 0 12.394599197885793 z final 0 14.873519037462952      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.9156794 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-709.716        1973651.1      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 238 atoms

177.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2782 ave        2782 max        2782 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          30445 ave       30445 max       30445 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 30445
Ave neighs/atom = 127.92017
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.102142619962475 y final 0 15.127678274953093 z final 0 18.153213929943714      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.102143 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102143 15.127678 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102143 15.127678 18.153214) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102143 15.127678 18.153214) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102143 15.127678 18.153214) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102143 15.127678 18.153214) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7313      1061.1253      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 238 atoms

82.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.102142619962475 y final 0 15.127678274953094 z final 0 18.153213929943714      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.102143 15.127678 18.153214) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102143 15.127678 18.153214) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102143 15.127678 18.153214) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102143 15.127678 18.153214) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102143 15.127678 18.153214) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102143 15.127678 18.153214) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7313      1061.1253      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 238 atoms

379.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.91e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.104636418604782 y final 0 15.130795523255985 z final 0 18.156954627907176      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.104636 15.127678 18.153214) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104636 15.130796 18.153214) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104636 15.130796 18.156955) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104636 15.130796 18.156955) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104636 15.130796 18.156955) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104636 15.130796 18.156955) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7586      3.2208613      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10762891592436 y final 0 15.13453614490546 z final 0 18.161443373886538      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.107629 15.130796 18.156955) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107629 15.134536 18.156955) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107629 15.134536 18.161443) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107629 15.134536 18.161443) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107629 15.134536 18.161443) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107629 15.134536 18.161443) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7701     -1276.2521      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 238 atoms

133.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10762891592436 y final 0 15.134536144905463 z final 0 18.161443373886538      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.107629 15.134536 18.161443) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107629 15.134536 18.161443) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107629 15.134536 18.161443) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107629 15.134536 18.161443) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107629 15.134536 18.161443) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107629 15.134536 18.161443) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7701     -1276.2521      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 238 atoms

266.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.104875371551525 y final 0 15.131094214439424 z final 0 18.157313057327286      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.104875 15.134536 18.161443) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104875 15.131094 18.161443) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104875 15.131094 18.157313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104875 15.131094 18.157313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104875 15.131094 18.157313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104875 15.131094 18.157313) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7778     -364.07012      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16050 ave       16050 max       16050 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16050
Ave neighs/atom = 67.436975
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.104875371551525 y final 0 15.131094214439425 z final 0 18.157313057327286      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.104875 15.131094 18.157313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104875 15.131094 18.157313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104875 15.131094 18.157313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104875 15.131094 18.157313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104875 15.131094 18.157313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.104875 15.131094 18.157313) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7778     -364.07012      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 238 atoms

282.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.416e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16050 ave       16050 max       16050 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16050
Ave neighs/atom = 67.436975
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103415353354151 y final 0 15.129269191692712 z final 0 18.155123030031223      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103415 15.131094 18.157313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103415 15.129269 18.157313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103415 15.129269 18.155123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103415 15.129269 18.155123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103415 15.129269 18.155123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103415 15.129269 18.155123) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.78        80.193526      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16050 ave       16050 max       16050 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16050
Ave neighs/atom = 67.436975
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103415353354151 y final 0 15.129269191692714 z final 0 18.155123030031223      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103415 15.129269 18.155123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103415 15.129269 18.155123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103415 15.129269 18.155123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103415 15.129269 18.155123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103415 15.129269 18.155123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103415 15.129269 18.155123) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.78        80.193526      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 238 atoms

400.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16050 ave       16050 max       16050 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16050
Ave neighs/atom = 67.436975
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10332539121657 y final 0 15.129156739020738 z final 0 18.15498808682485      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103325 15.129269 18.155123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103325 15.129157 18.155123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103325 15.129157 18.154988) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103325 15.129157 18.154988) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103325 15.129157 18.154988) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103325 15.129157 18.154988) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7804      86.350768      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 238 atoms

218.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.375e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103245352430484 y final 0 15.129056690538132 z final 0 18.154868028645723      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103245 15.129157 18.154988) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103245 15.129057 18.154988) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103245 15.129057 18.154868) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103245 15.129057 18.154868) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103245 15.129057 18.154868) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103245 15.129057 18.154868) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7805      91.722222      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 238 atoms

191.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103245352430484 y final 0 15.129056690538134 z final 0 18.154868028645723      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103245 15.129057 18.154868) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103245 15.129057 18.154868) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103245 15.129057 18.154868) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103245 15.129057 18.154868) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103245 15.129057 18.154868) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103245 15.129057 18.154868) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7805      91.722222      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 238 atoms

80.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.25e-06   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103424514139496 y final 0 15.129280642674399 z final 0 18.15513677120924      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103425 15.129057 18.154868) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103425 15.129281 18.154868) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103425 15.129281 18.155137) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103425 15.129281 18.155137) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103425 15.129281 18.155137) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103425 15.129281 18.155137) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7807      24.17833       0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.33e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103609617910106 y final 0 15.12951202238766 z final 0 18.155414426865153      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.10361 15.129281 18.155137) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10361 15.129512 18.155137) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10361 15.129512 18.155414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10361 15.129512 18.155414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10361 15.129512 18.155414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10361 15.129512 18.155414) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7807     -46.940592      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 238 atoms

141.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.416e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103609617910106 y final 0 15.129512022387662 z final 0 18.155414426865153      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.10361 15.129512 18.155414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10361 15.129512 18.155414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10361 15.129512 18.155414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10361 15.129512 18.155414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10361 15.129512 18.155414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10361 15.129512 18.155414) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7807     -46.940592      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 238 atoms

436.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.16e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10350791421262 y final 0 15.129384892765804 z final 0 18.155261871318924      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103508 15.129512 18.155414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103508 15.129385 18.155414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103508 15.129385 18.155262) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103508 15.129385 18.155262) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103508 15.129385 18.155262) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103508 15.129385 18.155262) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7807     -35.600216      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 238 atoms

240.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.33e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103101107968522 y final 0 15.128876384960682 z final 0 18.154651661952773      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103101 15.129385 18.155262) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103101 15.128876 18.155262) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103101 15.128876 18.154652) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103101 15.128876 18.154652) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103101 15.128876 18.154652) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103101 15.128876 18.154652) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7808      17.890346      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 238 atoms

104.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103101107968522 y final 0 15.128876384960684 z final 0 18.154651661952773      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103101 15.128876 18.154652) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103101 15.128876 18.154652) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103101 15.128876 18.154652) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103101 15.128876 18.154652) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103101 15.128876 18.154652) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103101 15.128876 18.154652) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7808      17.890346      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 238 atoms

126.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.91e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103061748397115 y final 0 15.128827185496425 z final 0 18.15459262259566      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103062 15.128876 18.154652) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103062 15.128827 18.154652) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103062 15.128827 18.154593) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103062 15.128827 18.154593) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103062 15.128827 18.154593) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103062 15.128827 18.154593) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7808      23.194121      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 238 atoms

266.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103021312446579 y final 0 15.128776640558254 z final 0 18.154531968669858      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103021 15.128827 18.154593) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103021 15.128777 18.154593) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103021 15.128777 18.154532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103021 15.128777 18.154532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103021 15.128777 18.154532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103021 15.128777 18.154532) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7809      35.121865      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 238 atoms

133.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103021312446579 y final 0 15.128776640558256 z final 0 18.154531968669858      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103021 15.128777 18.154532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103021 15.128777 18.154532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103021 15.128777 18.154532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103021 15.128777 18.154532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103021 15.128777 18.154532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103021 15.128777 18.154532) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7809      35.121865      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 238 atoms

165.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10316905261828 y final 0 15.128961315772882 z final 0 18.15475357892741      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103169 15.128777 18.154532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103169 15.128961 18.154532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103169 15.128961 18.154754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103169 15.128961 18.154754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103169 15.128961 18.154754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103169 15.128961 18.154754) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7809     -9.1932327      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 238 atoms

96.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.041e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103114349876096 y final 0 15.128892937345151 z final 0 18.154671524814134      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103114 15.128961 18.154754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103114 15.128893 18.154754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103114 15.128893 18.154672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103114 15.128893 18.154672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103114 15.128893 18.154672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103114 15.128893 18.154672) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7809      6.8281621      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 238 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16056 ave       16056 max       16056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16056
Ave neighs/atom = 67.462185
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.111038117270029 y final 0 15.138797646587536 z final 0 18.166557175905044      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.111038 15.128893 18.154672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.154672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.6906     -2926.9347      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 238 atoms

413.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 9.915679358308633 y final 0 12.394599197885793 z final 0 14.873519037462952      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.9156794 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-710.40246      1973023        0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 238 atoms

184.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2778 ave        2778 max        2778 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          30446 ave       30446 max       30446 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 30446
Ave neighs/atom = 127.92437
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100326060702443 y final 0 15.125407575878052 z final 0 18.150489091053664      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100326 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100326 15.125408 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100326 15.125408 18.150489) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100326 15.125408 18.150489) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100326 15.125408 18.150489) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100326 15.125408 18.150489) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.6949      1096.8274      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 238 atoms

480.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.33e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100326060702443 y final 0 15.125407575878052 z final 0 18.150489091053664      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100326 15.125408 18.150489) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100326 15.125408 18.150489) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100326 15.125408 18.150489) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100326 15.125408 18.150489) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100326 15.125408 18.150489) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100326 15.125408 18.150489) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.6949      1096.8274      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 238 atoms

300.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103042586790485 y final 0 15.128803233488107 z final 0 18.15456388018573      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103043 15.125408 18.150489) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103043 15.128803 18.150489) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103043 15.128803 18.154564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103043 15.128803 18.154564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103043 15.128803 18.154564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103043 15.128803 18.154564) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7283     -96.548248      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 238 atoms

184.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.106726029617452 y final 0 15.133407537021814 z final 0 18.16008904442618      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106726 15.128803 18.154564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106726 15.133408 18.154564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106726 15.133408 18.160089) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106726 15.133408 18.160089) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106726 15.133408 18.160089) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106726 15.133408 18.160089) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7452     -1789.7672      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 238 atoms

240.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.33e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.106726029617452 y final 0 15.133407537021816 z final 0 18.16008904442618      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106726 15.133408 18.160089) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106726 15.133408 18.160089) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106726 15.133408 18.160089) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106726 15.133408 18.160089) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106726 15.133408 18.160089) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106726 15.133408 18.160089) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7452     -1789.7672      0              0            
Loop time of 1.333e-06 on 1 procs for 0 steps with 238 atoms

300.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.333e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.102877870559038 y final 0 15.1285973381988 z final 0 18.15431680583856      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.102878 15.133408 18.160089) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102878 15.128597 18.160089) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102878 15.128597 18.154317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102878 15.128597 18.154317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102878 15.128597 18.154317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102878 15.128597 18.154317) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7554     -537.9331       0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 238 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.102877870559038 y final 0 15.128597338198801 z final 0 18.15431680583856      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.102878 15.128597 18.154317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102878 15.128597 18.154317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102878 15.128597 18.154317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102878 15.128597 18.154317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102878 15.128597 18.154317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.102878 15.128597 18.154317) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7554     -537.9331       0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 238 atoms

327.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.17e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100997767842216 y final 0 15.126247209802777 z final 0 18.151496651763328      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100998 15.128597 18.154317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100998 15.126247 18.154317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100998 15.126247 18.151497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100998 15.126247 18.151497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100998 15.126247 18.151497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100998 15.126247 18.151497) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7584     -218.6129       0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 238 atoms

252.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.92e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100997767842216 y final 0 15.126247209802779 z final 0 18.151496651763328      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100998 15.126247 18.151497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100998 15.126247 18.151497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100998 15.126247 18.151497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100998 15.126247 18.151497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100998 15.126247 18.151497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100998 15.126247 18.151497) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7584     -218.6129       0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 238 atoms

400.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.099114974785225 y final 0 15.123893718481545 z final 0 18.148672462177842      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.099115 15.126247 18.151497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099115 15.123894 18.151497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099115 15.123894 18.148672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099115 15.123894 18.148672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099115 15.123894 18.148672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099115 15.123894 18.148672) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7599      358.50874      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 238 atoms

554.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.099114974785225 y final 0 15.123893718481545 z final 0 18.148672462177842      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.099115 15.123894 18.148672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099115 15.123894 18.148672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099115 15.123894 18.148672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099115 15.123894 18.148672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099115 15.123894 18.148672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099115 15.123894 18.148672) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7599      358.50874      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 238 atoms

145.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.375e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.099814182661198 y final 0 15.124767728326512 z final 0 18.1497212739918      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.099814 15.123894 18.148672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099814 15.124768 18.148672) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099814 15.124768 18.149721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099814 15.124768 18.149721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099814 15.124768 18.149721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099814 15.124768 18.149721) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7604      111.41715      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 238 atoms

313.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100673626602221 y final 0 15.12584203325279 z final 0 18.151010439903335      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100674 15.124768 18.149721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100674 15.125842 18.149721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100674 15.125842 18.15101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100674 15.125842 18.15101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100674 15.125842 18.15101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100674 15.125842 18.15101) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7606     -192.46564      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 238 atoms

248.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100673626602221 y final 0 15.125842033252791 z final 0 18.151010439903335      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100674 15.125842 18.15101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100674 15.125842 18.15101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100674 15.125842 18.15101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100674 15.125842 18.15101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100674 15.125842 18.15101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100674 15.125842 18.15101) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7606     -192.46564      0              0            
Loop time of 1.417e-06 on 1 procs for 0 steps with 238 atoms

423.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.417e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.1003520417699 y final 0 15.125440052212392 z final 0 18.150528062654853      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100352 15.125842 18.15101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100352 15.12544 18.15101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100352 15.12544 18.150528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100352 15.12544 18.150528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100352 15.12544 18.150528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100352 15.12544 18.150528) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7607     -78.584384      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 238 atoms

248.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.1003520417699 y final 0 15.125440052212394 z final 0 18.150528062654853      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100352 15.12544 18.150528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100352 15.12544 18.150528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100352 15.12544 18.150528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100352 15.12544 18.150528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100352 15.12544 18.150528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100352 15.12544 18.150528) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.7607     -78.584384      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 238 atoms

287.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10022823130924 y final 0 15.12528528913657 z final 0 18.15034234696386      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100228 15.12544 18.150528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100228 15.125285 18.150528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100228 15.125285 18.150342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100228 15.125285 18.150342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100228 15.125285 18.150342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100228 15.125285 18.150342) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
WARNING: Too many warnings: 101 vs 100. All future warnings will be suppressed (src/src/thermo.cpp:465)
-1259.7607     -37.867199      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.33e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.111038117270029 y final 0 15.138797646587536 z final 0 18.166557175905044      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.111038 15.125285 18.150342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.150342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4103     -2740.9415      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 238 atoms

331.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 9.915679358308633 y final 0 12.394599197885793 z final 0 14.873519037462952      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.9156794 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-710.49194      1970457.8      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 238 atoms

154.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.292e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2778 ave        2778 max        2778 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          30445 ave       30445 max       30445 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 30445
Ave neighs/atom = 127.92017
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100478319592181 y final 0 15.125597899490225 z final 0 18.150717479388273      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100478 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100478 15.125598 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100478 15.125598 18.150717) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100478 15.125598 18.150717) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100478 15.125598 18.150717) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100478 15.125598 18.150717) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4144      1185.3632      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.92e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100478319592181 y final 0 15.125597899490227 z final 0 18.150717479388273      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100478 15.125598 18.150717) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100478 15.125598 18.150717) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100478 15.125598 18.150717) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100478 15.125598 18.150717) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100478 15.125598 18.150717) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100478 15.125598 18.150717) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4144      1185.3632      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 238 atoms

355.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10318088273547 y final 0 15.128976103419344 z final 0 18.154771324103212      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103181 15.125598 18.150717) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103181 15.128976 18.150717) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103181 15.128976 18.154771) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103181 15.128976 18.154771) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103181 15.128976 18.154771) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103181 15.128976 18.154771) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4506      23.563374      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10683263264482 y final 0 15.133540790806036 z final 0 18.160248948967244      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106833 15.128976 18.154771) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106833 15.133541 18.154771) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106833 15.133541 18.160249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106833 15.133541 18.160249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106833 15.133541 18.160249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106833 15.133541 18.160249) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4683     -1618.3795      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 238 atoms

208.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10683263264482 y final 0 15.133540790806038 z final 0 18.160248948967244      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.106833 15.133541 18.160249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106833 15.133541 18.160249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106833 15.133541 18.160249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106833 15.133541 18.160249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106833 15.133541 18.160249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.106833 15.133541 18.160249) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4683     -1618.3795      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 238 atoms

82.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103011911107465 y final 0 15.128764888884346 z final 0 18.154517866661216      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103012 15.133541 18.160249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103012 15.128765 18.160249) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103012 15.128765 18.154518) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103012 15.128765 18.154518) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103012 15.128765 18.154518) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103012 15.128765 18.154518) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4772     -577.24221      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103011911107465 y final 0 15.128764888884346 z final 0 18.154517866661216      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103012 15.128765 18.154518) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103012 15.128765 18.154518) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103012 15.128765 18.154518) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103012 15.128765 18.154518) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103012 15.128765 18.154518) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103012 15.128765 18.154518) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4772     -577.24221      0              0            
Loop time of 1.5e-06 on 1 procs for 0 steps with 238 atoms

400.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.5e-06    |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.099851760316682 y final 0 15.12481470039587 z final 0 18.149777640475044      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.099852 15.128765 18.154518) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099852 15.124815 18.154518) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099852 15.124815 18.149778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099852 15.124815 18.149778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099852 15.124815 18.149778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099852 15.124815 18.149778) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4807      304.7037       0              0            
Loop time of 1.625e-06 on 1 procs for 0 steps with 238 atoms

184.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.625e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.099851760316682 y final 0 15.124814700395872 z final 0 18.149777640475044      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.099852 15.124815 18.149778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099852 15.124815 18.149778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099852 15.124815 18.149778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099852 15.124815 18.149778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099852 15.124815 18.149778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099852 15.124815 18.149778) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4807      304.7037       0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 238 atoms

80.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.25e-06   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.099729572410753 y final 0 15.124661965513463 z final 0 18.149594358616152      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.09973 15.124815 18.149778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.09973 15.124662 18.149778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.09973 15.124662 18.149594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.09973 15.124662 18.149594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.09973 15.124662 18.149594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.09973 15.124662 18.149594) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4816      306.52951      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 238 atoms

160.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.25e-06   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.099618080930032 y final 0 15.124522601162562 z final 0 18.14942712139507      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.099618 15.124662 18.149594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099618 15.124523 18.149594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099618 15.124523 18.149427) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099618 15.124523 18.149427) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099618 15.124523 18.149427) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099618 15.124523 18.149427) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4818      347.12453      0              0            
Loop time of 1.542e-06 on 1 procs for 0 steps with 238 atoms

64.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.542e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.099618080930032 y final 0 15.124522601162564 z final 0 18.14942712139507      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.099618 15.124523 18.149427) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099618 15.124523 18.149427) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099618 15.124523 18.149427) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099618 15.124523 18.149427) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099618 15.124523 18.149427) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099618 15.124523 18.149427) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4818      347.12453      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 238 atoms

232.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.292e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100443003374341 y final 0 15.12555375421795 z final 0 18.150664505061535      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100443 15.124523 18.149427) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100443 15.125554 18.149427) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100443 15.125554 18.150665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100443 15.125554 18.150665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100443 15.125554 18.150665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100443 15.125554 18.150665) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4822      48.710236      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 238 atoms

200.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100443003374341 y final 0 15.125553754217952 z final 0 18.150664505061535      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100443 15.125554 18.150665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100443 15.125554 18.150665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100443 15.125554 18.150665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100443 15.125554 18.150665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100443 15.125554 18.150665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100443 15.125554 18.150665) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4822      48.710236      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 238 atoms

387.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.292e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100506934012197 y final 0 15.12563366751527 z final 0 18.150760401018317      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100507 15.125554 18.150665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100507 15.125634 18.150665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100507 15.125634 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100507 15.125634 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100507 15.125634 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100507 15.125634 18.15076) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4823      27.852249      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100506934012197 y final 0 15.125633667515272 z final 0 18.150760401018317      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100507 15.125634 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100507 15.125634 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100507 15.125634 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100507 15.125634 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100507 15.125634 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100507 15.125634 18.15076) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4823      27.852249      0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 238 atoms

496.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.209e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100442077690662 y final 0 15.125552597113357 z final 0 18.150663116536016      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100442 15.125634 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100442 15.125553 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100442 15.125553 18.150663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100442 15.125553 18.150663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100442 15.125553 18.150663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100442 15.125553 18.150663) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4823      49.9613        0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 238 atoms

313.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100442077690662 y final 0 15.125552597113359 z final 0 18.150663116536016      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100442 15.125553 18.150663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100442 15.125553 18.150663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100442 15.125553 18.150663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100442 15.125553 18.150663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100442 15.125553 18.150663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100442 15.125553 18.150663) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4823      49.9613        0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 238 atoms

109.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.17e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100388612248873 y final 0 15.125485765311126 z final 0 18.150582918373335      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100389 15.125553 18.150663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100389 15.125486 18.150663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100389 15.125486 18.150583) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100389 15.125486 18.150583) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100389 15.125486 18.150583) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100389 15.125486 18.150583) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4823      67.841829      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 238 atoms

228.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.111038117270029 y final 0 15.138797646587536 z final 0 18.166557175905044      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.111038 15.125486 18.150583) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.150583) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4334     -2580.8649      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 238 atoms

342.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 9.915679358308633 y final 0 12.394599197885793 z final 0 14.873519037462952      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.9156794 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-710.43425      1970963.3      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2778 ave        2778 max        2778 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          30446 ave       30446 max       30446 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 30446
Ave neighs/atom = 127.92437
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.101018587597418 y final 0 15.12627323449677 z final 0 18.151527881396127      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101019 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101019 15.126273 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101019 15.126273 18.151528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101019 15.126273 18.151528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101019 15.126273 18.151528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101019 15.126273 18.151528) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4371      1140.1537      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.59e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.101018587597418 y final 0 15.12627323449677 z final 0 18.151527881396127      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101019 15.126273 18.151528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101019 15.126273 18.151528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101019 15.126273 18.151528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101019 15.126273 18.151528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101019 15.126273 18.151528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101019 15.126273 18.151528) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4371      1140.1537      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 238 atoms

240.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.33e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103552659991177 y final 0 15.129440824988963 z final 0 18.155328989986756      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103553 15.126273 18.151528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103553 15.129441 18.151528) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103553 15.129441 18.155329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103553 15.129441 18.155329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103553 15.129441 18.155329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103553 15.129441 18.155329) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4761      55.022778      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.107448003570195 y final 0 15.134310004462726 z final 0 18.161172005355272      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.107448 15.129441 18.155329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107448 15.13431 18.155329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107448 15.13431 18.161172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107448 15.13431 18.161172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107448 15.13431 18.161172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107448 15.13431 18.161172) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.499      -1731.3408      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 238 atoms

88.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.107448003570195 y final 0 15.134310004462728 z final 0 18.161172005355272      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.107448 15.13431 18.161172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107448 15.13431 18.161172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107448 15.13431 18.161172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107448 15.13431 18.161172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107448 15.13431 18.161172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107448 15.13431 18.161172) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.499      -1731.3408      0              0            
Loop time of 1.792e-06 on 1 procs for 0 steps with 238 atoms

223.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.792e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103241075801208 y final 0 15.129051344751492 z final 0 18.154861613701783      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103241 15.13431 18.161172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103241 15.129051 18.161172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103241 15.129051 18.154862) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103241 15.129051 18.154862) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103241 15.129051 18.154862) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103241 15.129051 18.154862) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5095     -309.72676      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 238 atoms

114.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16054 ave       16054 max       16054 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16054
Ave neighs/atom = 67.453782
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103241075801208 y final 0 15.129051344751494 z final 0 18.154861613701783      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103241 15.129051 18.154862) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103241 15.129051 18.154862) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103241 15.129051 18.154862) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103241 15.129051 18.154862) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103241 15.129051 18.154862) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103241 15.129051 18.154862) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5095     -309.72676      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 238 atoms

200.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16054 ave       16054 max       16054 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16054
Ave neighs/atom = 67.453782
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.101366465967853 y final 0 15.126708082459798 z final 0 18.152049698951746      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101366 15.129051 18.154862) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101366 15.126708 18.154862) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101366 15.126708 18.15205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101366 15.126708 18.15205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101366 15.126708 18.15205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101366 15.126708 18.15205) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5123      342.5501       0              0            
Loop time of 7.09e-07 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.09e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16060 ave       16060 max       16060 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16060
Ave neighs/atom = 67.478992
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.101366465967853 y final 0 15.1267080824598 z final 0 18.152049698951746      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101366 15.126708 18.15205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101366 15.126708 18.15205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101366 15.126708 18.15205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101366 15.126708 18.15205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101366 15.126708 18.15205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101366 15.126708 18.15205) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5123      342.5501       0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 238 atoms

387.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.292e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16060 ave       16060 max       16060 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16060
Ave neighs/atom = 67.478992
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10146630397035 y final 0 15.126832879962919 z final 0 18.152199455955486      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101466 15.126708 18.15205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101466 15.126833 18.15205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101466 15.126833 18.152199) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101466 15.126833 18.152199) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101466 15.126833 18.152199) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101466 15.126833 18.152199) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.513       273.0593       0              0            
Loop time of 1.333e-06 on 1 procs for 0 steps with 238 atoms

150.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.333e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16064 ave       16064 max       16064 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16064
Ave neighs/atom = 67.495798
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10146630397035 y final 0 15.12683287996292 z final 0 18.152199455955486      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101466 15.126833 18.152199) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101466 15.126833 18.152199) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101466 15.126833 18.152199) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101466 15.126833 18.152199) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101466 15.126833 18.152199) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101466 15.126833 18.152199) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.513       273.0593       0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 238 atoms

500.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16064 ave       16064 max       16064 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16064
Ave neighs/atom = 67.495798
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.101663778831572 y final 0 15.12707972353945 z final 0 18.15249566824732      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101664 15.126833 18.152199) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101664 15.12708 18.152199) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101664 15.12708 18.152496) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101664 15.12708 18.152496) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101664 15.12708 18.152496) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101664 15.12708 18.152496) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5134      117.45374      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 238 atoms

96.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16066 ave       16066 max       16066 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16066
Ave neighs/atom = 67.504202
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.101663778831572 y final 0 15.127079723539453 z final 0 18.15249566824732      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101664 15.12708 18.152496) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101664 15.12708 18.152496) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101664 15.12708 18.152496) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101664 15.12708 18.152496) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101664 15.12708 18.152496) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101664 15.12708 18.152496) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5134      117.45374      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 238 atoms

383.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16066 ave       16066 max       16066 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16066
Ave neighs/atom = 67.504202
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.101515851726603 y final 0 15.126894814658243 z final 0 18.152273777589865      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101516 15.12708 18.152496) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101516 15.126895 18.152496) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101516 15.126895 18.152274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101516 15.126895 18.152274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101516 15.126895 18.152274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101516 15.126895 18.152274) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5136      24.335263      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 238 atoms

252.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.91e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10133337187625 y final 0 15.126666714845305 z final 0 18.15200005781433      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101333 15.126895 18.152274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101333 15.126667 18.152274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101333 15.126667 18.152) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101333 15.126667 18.152) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101333 15.126667 18.152) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101333 15.126667 18.152) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5137     -54.249772      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 238 atoms

208.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.10133337187625 y final 0 15.126666714845305 z final 0 18.15200005781433      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101333 15.126667 18.152) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101333 15.126667 18.152) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101333 15.126667 18.152) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101333 15.126667 18.152) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101333 15.126667 18.152) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101333 15.126667 18.152) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5137     -54.249772      0              0            
Loop time of 1.542e-06 on 1 procs for 0 steps with 238 atoms

259.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.542e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100949485040186 y final 0 15.126186856300226 z final 0 18.151424227560234      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100949 15.126667 18.152) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100949 15.126187 18.152) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100949 15.126187 18.151424) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100949 15.126187 18.151424) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100949 15.126187 18.151424) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100949 15.126187 18.151424) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.514      -4.6440748      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 238 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.099414059471293 y final 0 15.12426757433911 z final 0 18.149121089206893      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.099414 15.126187 18.151424) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099414 15.124268 18.151424) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099414 15.124268 18.149121) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099414 15.124268 18.149121) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099414 15.124268 18.149121) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.099414 15.124268 18.149121) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5137      579.51304      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 238 atoms

85.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.166e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100361933516972 y final 0 15.12545241689621 z final 0 18.15054290027541      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100362 15.124268 18.149121) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100362 15.125452 18.149121) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100362 15.125452 18.150543) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100362 15.125452 18.150543) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100362 15.125452 18.150543) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100362 15.125452 18.150543) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5142      227.97152      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 238 atoms

92.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100361933516972 y final 0 15.125452416896211 z final 0 18.15054290027541      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100362 15.125452 18.150543) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100362 15.125452 18.150543) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100362 15.125452 18.150543) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100362 15.125452 18.150543) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100362 15.125452 18.150543) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100362 15.125452 18.150543) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5142      227.97152      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 238 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100479371243377 y final 0 15.125599214054217 z final 0 18.150719056865018      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100479 15.125452 18.150543) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150543) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5143      157.59111      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 238 atoms

240.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.33e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100479371243377 y final 0 15.125599214054219 z final 0 18.150719056865018      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5143      157.59111      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 238 atoms

165.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100280499207862 y final 0 15.125350624009824 z final 0 18.15042074881174      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.10028 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10028 15.125351 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10028 15.125351 18.150421) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10028 15.125351 18.150421) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10028 15.125351 18.150421) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10028 15.125351 18.150421) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5143      256.39578      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 238 atoms

141.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.08e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100280499207862 y final 0 15.125350624009824 z final 0 18.15042074881174      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.10028 15.125351 18.150421) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10028 15.125351 18.150421) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10028 15.125351 18.150421) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10028 15.125351 18.150421) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10028 15.125351 18.150421) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10028 15.125351 18.150421) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5143      256.39578      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 238 atoms

342.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.167e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100449389441074 y final 0 15.12556173680134 z final 0 18.150674084161558      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100449 15.125351 18.150421) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100449 15.125562 18.150421) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100449 15.125562 18.150674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100449 15.125562 18.150674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100449 15.125562 18.150674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100449 15.125562 18.150674) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5144      189.13058      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 238 atoms

88.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100449389441074 y final 0 15.125561736801341 z final 0 18.150674084161558      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100449 15.125562 18.150674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100449 15.125562 18.150674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100449 15.125562 18.150674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100449 15.125562 18.150674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100449 15.125562 18.150674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100449 15.125562 18.150674) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5144      189.13058      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 238 atoms

387.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.292e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100479171864098 y final 0 15.125598964830122 z final 0 18.150718757796092      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100479 15.125562 18.150674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150674) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100479 15.125599 18.150719) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5144      177.45559      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 238 atoms

191.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16065 ave       16065 max       16065 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16065
Ave neighs/atom = 67.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100506345826993 y final 0 15.125632932283738 z final 0 18.150759518740433      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100506 15.125599 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100506 15.125633 18.150719) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100506 15.125633 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100506 15.125633 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100506 15.125633 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100506 15.125633 18.15076) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5144      166.63348      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.041e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16067 ave       16067 max       16067 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16067
Ave neighs/atom = 67.508403
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100506345826993 y final 0 15.125632932283738 z final 0 18.150759518740433      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100506 15.125633 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100506 15.125633 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100506 15.125633 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100506 15.125633 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100506 15.125633 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100506 15.125633 18.15076) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5144      166.63348      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 238 atoms

461.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16067 ave       16067 max       16067 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16067
Ave neighs/atom = 67.508403
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100493834805608 y final 0 15.125617293507009 z final 0 18.150740752208353      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100494 15.125633 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100494 15.125617 18.15076) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100494 15.125617 18.150741) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100494 15.125617 18.150741) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100494 15.125617 18.150741) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100494 15.125617 18.150741) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5144      172.71423      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.16e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16067 ave       16067 max       16067 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16067
Ave neighs/atom = 67.508403
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100477174704459 y final 0 15.125596468380571 z final 0 18.15071576205663      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100477 15.125617 18.150741) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100477 15.125596 18.150741) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100477 15.125596 18.150716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100477 15.125596 18.150716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100477 15.125596 18.150716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100477 15.125596 18.150716) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5144      180.64885      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 238 atoms

191.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16067 ave       16067 max       16067 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16067
Ave neighs/atom = 67.508403
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.111038117270029 y final 0 15.138797646587536 z final 0 18.166557175905044      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.111038 15.125596 18.150716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.150716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.111038 15.138798 18.166557) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4308     -2685.58        0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 238 atoms

85.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.167e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 9.915679358308633 y final 0 12.394599197885793 z final 0 14.873519037462952      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.9156794 15.138798 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 18.166557) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.9156794 12.394599 14.873519) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-710.40179      1970988.2      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 238 atoms

248.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2778 ave        2778 max        2778 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          30445 ave       30445 max       30445 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 30445
Ave neighs/atom = 127.92017
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100712464753201 y final 0 15.125890580941501 z final 0 18.1510686971298      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100712 12.394599 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100712 15.125891 14.873519) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100712 15.125891 18.151069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100712 15.125891 18.151069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100712 15.125891 18.151069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100712 15.125891 18.151069) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4349      1131.815       0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 238 atoms

184.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100712464753201 y final 0 15.125890580941503 z final 0 18.1510686971298      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100712 15.125891 18.151069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100712 15.125891 18.151069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100712 15.125891 18.151069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100712 15.125891 18.151069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100712 15.125891 18.151069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100712 15.125891 18.151069) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4349      1131.815       0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 238 atoms

232.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.292e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103291962827285 y final 0 15.129114953534112 z final 0 18.15493794424092      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103292 15.125891 18.151069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103292 15.129115 18.151069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103292 15.129115 18.154938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103292 15.129115 18.154938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103292 15.129115 18.154938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103292 15.129115 18.154938) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4698      33.755329      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 238 atoms

133.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.107009350887404 y final 0 15.133761688609267 z final 0 18.160514026331093      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.107009 15.129115 18.154938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107009 15.133762 18.154938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107009 15.133762 18.160514) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107009 15.133762 18.160514) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107009 15.133762 18.160514) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107009 15.133762 18.160514) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4886     -1634.4417      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 238 atoms

114.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.107009350887404 y final 0 15.133761688609269 z final 0 18.160514026331093      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.107009 15.133762 18.160514) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107009 15.133762 18.160514) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107009 15.133762 18.160514) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107009 15.133762 18.160514) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107009 15.133762 18.160514) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.107009 15.133762 18.160514) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4886     -1634.4417      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 238 atoms

355.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103189428092227 y final 0 15.128986785115304 z final 0 18.154784142138325      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103189 15.133762 18.160514) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103189 15.128987 18.160514) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103189 15.128987 18.154784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103189 15.128987 18.154784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103189 15.128987 18.154784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103189 15.128987 18.154784) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4966     -473.96741      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 238 atoms

133.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.103189428092227 y final 0 15.128986785115305 z final 0 18.154784142138325      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.103189 15.128987 18.154784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103189 15.128987 18.154784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103189 15.128987 18.154784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103189 15.128987 18.154784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103189 15.128987 18.154784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.103189 15.128987 18.154784) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4966     -473.96741      0              0            
Loop time of 1.542e-06 on 1 procs for 0 steps with 238 atoms

129.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.542e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16049 ave       16049 max       16049 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16049
Ave neighs/atom = 67.432773
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.101074702625764 y final 0 15.126343378282233 z final 0 18.15161205393863      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101075 15.128987 18.154784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101075 15.126343 18.154784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101075 15.126343 18.151612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101075 15.126343 18.151612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101075 15.126343 18.151612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101075 15.126343 18.151612) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4988      94.936211      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 238 atoms

287.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16053 ave       16053 max       16053 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16053
Ave neighs/atom = 67.44958
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.101074702625764 y final 0 15.126343378282233 z final 0 18.15161205393863      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.101075 15.126343 18.151612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101075 15.126343 18.151612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101075 15.126343 18.151612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101075 15.126343 18.151612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101075 15.126343 18.151612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.101075 15.126343 18.151612) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4988      94.936211      0              0            
Loop time of 1.334e-06 on 1 procs for 0 steps with 238 atoms

224.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.334e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16053 ave       16053 max       16053 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16053
Ave neighs/atom = 67.44958
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100626410450158 y final 0 15.125783013062728 z final 0 18.15093961567522      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100626 15.126343 18.151612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100626 15.125783 18.151612) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100626 15.125783 18.15094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100626 15.125783 18.15094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100626 15.125783 18.15094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100626 15.125783 18.15094) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4994      175.36208      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 238 atoms

145.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.375e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100626410450158 y final 0 15.125783013062728 z final 0 18.15093961567522      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100626 15.125783 18.15094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100626 15.125783 18.15094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100626 15.125783 18.15094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100626 15.125783 18.15094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100626 15.125783 18.15094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100626 15.125783 18.15094) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.4994      175.36208      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 238 atoms

383.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16057 ave       16057 max       16057 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16057
Ave neighs/atom = 67.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100630417127038 y final 0 15.12578802140883 z final 0 18.150945625690536      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.10063 15.125783 18.15094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10063 15.125788 18.15094) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10063 15.125788 18.150946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10063 15.125788 18.150946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10063 15.125788 18.150946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10063 15.125788 18.150946) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5        -9.3816334      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.16e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16061 ave       16061 max       16061 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16061
Ave neighs/atom = 67.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100630417127038 y final 0 15.125788021408832 z final 0 18.150945625690536      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.10063 15.125788 18.150946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10063 15.125788 18.150946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10063 15.125788 18.150946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10063 15.125788 18.150946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10063 15.125788 18.150946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.10063 15.125788 18.150946) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5        -9.3816334      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 238 atoms

521.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.59e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16061 ave       16061 max       16061 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16061
Ave neighs/atom = 67.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100110907928304 y final 0 15.12513863491042 z final 0 18.150166361892428      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100111 15.125788 18.150946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100111 15.125139 18.150946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100111 15.125139 18.150166) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100111 15.125139 18.150166) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100111 15.125139 18.150166) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100111 15.125139 18.150166) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5         132.09143      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16061 ave       16061 max       16061 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16061
Ave neighs/atom = 67.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.100382111145212 y final 0 15.125477638931551 z final 0 18.15057316671779      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.100382 15.125139 18.150166) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100382 15.125478 18.150166) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100382 15.125478 18.150573) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100382 15.125478 18.150573) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100382 15.125478 18.150573) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.100382 15.125478 18.150573) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1259.5002      24.39007       0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16061 ave       16061 max       16061 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16061
Ave neighs/atom = 67.483193
Neighbor list builds = 0
Dangerous builds = 0
change_box all boundary p p f
Changing box ...
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 8.563797179930395 y final 0 6.0555190586350145 z final 0 39.26854365484339      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5637972 15.125478 18.150573) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5637972 6.0555191 18.150573) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5637972 6.0555191 39.268544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5637972 6.0555191 39.268544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5637972 6.0555191 39.268544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5637972 6.0555191 39.268544) with tilt (0 0 0)
echo none
158 atoms in group delatoms
Deleted 158 atoms, new total = 80
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-413.96082     -12232.669      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 80 atoms

109.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.17e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 7.01144411433151 y final 0 4.957839679154317 z final 0 32.150364318805536      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.0114441 6.0555191 39.268544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.0114441 4.9578397 39.268544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.0114441 4.9578397 32.150364) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.0114441 4.9578397 32.150364) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.0114441 4.9578397 32.150364) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.0114441 4.9578397 32.150364) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-240.53834      1068724.3      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 80 atoms

109.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.17e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1025 ave        1025 max        1025 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           8624 ave        8624 max        8624 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 8624
Ave neighs/atom = 107.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.50639092495039 y final 0 6.014926706456129 z final 0 39.005312288852345      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5063909 4.9578397 32.150364) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5063909 6.0149267 32.150364) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5063909 6.0149267 39.005312) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5063909 6.0149267 39.005312) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5063909 6.0149267 39.005312) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5063909 6.0149267 39.005312) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.04818      5651.4562      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 80 atoms

109.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.17e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.50639092495039 y final 0 6.014926706456129 z final 0 39.005312288852345      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5063909 6.0149267 39.005312) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5063909 6.0149267 39.005312) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5063909 6.0149267 39.005312) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5063909 6.0149267 39.005312) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5063909 6.0149267 39.005312) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5063909 6.0149267 39.005312) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.04818      5651.4562      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

400.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.531637647994764 y final 0 6.032778835523544 z final 0 39.121078931286746      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5316376 6.0149267 39.005312) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5316376 6.0327788 39.005312) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5316376 6.0327788 39.121079) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5316376 6.0327788 39.121079) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5316376 6.0327788 39.121079) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5316376 6.0327788 39.121079) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.18497     -166.02223      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.041e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.531637647994764 y final 0 6.032778835523544 z final 0 39.121078931286746      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5316376 6.0327788 39.121079) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5316376 6.0327788 39.121079) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5316376 6.0327788 39.121079) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5316376 6.0327788 39.121079) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5316376 6.0327788 39.121079) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5316376 6.0327788 39.121079) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.18497     -166.02223      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 80 atoms

208.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.59e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.534359578853886 y final 0 6.03470353129195 z final 0 39.13356010739602      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5343596 6.0327788 39.121079) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5343596 6.0347035 39.121079) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5343596 6.0347035 39.13356) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5343596 6.0347035 39.13356) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5343596 6.0347035 39.13356) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5343596 6.0347035 39.13356) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.19506     -582.52938      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 80 atoms

92.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.534359578853886 y final 0 6.03470353129195 z final 0 39.13356010739602      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5343596 6.0347035 39.13356) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5343596 6.0347035 39.13356) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5343596 6.0347035 39.13356) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5343596 6.0347035 39.13356) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5343596 6.0347035 39.13356) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5343596 6.0347035 39.13356) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.19506     -582.52938      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.534687029777087 y final 0 6.034935074060251 z final 0 39.135061604991336      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.534687 6.0347035 39.13356) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.534687 6.0349351 39.13356) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.534687 6.0349351 39.135062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.534687 6.0349351 39.135062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.534687 6.0349351 39.135062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.534687 6.0349351 39.135062) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.19953     -466.0158       0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.534687029777087 y final 0 6.034935074060251 z final 0 39.135061604991336      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.534687 6.0349351 39.135062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.534687 6.0349351 39.135062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.534687 6.0349351 39.135062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.534687 6.0349351 39.135062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.534687 6.0349351 39.135062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.534687 6.0349351 39.135062) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.19953     -466.0158       0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.534210276829135 y final 0 6.034597958817804 z final 0 39.13287549600768      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5342103 6.0349351 39.135062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5342103 6.034598 39.135062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5342103 6.034598 39.132875) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5342103 6.034598 39.132875) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5342103 6.034598 39.132875) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5342103 6.034598 39.132875) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.20135     -345.05976      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.534210276829135 y final 0 6.034597958817804 z final 0 39.13287549600768      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5342103 6.034598 39.132875) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5342103 6.034598 39.132875) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5342103 6.034598 39.132875) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5342103 6.034598 39.132875) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5342103 6.034598 39.132875) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5342103 6.034598 39.132875) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.20135     -345.05976      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 80 atoms

228.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.533246339799561 y final 0 6.033916352407553 z final 0 39.1284554469884      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5332463 6.034598 39.132875) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5332463 6.0339164 39.132875) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5332463 6.0339164 39.128455) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5332463 6.0339164 39.128455) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5332463 6.0339164 39.128455) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5332463 6.0339164 39.128455) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.20273     -241.14841      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 80 atoms

141.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.08e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.532369917819166 y final 0 6.033296628482035 z final 0 39.12443669057998      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5323699 6.0339164 39.128455) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5323699 6.0332966 39.128455) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5323699 6.0332966 39.124437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5323699 6.0332966 39.124437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5323699 6.0332966 39.124437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5323699 6.0332966 39.124437) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.20314     -146.69844      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 80 atoms

252.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.92e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.532369917819166 y final 0 6.033296628482035 z final 0 39.12443669057998      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5323699 6.0332966 39.124437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5323699 6.0332966 39.124437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5323699 6.0332966 39.124437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5323699 6.0332966 39.124437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5323699 6.0332966 39.124437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5323699 6.0332966 39.124437) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.20314     -146.69844      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 80 atoms

461.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.531470674687192 y final 0 6.032660767565481 z final 0 39.12031328977513      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5314707 6.0332966 39.124437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5314707 6.0326608 39.124437) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5314707 6.0326608 39.120313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5314707 6.0326608 39.120313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5314707 6.0326608 39.120313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5314707 6.0326608 39.120313) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.20453     -115.77252      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.25e-06   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.527874649789323 y final 0 6.030117993974883 z final 0 39.10382403183044      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5278746 6.0326608 39.120313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5278746 6.030118 39.120313) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5278746 6.030118 39.103824) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5278746 6.030118 39.103824) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5278746 6.030118 39.103824) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5278746 6.030118 39.103824) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.20914      9.7133472      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 80 atoms

192.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.041e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.518748192289976 y final 0 6.023664613988884 z final 0 39.06197545844676      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5187482 6.030118 39.103824) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5187482 6.0236646 39.103824) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5187482 6.0236646 39.061975) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5187482 6.0236646 39.061975) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5187482 6.0236646 39.061975) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5187482 6.0236646 39.061975) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.21402      341.4594       0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

200.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.518748192289976 y final 0 6.023664613988884 z final 0 39.06197545844676      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5187482 6.0236646 39.061975) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5187482 6.0236646 39.061975) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5187482 6.0236646 39.061975) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5187482 6.0236646 39.061975) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5187482 6.0236646 39.061975) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5187482 6.0236646 39.061975) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.21402      341.4594       0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 80 atoms

252.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.91e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.514232577720287 y final 0 6.02047159230543 z final 0 39.04126950241665      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5142326 6.0236646 39.061975) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5142326 6.0204716 39.061975) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5142326 6.0204716 39.04127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5142326 6.0204716 39.04127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5142326 6.0204716 39.04127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5142326 6.0204716 39.04127) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.21904      740.65852      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 80 atoms

114.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.50892269979817 y final 0 6.0167169416194275 z final 0 39.01692152118755      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5089227 6.0204716 39.04127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5089227 6.0167169 39.04127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5089227 6.0167169 39.016922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5089227 6.0167169 39.016922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5089227 6.0167169 39.016922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5089227 6.0167169 39.016922) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.2211       1215.5629      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.92e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.509029284738453 y final 0 6.016792308553473 z final 0 39.01741025711771      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5090293 6.0167169 39.016922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5090293 6.0167923 39.016922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5090293 6.0167923 39.01741) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5090293 6.0167923 39.01741) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5090293 6.0167923 39.01741) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5090293 6.0167923 39.01741) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-414.22111      1189.0266      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.59e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4720 ave        4720 max        4720 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4720
Ave neighs/atom = 59
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.0555190586350145 y final 0 6.0555190586350145 z final 0 48.763715528516315      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0555191 6.0167923 39.01741) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0555191 6.0555191 39.01741) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0555191 6.0555191 48.763716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0555191 6.0555191 48.763716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0555191 6.0555191 48.763716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0555191 6.0555191 48.763716) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-415.52635     -17153.004      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 80 atoms

239.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.34e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4920 ave        4920 max        4920 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4920
Ave neighs/atom = 61.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 4.957839679154317 y final 0 4.957839679154317 z final 0 39.924353537542636      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (4.9578397 6.0555191 48.763716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (4.9578397 4.9578397 48.763716) with tilt (0 0 0)
  triclinic box = (0 0 0) to (4.9578397 4.9578397 39.924354) with tilt (0 0 0)
  triclinic box = (0 0 0) to (4.9578397 4.9578397 39.924354) with tilt (0 0 0)
  triclinic box = (0 0 0) to (4.9578397 4.9578397 39.924354) with tilt (0 0 0)
  triclinic box = (0 0 0) to (4.9578397 4.9578397 39.924354) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-242.08913      1185868.1      0              0            
Loop time of 6.66e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.66e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9104 ave        9104 max        9104 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9104
Ave neighs/atom = 113.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.011072201715999 y final 0 6.011072201715999 z final 0 48.40579511476666      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0110722 4.9578397 39.924354) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0110722 6.0110722 39.924354) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0110722 6.0110722 48.405795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0110722 6.0110722 48.405795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0110722 6.0110722 48.405795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0110722 6.0110722 48.405795) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-415.63813      4707.5803      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.92e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4920 ave        4920 max        4920 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4920
Ave neighs/atom = 61.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.011072201715999 y final 0 6.011072201715999 z final 0 48.40579511476666      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0110722 6.0110722 48.405795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0110722 6.0110722 48.405795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0110722 6.0110722 48.405795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0110722 6.0110722 48.405795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0110722 6.0110722 48.405795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0110722 6.0110722 48.405795) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-415.63813      4707.5803      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 80 atoms

88.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4920 ave        4920 max        4920 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4920
Ave neighs/atom = 61.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.032548256918373 y final 0 6.032548256918374 z final 0 48.578736895053844      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0325483 6.0110722 48.405795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0325483 6.0325483 48.405795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0325483 6.0325483 48.578737) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0325483 6.0325483 48.578737) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0325483 6.0325483 48.578737) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0325483 6.0325483 48.578737) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.04395     -909.65111      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.032548256918373 y final 0 6.032548256918374 z final 0 48.578736895053844      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0325483 6.0325483 48.578737) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0325483 6.0325483 48.578737) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0325483 6.0325483 48.578737) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0325483 6.0325483 48.578737) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0325483 6.0325483 48.578737) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0325483 6.0325483 48.578737) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.04395     -909.65111      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 80 atoms

114.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.036975180844266 y final 0 6.0369751808442675 z final 0 48.61438590497328      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0369752 6.0325483 48.578737) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0369752 6.0369752 48.578737) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0369752 6.0369752 48.614386) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0369752 6.0369752 48.614386) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0369752 6.0369752 48.614386) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0369752 6.0369752 48.614386) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.07668     -1377.4396      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.16e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.036975180844266 y final 0 6.0369751808442675 z final 0 48.61438590497328      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0369752 6.0369752 48.614386) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0369752 6.0369752 48.614386) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0369752 6.0369752 48.614386) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0369752 6.0369752 48.614386) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0369752 6.0369752 48.614386) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0369752 6.0369752 48.614386) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.07668     -1377.4396      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.17e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.0388905960724735 y final 0 6.038890596072475 z final 0 48.62981030747337      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0388906 6.0369752 48.614386) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0388906 6.0388906 48.614386) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0388906 6.0388906 48.62981) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0388906 6.0388906 48.62981) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0388906 6.0388906 48.62981) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0388906 6.0388906 48.62981) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.08911     -1415.1965      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 80 atoms

109.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.16e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.0388905960724735 y final 0 6.038890596072475 z final 0 48.62981030747337      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0388906 6.0388906 48.62981) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0388906 6.0388906 48.62981) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0388906 6.0388906 48.62981) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0388906 6.0388906 48.62981) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0388906 6.0388906 48.62981) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0388906 6.0388906 48.62981) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.08911     -1415.1965      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.039171264649087 y final 0 6.03917126464909 z final 0 48.63207046758439      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0391713 6.0388906 48.62981) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391713 6.0391713 48.62981) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391713 6.0391713 48.63207) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391713 6.0391713 48.63207) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391713 6.0391713 48.63207) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391713 6.0391713 48.63207) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.09358     -1186.3822      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 80 atoms

109.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.17e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.039171264649087 y final 0 6.03917126464909 z final 0 48.63207046758439      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0391713 6.0391713 48.63207) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391713 6.0391713 48.63207) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391713 6.0391713 48.63207) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391713 6.0391713 48.63207) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391713 6.0391713 48.63207) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391713 6.0391713 48.63207) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.09358     -1186.3822      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 80 atoms

276.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.084e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.039151054157297 y final 0 6.0391510541573 z final 0 48.63190771709797      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0391511 6.0391713 48.63207) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391511 6.0391511 48.63207) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391511 6.0391511 48.631908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391511 6.0391511 48.631908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391511 6.0391511 48.631908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391511 6.0391511 48.631908) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.09557     -1070.7459      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 80 atoms

85.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.167e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.039151054157297 y final 0 6.0391510541573 z final 0 48.63190771709797      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0391511 6.0391511 48.631908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391511 6.0391511 48.631908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391511 6.0391511 48.631908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391511 6.0391511 48.631908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391511 6.0391511 48.631908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391511 6.0391511 48.631908) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.09557     -1070.7459      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 80 atoms

240.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.33e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.0391438375862325 y final 0 6.039143837586237 z final 0 48.63184960369513      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0391438 6.0391511 48.631908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391438 6.0391438 48.631908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391438 6.0391438 48.63185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391438 6.0391438 48.63185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391438 6.0391438 48.63185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391438 6.0391438 48.63185) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.09682     -1059.5666      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 80 atoms

88.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.0391438375862325 y final 0 6.039143837586237 z final 0 48.63184960369513      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0391438 6.0391438 48.63185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391438 6.0391438 48.63185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391438 6.0391438 48.63185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391438 6.0391438 48.63185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391438 6.0391438 48.63185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0391438 6.0391438 48.63185) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.09682     -1059.5666      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.041e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.039142018775809 y final 0 6.039142018775814 z final 0 48.63183495722912      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.039142 6.0391438 48.63185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.039142 6.039142 48.63185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.039142 6.039142 48.631835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.039142 6.039142 48.631835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.039142 6.039142 48.631835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.039142 6.039142 48.631835) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.09737     -1104.657       0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 80 atoms

104.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.039142018775809 y final 0 6.039142018775814 z final 0 48.63183495722912      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.039142 6.039142 48.631835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.039142 6.039142 48.631835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.039142 6.039142 48.631835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.039142 6.039142 48.631835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.039142 6.039142 48.631835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.039142 6.039142 48.631835) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.09737     -1104.657       0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.039071814573939 y final 0 6.039071814573942 z final 0 48.631269618784      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.0390718 6.039142 48.631835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0390718 6.0390718 48.631835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0390718 6.0390718 48.63127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0390718 6.0390718 48.63127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0390718 6.0390718 48.63127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.0390718 6.0390718 48.63127) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-416.09763     -1134.701       0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 80 atoms

104.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.59e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4984 ave        4984 max        4984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4984
Ave neighs/atom = 62.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.563797179930395 y final 0 7.416465910677259 z final 0 36.60673890328272      xy final 4.281898589965197 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5637972 6.0390718 48.63127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5637972 7.4164659 48.63127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5637972 7.4164659 36.606739) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5637972 7.4164659 36.606739) with tilt (4.2818986 0 0)
  triclinic box = (0 0 0) to (8.5637972 7.4164659 36.606739) with tilt (4.2818986 0 0)
  triclinic box = (0 0 0) to (8.5637972 7.4164659 36.606739) with tilt (4.2818986 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-406.41954     -17378.766      0              0            
Loop time of 1.333e-06 on 1 procs for 0 steps with 80 atoms

75.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.333e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4552 ave        4552 max        4552 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4552
Ave neighs/atom = 56.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 7.01144411433151 y final 0 6.072088720225972 z final 0 29.971062910013714      xy final 3.505722057165754 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.0114441 7.4164659 36.606739) with tilt (4.2818986 0 0)
  triclinic box = (0 0 0) to (7.0114441 6.0720887 36.606739) with tilt (4.2818986 0 0)
  triclinic box = (0 0 0) to (7.0114441 6.0720887 29.971063) with tilt (4.2818986 0 0)
  triclinic box = (0 0 0) to (7.0114441 6.0720887 29.971063) with tilt (3.5057221 0 0)
  triclinic box = (0 0 0) to (7.0114441 6.0720887 29.971063) with tilt (3.5057221 0 0)
  triclinic box = (0 0 0) to (7.0114441 6.0720887 29.971063) with tilt (3.5057221 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-243.74885      881567.09      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            900 ave         900 max         900 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           8232 ave        8232 max        8232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 8232
Ave neighs/atom = 102.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.46850016987193 y final 0 7.333936279061927 z final 0 36.19938306659275      xy final 4.234250084935964 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4685002 6.0720887 29.971063) with tilt (3.5057221 0 0)
  triclinic box = (0 0 0) to (8.4685002 7.3339363 29.971063) with tilt (3.5057221 0 0)
  triclinic box = (0 0 0) to (8.4685002 7.3339363 36.199383) with tilt (3.5057221 0 0)
  triclinic box = (0 0 0) to (8.4685002 7.3339363 36.199383) with tilt (4.2342501 0 0)
  triclinic box = (0 0 0) to (8.4685002 7.3339363 36.199383) with tilt (4.2342501 0 0)
  triclinic box = (0 0 0) to (8.4685002 7.3339363 36.199383) with tilt (4.2342501 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-406.7091       6829.5025      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 80 atoms

77.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.292e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5520 ave        5520 max        5520 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5520
Ave neighs/atom = 69
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.46850016987193 y final 0 7.333936279061926 z final 0 36.19938306659275      xy final 4.234250084935964 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4685002 7.3339363 36.199383) with tilt (4.2342501 0 0)
  triclinic box = (0 0 0) to (8.4685002 7.3339363 36.199383) with tilt (4.2342501 0 0)
  triclinic box = (0 0 0) to (8.4685002 7.3339363 36.199383) with tilt (4.2342501 0 0)
  triclinic box = (0 0 0) to (8.4685002 7.3339363 36.199383) with tilt (4.2342501 0 0)
  triclinic box = (0 0 0) to (8.4685002 7.3339363 36.199383) with tilt (4.2342501 0 0)
  triclinic box = (0 0 0) to (8.4685002 7.3339363 36.199383) with tilt (4.2342501 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-406.7091       6829.5025      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 80 atoms

119.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.34e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5520 ave        5520 max        5520 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5520
Ave neighs/atom = 69
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.512753184055494 y final 0 7.372260513538924 z final 0 36.38854664693738      xy final 4.256376592027746 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5127532 7.3339363 36.199383) with tilt (4.2342501 0 0)
  triclinic box = (0 0 0) to (8.5127532 7.3722605 36.199383) with tilt (4.2342501 0 0)
  triclinic box = (0 0 0) to (8.5127532 7.3722605 36.388547) with tilt (4.2342501 0 0)
  triclinic box = (0 0 0) to (8.5127532 7.3722605 36.388547) with tilt (4.2563766 0 0)
  triclinic box = (0 0 0) to (8.5127532 7.3722605 36.388547) with tilt (4.2563766 0 0)
  triclinic box = (0 0 0) to (8.5127532 7.3722605 36.388547) with tilt (4.2563766 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-407.82998     -218.8871       0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.512753184055494 y final 0 7.372260513538923 z final 0 36.38854664693738      xy final 4.256376592027746 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5127532 7.3722605 36.388547) with tilt (4.2563766 0 0)
  triclinic box = (0 0 0) to (8.5127532 7.3722605 36.388547) with tilt (4.2563766 0 0)
  triclinic box = (0 0 0) to (8.5127532 7.3722605 36.388547) with tilt (4.2563766 0 0)
  triclinic box = (0 0 0) to (8.5127532 7.3722605 36.388547) with tilt (4.2563766 0 0)
  triclinic box = (0 0 0) to (8.5127532 7.3722605 36.388547) with tilt (4.2563766 0 0)
  triclinic box = (0 0 0) to (8.5127532 7.3722605 36.388547) with tilt (4.2563766 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-407.82998     -218.8871       0              0            
Loop time of 1.666e-06 on 1 procs for 0 steps with 80 atoms

180.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.666e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.523813926209941 y final 0 7.381839397229384 z final 0 36.43582680684977      xy final 4.26190696310497 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5238139 7.3722605 36.388547) with tilt (4.2563766 0 0)
  triclinic box = (0 0 0) to (8.5238139 7.3818394 36.388547) with tilt (4.2563766 0 0)
  triclinic box = (0 0 0) to (8.5238139 7.3818394 36.435827) with tilt (4.2563766 0 0)
  triclinic box = (0 0 0) to (8.5238139 7.3818394 36.435827) with tilt (4.261907 0 0)
  triclinic box = (0 0 0) to (8.5238139 7.3818394 36.435827) with tilt (4.261907 0 0)
  triclinic box = (0 0 0) to (8.5238139 7.3818394 36.435827) with tilt (4.261907 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-407.99975     -1619.0748      0              0            
Loop time of 1.541e-06 on 1 procs for 0 steps with 80 atoms

194.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.541e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.523813926209941 y final 0 7.381839397229384 z final 0 36.43582680684977      xy final 4.26190696310497 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5238139 7.3818394 36.435827) with tilt (4.261907 0 0)
  triclinic box = (0 0 0) to (8.5238139 7.3818394 36.435827) with tilt (4.261907 0 0)
  triclinic box = (0 0 0) to (8.5238139 7.3818394 36.435827) with tilt (4.261907 0 0)
  triclinic box = (0 0 0) to (8.5238139 7.3818394 36.435827) with tilt (4.261907 0 0)
  triclinic box = (0 0 0) to (8.5238139 7.3818394 36.435827) with tilt (4.261907 0 0)
  triclinic box = (0 0 0) to (8.5238139 7.3818394 36.435827) with tilt (4.261907 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-407.99975     -1619.0748      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 80 atoms

184.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.525614490504267 y final 0 7.383398731649415 z final 0 36.44352348457532      xy final 4.262807245252133 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5256145 7.3818394 36.435827) with tilt (4.261907 0 0)
  triclinic box = (0 0 0) to (8.5256145 7.3833987 36.435827) with tilt (4.261907 0 0)
  triclinic box = (0 0 0) to (8.5256145 7.3833987 36.443523) with tilt (4.261907 0 0)
  triclinic box = (0 0 0) to (8.5256145 7.3833987 36.443523) with tilt (4.2628072 0 0)
  triclinic box = (0 0 0) to (8.5256145 7.3833987 36.443523) with tilt (4.2628072 0 0)
  triclinic box = (0 0 0) to (8.5256145 7.3833987 36.443523) with tilt (4.2628072 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.07125     -1617.1373      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.525614490504267 y final 0 7.3833987316494145 z final 0 36.44352348457532      xy final 4.262807245252133 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5256145 7.3833987 36.443523) with tilt (4.2628072 0 0)
  triclinic box = (0 0 0) to (8.5256145 7.3833987 36.443523) with tilt (4.2628072 0 0)
  triclinic box = (0 0 0) to (8.5256145 7.3833987 36.443523) with tilt (4.2628072 0 0)
  triclinic box = (0 0 0) to (8.5256145 7.3833987 36.443523) with tilt (4.2628072 0 0)
  triclinic box = (0 0 0) to (8.5256145 7.3833987 36.443523) with tilt (4.2628072 0 0)
  triclinic box = (0 0 0) to (8.5256145 7.3833987 36.443523) with tilt (4.2628072 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.07125     -1617.1373      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.91e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.522081165771363 y final 0 7.3803387826709 z final 0 36.428419962943465      xy final 4.2610405828856805 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5220812 7.3833987 36.443523) with tilt (4.2628072 0 0)
  triclinic box = (0 0 0) to (8.5220812 7.3803388 36.443523) with tilt (4.2628072 0 0)
  triclinic box = (0 0 0) to (8.5220812 7.3803388 36.42842) with tilt (4.2628072 0 0)
  triclinic box = (0 0 0) to (8.5220812 7.3803388 36.42842) with tilt (4.2610406 0 0)
  triclinic box = (0 0 0) to (8.5220812 7.3803388 36.42842) with tilt (4.2610406 0 0)
  triclinic box = (0 0 0) to (8.5220812 7.3803388 36.42842) with tilt (4.2610406 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.07796     -568.19271      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.522081165771363 y final 0 7.3803387826709 z final 0 36.428419962943465      xy final 4.2610405828856805 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5220812 7.3803388 36.42842) with tilt (4.2610406 0 0)
  triclinic box = (0 0 0) to (8.5220812 7.3803388 36.42842) with tilt (4.2610406 0 0)
  triclinic box = (0 0 0) to (8.5220812 7.3803388 36.42842) with tilt (4.2610406 0 0)
  triclinic box = (0 0 0) to (8.5220812 7.3803388 36.42842) with tilt (4.2610406 0 0)
  triclinic box = (0 0 0) to (8.5220812 7.3803388 36.42842) with tilt (4.2610406 0 0)
  triclinic box = (0 0 0) to (8.5220812 7.3803388 36.42842) with tilt (4.2610406 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.07796     -568.19271      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.521569308965868 y final 0 7.379895501674239 z final 0 36.42623198393824      xy final 4.260784654482933 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5215693 7.3803388 36.42842) with tilt (4.2610406 0 0)
  triclinic box = (0 0 0) to (8.5215693 7.3798955 36.42842) with tilt (4.2610406 0 0)
  triclinic box = (0 0 0) to (8.5215693 7.3798955 36.426232) with tilt (4.2610406 0 0)
  triclinic box = (0 0 0) to (8.5215693 7.3798955 36.426232) with tilt (4.2607847 0 0)
  triclinic box = (0 0 0) to (8.5215693 7.3798955 36.426232) with tilt (4.2607847 0 0)
  triclinic box = (0 0 0) to (8.5215693 7.3798955 36.426232) with tilt (4.2607847 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.07882     -485.3057       0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.521569308965868 y final 0 7.3798955016742385 z final 0 36.42623198393824      xy final 4.260784654482933 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5215693 7.3798955 36.426232) with tilt (4.2607847 0 0)
  triclinic box = (0 0 0) to (8.5215693 7.3798955 36.426232) with tilt (4.2607847 0 0)
  triclinic box = (0 0 0) to (8.5215693 7.3798955 36.426232) with tilt (4.2607847 0 0)
  triclinic box = (0 0 0) to (8.5215693 7.3798955 36.426232) with tilt (4.2607847 0 0)
  triclinic box = (0 0 0) to (8.5215693 7.3798955 36.426232) with tilt (4.2607847 0 0)
  triclinic box = (0 0 0) to (8.5215693 7.3798955 36.426232) with tilt (4.2607847 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.07882     -485.3057       0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 80 atoms

104.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.521265061857092 y final 0 7.37963201594901 z final 0 36.42493145168145      xy final 4.260632530928545 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5212651 7.3798955 36.426232) with tilt (4.2607847 0 0)
  triclinic box = (0 0 0) to (8.5212651 7.379632 36.426232) with tilt (4.2607847 0 0)
  triclinic box = (0 0 0) to (8.5212651 7.379632 36.424931) with tilt (4.2607847 0 0)
  triclinic box = (0 0 0) to (8.5212651 7.379632 36.424931) with tilt (4.2606325 0 0)
  triclinic box = (0 0 0) to (8.5212651 7.379632 36.424931) with tilt (4.2606325 0 0)
  triclinic box = (0 0 0) to (8.5212651 7.379632 36.424931) with tilt (4.2606325 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.07911     -480.46712      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 80 atoms

141.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.08e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.521265061857092 y final 0 7.37963201594901 z final 0 36.42493145168145      xy final 4.260632530928545 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5212651 7.379632 36.424931) with tilt (4.2606325 0 0)
  triclinic box = (0 0 0) to (8.5212651 7.379632 36.424931) with tilt (4.2606325 0 0)
  triclinic box = (0 0 0) to (8.5212651 7.379632 36.424931) with tilt (4.2606325 0 0)
  triclinic box = (0 0 0) to (8.5212651 7.379632 36.424931) with tilt (4.2606325 0 0)
  triclinic box = (0 0 0) to (8.5212651 7.379632 36.424931) with tilt (4.2606325 0 0)
  triclinic box = (0 0 0) to (8.5212651 7.379632 36.424931) with tilt (4.2606325 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.07911     -480.46712      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 80 atoms

288.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.041e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.520825481761172 y final 0 7.379251328418945 z final 0 36.42305242612068      xy final 4.260412740880585 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5208255 7.379632 36.424931) with tilt (4.2606325 0 0)
  triclinic box = (0 0 0) to (8.5208255 7.3792513 36.424931) with tilt (4.2606325 0 0)
  triclinic box = (0 0 0) to (8.5208255 7.3792513 36.423052) with tilt (4.2606325 0 0)
  triclinic box = (0 0 0) to (8.5208255 7.3792513 36.423052) with tilt (4.2604127 0 0)
  triclinic box = (0 0 0) to (8.5208255 7.3792513 36.423052) with tilt (4.2604127 0 0)
  triclinic box = (0 0 0) to (8.5208255 7.3792513 36.423052) with tilt (4.2604127 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.07935     -482.42775      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 80 atoms

96.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.520271725783898 y final 0 7.378771761675126 z final 0 36.42068534525102      xy final 4.260135862891948 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5202717 7.3792513 36.423052) with tilt (4.2604127 0 0)
  triclinic box = (0 0 0) to (8.5202717 7.3787718 36.423052) with tilt (4.2604127 0 0)
  triclinic box = (0 0 0) to (8.5202717 7.3787718 36.420685) with tilt (4.2604127 0 0)
  triclinic box = (0 0 0) to (8.5202717 7.3787718 36.420685) with tilt (4.2601359 0 0)
  triclinic box = (0 0 0) to (8.5202717 7.3787718 36.420685) with tilt (4.2601359 0 0)
  triclinic box = (0 0 0) to (8.5202717 7.3787718 36.420685) with tilt (4.2601359 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.07947     -485.02449      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.167e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.520271725783898 y final 0 7.378771761675125 z final 0 36.42068534525102      xy final 4.260135862891948 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5202717 7.3787718 36.420685) with tilt (4.2601359 0 0)
  triclinic box = (0 0 0) to (8.5202717 7.3787718 36.420685) with tilt (4.2601359 0 0)
  triclinic box = (0 0 0) to (8.5202717 7.3787718 36.420685) with tilt (4.2601359 0 0)
  triclinic box = (0 0 0) to (8.5202717 7.3787718 36.420685) with tilt (4.2601359 0 0)
  triclinic box = (0 0 0) to (8.5202717 7.3787718 36.420685) with tilt (4.2601359 0 0)
  triclinic box = (0 0 0) to (8.5202717 7.3787718 36.420685) with tilt (4.2601359 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.07947     -485.02449      0              0            
Loop time of 6.66e-07 on 1 procs for 0 steps with 80 atoms

150.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.66e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.519732270910566 y final 0 7.3783045800506235 z final 0 36.418379395765854      xy final 4.259866135455282 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5197323 7.3787718 36.420685) with tilt (4.2601359 0 0)
  triclinic box = (0 0 0) to (8.5197323 7.3783046 36.420685) with tilt (4.2601359 0 0)
  triclinic box = (0 0 0) to (8.5197323 7.3783046 36.418379) with tilt (4.2601359 0 0)
  triclinic box = (0 0 0) to (8.5197323 7.3783046 36.418379) with tilt (4.2598661 0 0)
  triclinic box = (0 0 0) to (8.5197323 7.3783046 36.418379) with tilt (4.2598661 0 0)
  triclinic box = (0 0 0) to (8.5197323 7.3783046 36.418379) with tilt (4.2598661 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.07978     -453.90199      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

200.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.517574792947586 y final 0 7.376436149326574 z final 0 36.40915705772806      xy final 4.258787396473792 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5175748 7.3783046 36.418379) with tilt (4.2598661 0 0)
  triclinic box = (0 0 0) to (8.5175748 7.3764361 36.418379) with tilt (4.2598661 0 0)
  triclinic box = (0 0 0) to (8.5175748 7.3764361 36.409157) with tilt (4.2598661 0 0)
  triclinic box = (0 0 0) to (8.5175748 7.3764361 36.409157) with tilt (4.2587874 0 0)
  triclinic box = (0 0 0) to (8.5175748 7.3764361 36.409157) with tilt (4.2587874 0 0)
  triclinic box = (0 0 0) to (8.5175748 7.3764361 36.409157) with tilt (4.2587874 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.08062     -329.25747      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.517574792947586 y final 0 7.376436149326573 z final 0 36.40915705772806      xy final 4.258787396473792 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5175748 7.3764361 36.409157) with tilt (4.2587874 0 0)
  triclinic box = (0 0 0) to (8.5175748 7.3764361 36.409157) with tilt (4.2587874 0 0)
  triclinic box = (0 0 0) to (8.5175748 7.3764361 36.409157) with tilt (4.2587874 0 0)
  triclinic box = (0 0 0) to (8.5175748 7.3764361 36.409157) with tilt (4.2587874 0 0)
  triclinic box = (0 0 0) to (8.5175748 7.3764361 36.409157) with tilt (4.2587874 0 0)
  triclinic box = (0 0 0) to (8.5175748 7.3764361 36.409157) with tilt (4.2587874 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.08062     -329.25747      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 80 atoms

342.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.514560466093773 y final 0 7.373825665695854 z final 0 36.39627204027746      xy final 4.2572802330468855 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5145605 7.3764361 36.409157) with tilt (4.2587874 0 0)
  triclinic box = (0 0 0) to (8.5145605 7.3738257 36.409157) with tilt (4.2587874 0 0)
  triclinic box = (0 0 0) to (8.5145605 7.3738257 36.396272) with tilt (4.2587874 0 0)
  triclinic box = (0 0 0) to (8.5145605 7.3738257 36.396272) with tilt (4.2572802 0 0)
  triclinic box = (0 0 0) to (8.5145605 7.3738257 36.396272) with tilt (4.2572802 0 0)
  triclinic box = (0 0 0) to (8.5145605 7.3738257 36.396272) with tilt (4.2572802 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.08136     -154.09844      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.514560466093773 y final 0 7.373825665695853 z final 0 36.39627204027746      xy final 4.2572802330468855 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5145605 7.3738257 36.396272) with tilt (4.2572802 0 0)
  triclinic box = (0 0 0) to (8.5145605 7.3738257 36.396272) with tilt (4.2572802 0 0)
  triclinic box = (0 0 0) to (8.5145605 7.3738257 36.396272) with tilt (4.2572802 0 0)
  triclinic box = (0 0 0) to (8.5145605 7.3738257 36.396272) with tilt (4.2572802 0 0)
  triclinic box = (0 0 0) to (8.5145605 7.3738257 36.396272) with tilt (4.2572802 0 0)
  triclinic box = (0 0 0) to (8.5145605 7.3738257 36.396272) with tilt (4.2572802 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.08136     -154.09844      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 80 atoms

104.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.59e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.5118426525622 y final 0 7.371471970134753 z final 0 36.384654496301586      xy final 4.255921326281099 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5118427 7.3738257 36.396272) with tilt (4.2572802 0 0)
  triclinic box = (0 0 0) to (8.5118427 7.371472 36.396272) with tilt (4.2572802 0 0)
  triclinic box = (0 0 0) to (8.5118427 7.371472 36.384654) with tilt (4.2572802 0 0)
  triclinic box = (0 0 0) to (8.5118427 7.371472 36.384654) with tilt (4.2559213 0 0)
  triclinic box = (0 0 0) to (8.5118427 7.371472 36.384654) with tilt (4.2559213 0 0)
  triclinic box = (0 0 0) to (8.5118427 7.371472 36.384654) with tilt (4.2559213 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.08186     -10.321294      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 80 atoms

96.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.041e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.5118426525622 y final 0 7.371471970134753 z final 0 36.384654496301586      xy final 4.255921326281099 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5118427 7.371472 36.384654) with tilt (4.2559213 0 0)
  triclinic box = (0 0 0) to (8.5118427 7.371472 36.384654) with tilt (4.2559213 0 0)
  triclinic box = (0 0 0) to (8.5118427 7.371472 36.384654) with tilt (4.2559213 0 0)
  triclinic box = (0 0 0) to (8.5118427 7.371472 36.384654) with tilt (4.2559213 0 0)
  triclinic box = (0 0 0) to (8.5118427 7.371472 36.384654) with tilt (4.2559213 0 0)
  triclinic box = (0 0 0) to (8.5118427 7.371472 36.384654) with tilt (4.2559213 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.08186     -10.321294      0              0            
Loop time of 1.333e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.333e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.510941870212903 y final 0 7.3706918697369765 z final 0 36.380804019278756      xy final 4.255470935106451 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5109419 7.371472 36.384654) with tilt (4.2559213 0 0)
  triclinic box = (0 0 0) to (8.5109419 7.3706919 36.384654) with tilt (4.2559213 0 0)
  triclinic box = (0 0 0) to (8.5109419 7.3706919 36.380804) with tilt (4.2559213 0 0)
  triclinic box = (0 0 0) to (8.5109419 7.3706919 36.380804) with tilt (4.2554709 0 0)
  triclinic box = (0 0 0) to (8.5109419 7.3706919 36.380804) with tilt (4.2554709 0 0)
  triclinic box = (0 0 0) to (8.5109419 7.3706919 36.380804) with tilt (4.2554709 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.08201      19.793584      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 80 atoms

228.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.510941870212903 y final 0 7.370691869736976 z final 0 36.380804019278756      xy final 4.255470935106451 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5109419 7.3706919 36.380804) with tilt (4.2554709 0 0)
  triclinic box = (0 0 0) to (8.5109419 7.3706919 36.380804) with tilt (4.2554709 0 0)
  triclinic box = (0 0 0) to (8.5109419 7.3706919 36.380804) with tilt (4.2554709 0 0)
  triclinic box = (0 0 0) to (8.5109419 7.3706919 36.380804) with tilt (4.2554709 0 0)
  triclinic box = (0 0 0) to (8.5109419 7.3706919 36.380804) with tilt (4.2554709 0 0)
  triclinic box = (0 0 0) to (8.5109419 7.3706919 36.380804) with tilt (4.2554709 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.08201      19.793584      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.511060992362397 y final 0 7.3707950325445895 z final 0 36.38131321786584      xy final 4.255530496181198 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.511061 7.3706919 36.380804) with tilt (4.2554709 0 0)
  triclinic box = (0 0 0) to (8.511061 7.370795 36.380804) with tilt (4.2554709 0 0)
  triclinic box = (0 0 0) to (8.511061 7.370795 36.381313) with tilt (4.2554709 0 0)
  triclinic box = (0 0 0) to (8.511061 7.370795 36.381313) with tilt (4.2555305 0 0)
  triclinic box = (0 0 0) to (8.511061 7.370795 36.381313) with tilt (4.2555305 0 0)
  triclinic box = (0 0 0) to (8.511061 7.370795 36.381313) with tilt (4.2555305 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-408.08203      4.9135838      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

200.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4680 ave        4680 max        4680 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4680
Ave neighs/atom = 58.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.563797179930395 y final 0 10.488466675757506 z final 0 31.124698866015883      xy final -8.563797179930393 xz final 0.0 yz final -0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5637972 7.370795 36.381313) with tilt (4.2555305 0 0)
  triclinic box = (0 0 0) to (8.5637972 10.488467 36.381313) with tilt (4.2555305 0 0)
  triclinic box = (0 0 0) to (8.5637972 10.488467 31.124699) with tilt (4.2555305 0 0)
  triclinic box = (0 0 0) to (8.5637972 10.488467 31.124699) with tilt (-8.5637972 0 0)
  triclinic box = (0 0 0) to (8.5637972 10.488467 31.124699) with tilt (-8.5637972 0 0)
  triclinic box = (0 0 0) to (8.5637972 10.488467 31.124699) with tilt (-8.5637972 0 -0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-401.4648      -17614.103      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

200.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            574 ave         574 max         574 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4176 ave        4176 max        4176 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4176
Ave neighs/atom = 52.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 7.01144411433151 y final 0 8.58723022007626 z final 0 25.48274814189859      xy final -7.011444114331508 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.0114441 10.488467 31.124699) with tilt (-8.5637972 0 -0)
  triclinic box = (0 0 0) to (7.0114441 8.5872302 31.124699) with tilt (-8.5637972 0 -0)
  triclinic box = (0 0 0) to (7.0114441 8.5872302 25.482748) with tilt (-8.5637972 0 -0)
  triclinic box = (0 0 0) to (7.0114441 8.5872302 25.482748) with tilt (-7.0114441 0 -0)
  triclinic box = (0 0 0) to (7.0114441 8.5872302 25.482748) with tilt (-7.0114441 0 -0)
  triclinic box = (0 0 0) to (7.0114441 8.5872302 25.482748) with tilt (-7.0114441 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-244.70858      697561.26      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            778 ave         778 max         778 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           7360 ave        7360 max        7360 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 7360
Ave neighs/atom = 92
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.441724797054544 y final 0 10.338959150891757 z final 0 30.68103280561781      xy final -8.441724797054542 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4417248 8.5872302 25.482748) with tilt (-7.0114441 0 0)
  triclinic box = (0 0 0) to (8.4417248 10.338959 25.482748) with tilt (-7.0114441 0 0)
  triclinic box = (0 0 0) to (8.4417248 10.338959 30.681033) with tilt (-7.0114441 0 0)
  triclinic box = (0 0 0) to (8.4417248 10.338959 30.681033) with tilt (-8.4417248 0 0)
  triclinic box = (0 0 0) to (8.4417248 10.338959 30.681033) with tilt (-8.4417248 0 0)
  triclinic box = (0 0 0) to (8.4417248 10.338959 30.681033) with tilt (-8.4417248 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-401.96135      7079.8388      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 80 atoms

109.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.17e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            626 ave         626 max         626 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5040 ave        5040 max        5040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5040
Ave neighs/atom = 63
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.441724797054544 y final 0 10.338959150891757 z final 0 30.681032805617814      xy final -8.441724797054542 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4417248 10.338959 30.681033) with tilt (-8.4417248 0 0)
  triclinic box = (0 0 0) to (8.4417248 10.338959 30.681033) with tilt (-8.4417248 0 0)
  triclinic box = (0 0 0) to (8.4417248 10.338959 30.681033) with tilt (-8.4417248 0 0)
  triclinic box = (0 0 0) to (8.4417248 10.338959 30.681033) with tilt (-8.4417248 0 0)
  triclinic box = (0 0 0) to (8.4417248 10.338959 30.681033) with tilt (-8.4417248 0 0)
  triclinic box = (0 0 0) to (8.4417248 10.338959 30.681033) with tilt (-8.4417248 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-401.96135      7079.8388      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 80 atoms

109.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.17e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            626 ave         626 max         626 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5040 ave        5040 max        5040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5040
Ave neighs/atom = 63
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.486862754886515 y final 0 10.394241633251552 z final 0 30.845084489167657      xy final -8.486862754886513 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4868628 10.338959 30.681033) with tilt (-8.4417248 0 0)
  triclinic box = (0 0 0) to (8.4868628 10.394242 30.681033) with tilt (-8.4417248 0 0)
  triclinic box = (0 0 0) to (8.4868628 10.394242 30.845084) with tilt (-8.4417248 0 0)
  triclinic box = (0 0 0) to (8.4868628 10.394242 30.845084) with tilt (-8.4868628 0 0)
  triclinic box = (0 0 0) to (8.4868628 10.394242 30.845084) with tilt (-8.4868628 0 0)
  triclinic box = (0 0 0) to (8.4868628 10.394242 30.845084) with tilt (-8.4868628 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.76999      948.03748      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

300.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4608 ave        4608 max        4608 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4608
Ave neighs/atom = 57.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.486862754886515 y final 0 10.394241633251552 z final 0 30.845084489167657      xy final -8.486862754886513 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4868628 10.394242 30.845084) with tilt (-8.4868628 0 0)
  triclinic box = (0 0 0) to (8.4868628 10.394242 30.845084) with tilt (-8.4868628 0 0)
  triclinic box = (0 0 0) to (8.4868628 10.394242 30.845084) with tilt (-8.4868628 0 0)
  triclinic box = (0 0 0) to (8.4868628 10.394242 30.845084) with tilt (-8.4868628 0 0)
  triclinic box = (0 0 0) to (8.4868628 10.394242 30.845084) with tilt (-8.4868628 0 0)
  triclinic box = (0 0 0) to (8.4868628 10.394242 30.845084) with tilt (-8.4868628 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.76999      948.03748      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 80 atoms

82.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4608 ave        4608 max        4608 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4608
Ave neighs/atom = 57.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.496100152225853 y final 0 10.405555088267914 z final 0 30.878657354621136      xy final -8.496100152225852 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4961002 10.394242 30.845084) with tilt (-8.4868628 0 0)
  triclinic box = (0 0 0) to (8.4961002 10.405555 30.845084) with tilt (-8.4868628 0 0)
  triclinic box = (0 0 0) to (8.4961002 10.405555 30.878657) with tilt (-8.4868628 0 0)
  triclinic box = (0 0 0) to (8.4961002 10.405555 30.878657) with tilt (-8.4961002 0 0)
  triclinic box = (0 0 0) to (8.4961002 10.405555 30.878657) with tilt (-8.4961002 0 0)
  triclinic box = (0 0 0) to (8.4961002 10.405555 30.878657) with tilt (-8.4961002 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.89756      314.01799      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4416 ave        4416 max        4416 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4416
Ave neighs/atom = 55.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.496100152225853 y final 0 10.405555088267914 z final 0 30.87865735462114      xy final -8.496100152225852 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4961002 10.405555 30.878657) with tilt (-8.4961002 0 0)
  triclinic box = (0 0 0) to (8.4961002 10.405555 30.878657) with tilt (-8.4961002 0 0)
  triclinic box = (0 0 0) to (8.4961002 10.405555 30.878657) with tilt (-8.4961002 0 0)
  triclinic box = (0 0 0) to (8.4961002 10.405555 30.878657) with tilt (-8.4961002 0 0)
  triclinic box = (0 0 0) to (8.4961002 10.405555 30.878657) with tilt (-8.4961002 0 0)
  triclinic box = (0 0 0) to (8.4961002 10.405555 30.878657) with tilt (-8.4961002 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.89756      314.01799      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4416 ave        4416 max        4416 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4416
Ave neighs/atom = 55.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.495291524318421 y final 0 10.40456472538542 z final 0 30.87571843633703      xy final -8.49529152431842 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4952915 10.405555 30.878657) with tilt (-8.4961002 0 0)
  triclinic box = (0 0 0) to (8.4952915 10.404565 30.878657) with tilt (-8.4961002 0 0)
  triclinic box = (0 0 0) to (8.4952915 10.404565 30.875718) with tilt (-8.4961002 0 0)
  triclinic box = (0 0 0) to (8.4952915 10.404565 30.875718) with tilt (-8.4952915 0 0)
  triclinic box = (0 0 0) to (8.4952915 10.404565 30.875718) with tilt (-8.4952915 0 0)
  triclinic box = (0 0 0) to (8.4952915 10.404565 30.875718) with tilt (-8.4952915 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.93187      215.65691      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 80 atoms

88.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            600 ave         600 max         600 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4448 ave        4448 max        4448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4448
Ave neighs/atom = 55.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.495291524318421 y final 0 10.40456472538542 z final 0 30.875718436337035      xy final -8.49529152431842 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4952915 10.404565 30.875718) with tilt (-8.4952915 0 0)
  triclinic box = (0 0 0) to (8.4952915 10.404565 30.875718) with tilt (-8.4952915 0 0)
  triclinic box = (0 0 0) to (8.4952915 10.404565 30.875718) with tilt (-8.4952915 0 0)
  triclinic box = (0 0 0) to (8.4952915 10.404565 30.875718) with tilt (-8.4952915 0 0)
  triclinic box = (0 0 0) to (8.4952915 10.404565 30.875718) with tilt (-8.4952915 0 0)
  triclinic box = (0 0 0) to (8.4952915 10.404565 30.875718) with tilt (-8.4952915 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.93187      215.65691      0              0            
Loop time of 1.5e-06 on 1 procs for 0 steps with 80 atoms

133.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.5e-06    |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            600 ave         600 max         600 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4448 ave        4448 max        4448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4448
Ave neighs/atom = 55.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.49003020928541 y final 0 10.39812095678196 z final 0 30.8565964461029      xy final -8.490030209285408 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4900302 10.404565 30.875718) with tilt (-8.4952915 0 0)
  triclinic box = (0 0 0) to (8.4900302 10.398121 30.875718) with tilt (-8.4952915 0 0)
  triclinic box = (0 0 0) to (8.4900302 10.398121 30.856596) with tilt (-8.4952915 0 0)
  triclinic box = (0 0 0) to (8.4900302 10.398121 30.856596) with tilt (-8.4900302 0 0)
  triclinic box = (0 0 0) to (8.4900302 10.398121 30.856596) with tilt (-8.4900302 0 0)
  triclinic box = (0 0 0) to (8.4900302 10.398121 30.856596) with tilt (-8.4900302 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.94944      526.83954      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4544 ave        4544 max        4544 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4544
Ave neighs/atom = 56.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.49003020928541 y final 0 10.39812095678196 z final 0 30.8565964461029      xy final -8.490030209285408 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4900302 10.398121 30.856596) with tilt (-8.4900302 0 0)
  triclinic box = (0 0 0) to (8.4900302 10.398121 30.856596) with tilt (-8.4900302 0 0)
  triclinic box = (0 0 0) to (8.4900302 10.398121 30.856596) with tilt (-8.4900302 0 0)
  triclinic box = (0 0 0) to (8.4900302 10.398121 30.856596) with tilt (-8.4900302 0 0)
  triclinic box = (0 0 0) to (8.4900302 10.398121 30.856596) with tilt (-8.4900302 0 0)
  triclinic box = (0 0 0) to (8.4900302 10.398121 30.856596) with tilt (-8.4900302 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.94944      526.83954      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 80 atoms

88.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4544 ave        4544 max        4544 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4544
Ave neighs/atom = 56.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.486170124161061 y final 0 10.393393337322777 z final 0 30.84256715692565      xy final -8.48617012416106 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4861701 10.398121 30.856596) with tilt (-8.4900302 0 0)
  triclinic box = (0 0 0) to (8.4861701 10.393393 30.856596) with tilt (-8.4900302 0 0)
  triclinic box = (0 0 0) to (8.4861701 10.393393 30.842567) with tilt (-8.4900302 0 0)
  triclinic box = (0 0 0) to (8.4861701 10.393393 30.842567) with tilt (-8.4861701 0 0)
  triclinic box = (0 0 0) to (8.4861701 10.393393 30.842567) with tilt (-8.4861701 0 0)
  triclinic box = (0 0 0) to (8.4861701 10.393393 30.842567) with tilt (-8.4861701 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.95629      618.34523      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4576 ave        4576 max        4576 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4576
Ave neighs/atom = 57.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.482519153304413 y final 0 10.388921829490496 z final 0 30.82929788325017      xy final -8.48251915330441 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4825192 10.393393 30.842567) with tilt (-8.4861701 0 0)
  triclinic box = (0 0 0) to (8.4825192 10.388922 30.842567) with tilt (-8.4861701 0 0)
  triclinic box = (0 0 0) to (8.4825192 10.388922 30.829298) with tilt (-8.4861701 0 0)
  triclinic box = (0 0 0) to (8.4825192 10.388922 30.829298) with tilt (-8.4825192 0 0)
  triclinic box = (0 0 0) to (8.4825192 10.388922 30.829298) with tilt (-8.4825192 0 0)
  triclinic box = (0 0 0) to (8.4825192 10.388922 30.829298) with tilt (-8.4825192 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.95839      695.59654      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 80 atoms

266.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4576 ave        4576 max        4576 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4576
Ave neighs/atom = 57.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.482519153304413 y final 0 10.388921829490496 z final 0 30.829297883250174      xy final -8.48251915330441 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4825192 10.388922 30.829298) with tilt (-8.4825192 0 0)
  triclinic box = (0 0 0) to (8.4825192 10.388922 30.829298) with tilt (-8.4825192 0 0)
  triclinic box = (0 0 0) to (8.4825192 10.388922 30.829298) with tilt (-8.4825192 0 0)
  triclinic box = (0 0 0) to (8.4825192 10.388922 30.829298) with tilt (-8.4825192 0 0)
  triclinic box = (0 0 0) to (8.4825192 10.388922 30.829298) with tilt (-8.4825192 0 0)
  triclinic box = (0 0 0) to (8.4825192 10.388922 30.829298) with tilt (-8.4825192 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.95839      695.59654      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 80 atoms

208.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4576 ave        4576 max        4576 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4576
Ave neighs/atom = 57.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.481593228904341 y final 0 10.38778780833021 z final 0 30.825932656643396      xy final -8.48159322890434 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4815932 10.388922 30.829298) with tilt (-8.4825192 0 0)
  triclinic box = (0 0 0) to (8.4815932 10.387788 30.829298) with tilt (-8.4825192 0 0)
  triclinic box = (0 0 0) to (8.4815932 10.387788 30.825933) with tilt (-8.4825192 0 0)
  triclinic box = (0 0 0) to (8.4815932 10.387788 30.825933) with tilt (-8.4815932 0 0)
  triclinic box = (0 0 0) to (8.4815932 10.387788 30.825933) with tilt (-8.4815932 0 0)
  triclinic box = (0 0 0) to (8.4815932 10.387788 30.825933) with tilt (-8.4815932 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.96317      556.15979      0              0            
Loop time of 1.833e-06 on 1 procs for 0 steps with 80 atoms

654.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.833e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4576 ave        4576 max        4576 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4576
Ave neighs/atom = 57.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.478609537476082 y final 0 10.384133547555635 z final 0 30.815088577169945      xy final -8.47860953747608 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4786095 10.387788 30.825933) with tilt (-8.4815932 0 0)
  triclinic box = (0 0 0) to (8.4786095 10.384134 30.825933) with tilt (-8.4815932 0 0)
  triclinic box = (0 0 0) to (8.4786095 10.384134 30.815089) with tilt (-8.4815932 0 0)
  triclinic box = (0 0 0) to (8.4786095 10.384134 30.815089) with tilt (-8.4786095 0 0)
  triclinic box = (0 0 0) to (8.4786095 10.384134 30.815089) with tilt (-8.4786095 0 0)
  triclinic box = (0 0 0) to (8.4786095 10.384134 30.815089) with tilt (-8.4786095 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.96982      123.07668      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4576 ave        4576 max        4576 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4576
Ave neighs/atom = 57.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.478609537476082 y final 0 10.384133547555635 z final 0 30.81508857716995      xy final -8.47860953747608 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4786095 10.384134 30.815089) with tilt (-8.4786095 0 0)
  triclinic box = (0 0 0) to (8.4786095 10.384134 30.815089) with tilt (-8.4786095 0 0)
  triclinic box = (0 0 0) to (8.4786095 10.384134 30.815089) with tilt (-8.4786095 0 0)
  triclinic box = (0 0 0) to (8.4786095 10.384134 30.815089) with tilt (-8.4786095 0 0)
  triclinic box = (0 0 0) to (8.4786095 10.384134 30.815089) with tilt (-8.4786095 0 0)
  triclinic box = (0 0 0) to (8.4786095 10.384134 30.815089) with tilt (-8.4786095 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.96982      123.07668      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4576 ave        4576 max        4576 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4576
Ave neighs/atom = 57.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.477145179361715 y final 0 10.382340082465182 z final 0 30.809766439760093      xy final -8.477145179361713 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4771452 10.384134 30.815089) with tilt (-8.4786095 0 0)
  triclinic box = (0 0 0) to (8.4771452 10.38234 30.815089) with tilt (-8.4786095 0 0)
  triclinic box = (0 0 0) to (8.4771452 10.38234 30.809766) with tilt (-8.4786095 0 0)
  triclinic box = (0 0 0) to (8.4771452 10.38234 30.809766) with tilt (-8.4771452 0 0)
  triclinic box = (0 0 0) to (8.4771452 10.38234 30.809766) with tilt (-8.4771452 0 0)
  triclinic box = (0 0 0) to (8.4771452 10.38234 30.809766) with tilt (-8.4771452 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.97251      50.643979      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.33e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4576 ave        4576 max        4576 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4576
Ave neighs/atom = 57.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.474322541705728 y final 0 10.378883071472213 z final 0 30.79950770228598      xy final -8.474322541705726 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4743225 10.38234 30.809766) with tilt (-8.4771452 0 0)
  triclinic box = (0 0 0) to (8.4743225 10.378883 30.809766) with tilt (-8.4771452 0 0)
  triclinic box = (0 0 0) to (8.4743225 10.378883 30.799508) with tilt (-8.4771452 0 0)
  triclinic box = (0 0 0) to (8.4743225 10.378883 30.799508) with tilt (-8.4743225 0 0)
  triclinic box = (0 0 0) to (8.4743225 10.378883 30.799508) with tilt (-8.4743225 0 0)
  triclinic box = (0 0 0) to (8.4743225 10.378883 30.799508) with tilt (-8.4743225 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.97455     -89.278988      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 80 atoms

282.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.08e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4560 ave        4560 max        4560 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4560
Ave neighs/atom = 57
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.474322541705728 y final 0 10.378883071472213 z final 0 30.79950770228598      xy final -8.474322541705726 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4743225 10.378883 30.799508) with tilt (-8.4743225 0 0)
  triclinic box = (0 0 0) to (8.4743225 10.378883 30.799508) with tilt (-8.4743225 0 0)
  triclinic box = (0 0 0) to (8.4743225 10.378883 30.799508) with tilt (-8.4743225 0 0)
  triclinic box = (0 0 0) to (8.4743225 10.378883 30.799508) with tilt (-8.4743225 0 0)
  triclinic box = (0 0 0) to (8.4743225 10.378883 30.799508) with tilt (-8.4743225 0 0)
  triclinic box = (0 0 0) to (8.4743225 10.378883 30.799508) with tilt (-8.4743225 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.97455     -89.278988      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 80 atoms

184.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.084e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4560 ave        4560 max        4560 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4560
Ave neighs/atom = 57
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.472258749967416 y final 0 10.376355453125095 z final 0 30.792006952907226      xy final -8.472258749967414 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4722587 10.378883 30.799508) with tilt (-8.4743225 0 0)
  triclinic box = (0 0 0) to (8.4722587 10.376355 30.799508) with tilt (-8.4743225 0 0)
  triclinic box = (0 0 0) to (8.4722587 10.376355 30.792007) with tilt (-8.4743225 0 0)
  triclinic box = (0 0 0) to (8.4722587 10.376355 30.792007) with tilt (-8.4722587 0 0)
  triclinic box = (0 0 0) to (8.4722587 10.376355 30.792007) with tilt (-8.4722587 0 0)
  triclinic box = (0 0 0) to (8.4722587 10.376355 30.792007) with tilt (-8.4722587 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.97631      137.29677      0              0            
Loop time of 1.917e-06 on 1 procs for 0 steps with 80 atoms

313.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.917e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4592 ave        4592 max        4592 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4592
Ave neighs/atom = 57.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.465344531080705 y final 0 10.367887299003916 z final 0 30.766877565064302      xy final -8.465344531080703 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4653445 10.376355 30.792007) with tilt (-8.4722587 0 0)
  triclinic box = (0 0 0) to (8.4653445 10.367887 30.792007) with tilt (-8.4722587 0 0)
  triclinic box = (0 0 0) to (8.4653445 10.367887 30.766878) with tilt (-8.4722587 0 0)
  triclinic box = (0 0 0) to (8.4653445 10.367887 30.766878) with tilt (-8.4653445 0 0)
  triclinic box = (0 0 0) to (8.4653445 10.367887 30.766878) with tilt (-8.4653445 0 0)
  triclinic box = (0 0 0) to (8.4653445 10.367887 30.766878) with tilt (-8.4653445 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.97888      899.04566      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 80 atoms

92.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4608 ave        4608 max        4608 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4608
Ave neighs/atom = 57.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.465344531080705 y final 0 10.367887299003916 z final 0 30.766877565064306      xy final -8.465344531080703 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4653445 10.367887 30.766878) with tilt (-8.4653445 0 0)
  triclinic box = (0 0 0) to (8.4653445 10.367887 30.766878) with tilt (-8.4653445 0 0)
  triclinic box = (0 0 0) to (8.4653445 10.367887 30.766878) with tilt (-8.4653445 0 0)
  triclinic box = (0 0 0) to (8.4653445 10.367887 30.766878) with tilt (-8.4653445 0 0)
  triclinic box = (0 0 0) to (8.4653445 10.367887 30.766878) with tilt (-8.4653445 0 0)
  triclinic box = (0 0 0) to (8.4653445 10.367887 30.766878) with tilt (-8.4653445 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.97888      899.04566      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 80 atoms

171.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.167e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4608 ave        4608 max        4608 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4608
Ave neighs/atom = 57.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.464847958993051 y final 0 10.367279124886286 z final 0 30.765072798280507      xy final -8.46484795899305 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.464848 10.367887 30.766878) with tilt (-8.4653445 0 0)
  triclinic box = (0 0 0) to (8.464848 10.367279 30.766878) with tilt (-8.4653445 0 0)
  triclinic box = (0 0 0) to (8.464848 10.367279 30.765073) with tilt (-8.4653445 0 0)
  triclinic box = (0 0 0) to (8.464848 10.367279 30.765073) with tilt (-8.464848 0 0)
  triclinic box = (0 0 0) to (8.464848 10.367279 30.765073) with tilt (-8.464848 0 0)
  triclinic box = (0 0 0) to (8.464848 10.367279 30.765073) with tilt (-8.464848 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.9805       558.6803       0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 80 atoms

177.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4608 ave        4608 max        4608 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4608
Ave neighs/atom = 57.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.464382077045085 y final 0 10.366708538359841 z final 0 30.763379573297144      xy final -8.464382077045084 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4643821 10.367279 30.765073) with tilt (-8.464848 0 0)
  triclinic box = (0 0 0) to (8.4643821 10.366709 30.765073) with tilt (-8.464848 0 0)
  triclinic box = (0 0 0) to (8.4643821 10.366709 30.76338) with tilt (-8.464848 0 0)
  triclinic box = (0 0 0) to (8.4643821 10.366709 30.76338) with tilt (-8.4643821 0 0)
  triclinic box = (0 0 0) to (8.4643821 10.366709 30.76338) with tilt (-8.4643821 0 0)
  triclinic box = (0 0 0) to (8.4643821 10.366709 30.76338) with tilt (-8.4643821 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.981        240.29058      0              0            
Loop time of 1.291e-06 on 1 procs for 0 steps with 80 atoms

77.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.291e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4608 ave        4608 max        4608 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4608
Ave neighs/atom = 57.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.464382077045085 y final 0 10.366708538359841 z final 0 30.763379573297144      xy final -8.464382077045084 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4643821 10.366709 30.76338) with tilt (-8.4643821 0 0)
  triclinic box = (0 0 0) to (8.4643821 10.366709 30.76338) with tilt (-8.4643821 0 0)
  triclinic box = (0 0 0) to (8.4643821 10.366709 30.76338) with tilt (-8.4643821 0 0)
  triclinic box = (0 0 0) to (8.4643821 10.366709 30.76338) with tilt (-8.4643821 0 0)
  triclinic box = (0 0 0) to (8.4643821 10.366709 30.76338) with tilt (-8.4643821 0 0)
  triclinic box = (0 0 0) to (8.4643821 10.366709 30.76338) with tilt (-8.4643821 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.981        240.29058      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4608 ave        4608 max        4608 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4608
Ave neighs/atom = 57.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.463184778489826 y final 0 10.365242153094762 z final 0 30.759028050695406      xy final -8.463184778489824 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4631848 10.366709 30.76338) with tilt (-8.4643821 0 0)
  triclinic box = (0 0 0) to (8.4631848 10.365242 30.76338) with tilt (-8.4643821 0 0)
  triclinic box = (0 0 0) to (8.4631848 10.365242 30.759028) with tilt (-8.4643821 0 0)
  triclinic box = (0 0 0) to (8.4631848 10.365242 30.759028) with tilt (-8.4631848 0 0)
  triclinic box = (0 0 0) to (8.4631848 10.365242 30.759028) with tilt (-8.4631848 0 0)
  triclinic box = (0 0 0) to (8.4631848 10.365242 30.759028) with tilt (-8.4631848 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.98201      335.28482      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

200.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4608 ave        4608 max        4608 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4608
Ave neighs/atom = 57.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.459747232211248 y final 0 10.361032035919902 z final 0 30.74653445813234      xy final -8.459747232211246 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4597472 10.365242 30.759028) with tilt (-8.4631848 0 0)
  triclinic box = (0 0 0) to (8.4597472 10.361032 30.759028) with tilt (-8.4631848 0 0)
  triclinic box = (0 0 0) to (8.4597472 10.361032 30.746534) with tilt (-8.4631848 0 0)
  triclinic box = (0 0 0) to (8.4597472 10.361032 30.746534) with tilt (-8.4597472 0 0)
  triclinic box = (0 0 0) to (8.4597472 10.361032 30.746534) with tilt (-8.4597472 0 0)
  triclinic box = (0 0 0) to (8.4597472 10.361032 30.746534) with tilt (-8.4597472 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.98326      610.77767      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 80 atoms

92.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.083e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4576 ave        4576 max        4576 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4576
Ave neighs/atom = 57.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.459747232211248 y final 0 10.361032035919902 z final 0 30.74653445813234      xy final -8.459747232211246 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4597472 10.361032 30.746534) with tilt (-8.4597472 0 0)
  triclinic box = (0 0 0) to (8.4597472 10.361032 30.746534) with tilt (-8.4597472 0 0)
  triclinic box = (0 0 0) to (8.4597472 10.361032 30.746534) with tilt (-8.4597472 0 0)
  triclinic box = (0 0 0) to (8.4597472 10.361032 30.746534) with tilt (-8.4597472 0 0)
  triclinic box = (0 0 0) to (8.4597472 10.361032 30.746534) with tilt (-8.4597472 0 0)
  triclinic box = (0 0 0) to (8.4597472 10.361032 30.746534) with tilt (-8.4597472 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.98326      610.77767      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

300.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4576 ave        4576 max        4576 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4576
Ave neighs/atom = 57.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.457691804585025 y final 0 10.358514661476166 z final 0 30.739064107706827      xy final -8.457691804585023 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4576918 10.361032 30.746534) with tilt (-8.4597472 0 0)
  triclinic box = (0 0 0) to (8.4576918 10.358515 30.746534) with tilt (-8.4597472 0 0)
  triclinic box = (0 0 0) to (8.4576918 10.358515 30.739064) with tilt (-8.4597472 0 0)
  triclinic box = (0 0 0) to (8.4576918 10.358515 30.739064) with tilt (-8.4576918 0 0)
  triclinic box = (0 0 0) to (8.4576918 10.358515 30.739064) with tilt (-8.4576918 0 0)
  triclinic box = (0 0 0) to (8.4576918 10.358515 30.739064) with tilt (-8.4576918 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.98371      679.1477       0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 80 atoms

119.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.34e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4576 ave        4576 max        4576 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4576
Ave neighs/atom = 57.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.457691804585025 y final 0 10.358514661476166 z final 0 30.73906410770683      xy final -8.457691804585023 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4576918 10.358515 30.739064) with tilt (-8.4576918 0 0)
  triclinic box = (0 0 0) to (8.4576918 10.358515 30.739064) with tilt (-8.4576918 0 0)
  triclinic box = (0 0 0) to (8.4576918 10.358515 30.739064) with tilt (-8.4576918 0 0)
  triclinic box = (0 0 0) to (8.4576918 10.358515 30.739064) with tilt (-8.4576918 0 0)
  triclinic box = (0 0 0) to (8.4576918 10.358515 30.739064) with tilt (-8.4576918 0 0)
  triclinic box = (0 0 0) to (8.4576918 10.358515 30.739064) with tilt (-8.4576918 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.98371      679.1477       0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4576 ave        4576 max        4576 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4576
Ave neighs/atom = 57.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.456871389062249 y final 0 10.357509861772234 z final 0 30.736082347915588      xy final -8.456871389062247 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4568714 10.358515 30.739064) with tilt (-8.4576918 0 0)
  triclinic box = (0 0 0) to (8.4568714 10.35751 30.739064) with tilt (-8.4576918 0 0)
  triclinic box = (0 0 0) to (8.4568714 10.35751 30.736082) with tilt (-8.4576918 0 0)
  triclinic box = (0 0 0) to (8.4568714 10.35751 30.736082) with tilt (-8.4568714 0 0)
  triclinic box = (0 0 0) to (8.4568714 10.35751 30.736082) with tilt (-8.4568714 0 0)
  triclinic box = (0 0 0) to (8.4568714 10.35751 30.736082) with tilt (-8.4568714 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.98384      691.11945      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 80 atoms

208.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4608 ave        4608 max        4608 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4608
Ave neighs/atom = 57.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.456871389062249 y final 0 10.357509861772234 z final 0 30.736082347915588      xy final -8.456871389062247 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4568714 10.35751 30.736082) with tilt (-8.4568714 0 0)
  triclinic box = (0 0 0) to (8.4568714 10.35751 30.736082) with tilt (-8.4568714 0 0)
  triclinic box = (0 0 0) to (8.4568714 10.35751 30.736082) with tilt (-8.4568714 0 0)
  triclinic box = (0 0 0) to (8.4568714 10.35751 30.736082) with tilt (-8.4568714 0 0)
  triclinic box = (0 0 0) to (8.4568714 10.35751 30.736082) with tilt (-8.4568714 0 0)
  triclinic box = (0 0 0) to (8.4568714 10.35751 30.736082) with tilt (-8.4568714 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.98384      691.11945      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 80 atoms

114.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4608 ave        4608 max        4608 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4608
Ave neighs/atom = 57.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.456906166282824 y final 0 10.357552454994774 z final 0 30.736208744010106      xy final -8.456906166282822 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4569062 10.35751 30.736082) with tilt (-8.4568714 0 0)
  triclinic box = (0 0 0) to (8.4569062 10.357552 30.736082) with tilt (-8.4568714 0 0)
  triclinic box = (0 0 0) to (8.4569062 10.357552 30.736209) with tilt (-8.4568714 0 0)
  triclinic box = (0 0 0) to (8.4569062 10.357552 30.736209) with tilt (-8.4569062 0 0)
  triclinic box = (0 0 0) to (8.4569062 10.357552 30.736209) with tilt (-8.4569062 0 0)
  triclinic box = (0 0 0) to (8.4569062 10.357552 30.736209) with tilt (-8.4569062 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.98386      688.63732      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 80 atoms

109.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.17e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4608 ave        4608 max        4608 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4608
Ave neighs/atom = 57.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.456945261735951 y final 0 10.357600336950489 z final 0 30.736350834508055      xy final -8.45694526173595 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4569453 10.357552 30.736209) with tilt (-8.4569062 0 0)
  triclinic box = (0 0 0) to (8.4569453 10.3576 30.736209) with tilt (-8.4569062 0 0)
  triclinic box = (0 0 0) to (8.4569453 10.3576 30.736351) with tilt (-8.4569062 0 0)
  triclinic box = (0 0 0) to (8.4569453 10.3576 30.736351) with tilt (-8.4569453 0 0)
  triclinic box = (0 0 0) to (8.4569453 10.3576 30.736351) with tilt (-8.4569453 0 0)
  triclinic box = (0 0 0) to (8.4569453 10.3576 30.736351) with tilt (-8.4569453 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-402.98386      685.84193      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 80 atoms

100.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            613 ave         613 max         613 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4608 ave        4608 max        4608 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4608
Ave neighs/atom = 57.6
Neighbor list builds = 0
Dangerous builds = 0
Total wall time: 0:00:03
