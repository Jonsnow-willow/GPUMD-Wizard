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
pair_style eam/alloy
pair_coeff * * potentials/W.eam.alloy W
Reading eam/alloy potential file potentials/W.eam.alloy with DATE: 2018-03-30
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
  master list distance cutoff = 8.1287046
  ghost atom cutoff = 8.1287046
  binsize = 4.0643523, bins = 1 1 1
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair eam/alloy, perpetual
      attributes: half, newton on
      pair build: half/bin/newton/tri
      stencil: half/bin/3d/tri
      bin: standard
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.509054     -56440.564      0              0            
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
change_box all     x final 0 2.8423601065443576 y final 0 2.8423601065443576 z final 0 2.8423601065443576      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8423601 3.185 3.185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8423601 2.8423601 3.185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8423601 2.8423601 2.8423601) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8423601 2.8423601 2.8423601) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8423601 2.8423601 2.8423601) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8423601 2.8423601 2.8423601) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.670709      1797721.6      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.1668108703228675 y final 0 3.1668108703228675 z final 0 3.1668108703228675      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1668109 2.8423601 2.8423601) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1668109 3.1668109 2.8423601) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1668109 3.1668109 3.1668109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1668109 3.1668109 3.1668109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1668109 3.1668109 3.1668109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1668109 3.1668109 3.1668109) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.519882     -5742.5446      0              0            
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
change_box all     x final 0 3.164796015543819 y final 0 3.164796015543819 z final 0 3.164796015543819      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.164796 3.1668109 3.1668109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.164796 3.164796 3.1668109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.164796 3.164796 3.164796) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.164796 3.164796 3.164796) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.164796 3.164796 3.164796) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.164796 3.164796 3.164796) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.519988      157.21793      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.848316413989437 y final 0 2.848316413989437 z final 0 2.848316413989437      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8483164 3.164796 3.164796) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8483164 2.8483164 3.164796) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8483164 2.8483164 2.8483164) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8483164 2.8483164 2.8483164) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8483164 2.8483164 2.8483164) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8483164 2.8483164 2.8483164) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.830533      1742629        0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.8799643741448753 y final 0 2.8799643741448753 z final 0 2.8799643741448753      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8799644 2.8483164 2.8483164) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8799644 2.8799644 2.8483164) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8799644 2.8799644 2.8799644) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8799644 2.8799644 2.8799644) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8799644 2.8799644 2.8799644) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8799644 2.8799644 2.8799644) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-14.609829      1469973.3      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.9116123343003135 y final 0 2.9116123343003135 z final 0 2.9116123343003135      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9116123 2.8799644 2.8799644) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9116123 2.9116123 2.8799644) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9116123 2.9116123 2.9116123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9116123 2.9116123 2.9116123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9116123 2.9116123 2.9116123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9116123 2.9116123 2.9116123) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-15.27875       1227857        0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.9432602944557518 y final 0 2.9432602944557518 z final 0 2.9432602944557518      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9432603 2.9116123 2.9116123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9432603 2.9432603 2.9116123) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9432603 2.9432603 2.9432603) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9432603 2.9432603 2.9432603) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9432603 2.9432603 2.9432603) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9432603 2.9432603 2.9432603) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-15.846298      1012080.2      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.97490825461119 y final 0 2.97490825461119 z final 0 2.97490825461119      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9749083 2.9432603 2.9432603) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9749083 2.9749083 2.9432603) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9749083 2.9749083 2.9749083) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9749083 2.9749083 2.9749083) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9749083 2.9749083 2.9749083) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9749083 2.9749083 2.9749083) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-16.320251      818880.05      0              0            
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
change_box all     x final 0 3.0065562147666283 y final 0 3.0065562147666283 z final 0 3.0065562147666283      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0065562 2.9749083 2.9749083) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0065562 3.0065562 2.9749083) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0065562 3.0065562 3.0065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0065562 3.0065562 3.0065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0065562 3.0065562 3.0065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0065562 3.0065562 3.0065562) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-16.707275      645067.39      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.0382041749220665 y final 0 3.0382041749220665 z final 0 3.0382041749220665      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0382042 3.0065562 3.0065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0382042 3.0382042 3.0065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0382042 3.0382042 3.0382042) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0382042 3.0382042 3.0382042) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0382042 3.0382042 3.0382042) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0382042 3.0382042 3.0382042) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.013125      488118.59      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.0698521350775048 y final 0 3.0698521350775048 z final 0 3.0698521350775048      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0698521 3.0382042 3.0382042) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0698521 3.0698521 3.0382042) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0698521 3.0698521 3.0698521) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0698521 3.0698521 3.0698521) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0698521 3.0698521 3.0698521) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0698521 3.0698521 3.0698521) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.242901      346191.32      0              0            
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
change_box all     x final 0 3.101500095232943 y final 0 3.101500095232943 z final 0 3.101500095232943      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1015001 3.0698521 3.0698521) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1015001 3.1015001 3.0698521) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1015001 3.1015001 3.1015001) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1015001 3.1015001 3.1015001) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1015001 3.1015001 3.1015001) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1015001 3.1015001 3.1015001) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.40093       216070.77      0              0            
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
change_box all     x final 0 3.1331480553883813 y final 0 3.1331480553883813 z final 0 3.1331480553883813      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1331481 3.1015001 3.1015001) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1331481 3.1331481 3.1015001) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1331481 3.1331481 3.1331481) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1331481 3.1331481 3.1331481) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1331481 3.1331481 3.1331481) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1331481 3.1331481 3.1331481) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.491223      100526.43      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 2 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.084e-06  |            |       |100.00

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
change_box all     x final 0 3.164796015543819 y final 0 3.164796015543819 z final 0 3.164796015543819      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.164796 3.1331481 3.1331481) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.164796 3.164796 3.1331481) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.164796 3.164796 3.164796) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.164796 3.164796 3.164796) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.164796 3.164796 3.164796) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.164796 3.164796 3.164796) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.519988      157.21793      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.196443975699258 y final 0 3.196443975699258 z final 0 3.196443975699258      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.196444 3.164796 3.164796) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.196444 3.196444 3.164796) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.196444 3.196444 3.196444) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.196444 3.196444 3.196444) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.196444 3.196444 3.196444) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.196444 3.196444 3.196444) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.493478     -86024.936      0              0            
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
change_box all     x final 0 3.2280919358546956 y final 0 3.2280919358546956 z final 0 3.2280919358546956      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2280919 3.196444 3.196444) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2280919 3.2280919 3.196444) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2280919 3.2280919 3.2280919) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2280919 3.2280919 3.2280919) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2280919 3.2280919 3.2280919) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2280919 3.2280919 3.2280919) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.417857     -158936.28      0              0            
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
change_box all     x final 0 3.2597398960101343 y final 0 3.2597398960101343 z final 0 3.2597398960101343      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2597399 3.2280919 3.2280919) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2597399 3.2597399 3.2280919) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2597399 3.2597399 3.2597399) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2597399 3.2597399 3.2597399) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2597399 3.2597399 3.2597399) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2597399 3.2597399 3.2597399) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.299185     -219545.72      0              0            
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
change_box all     x final 0 3.291387856165572 y final 0 3.291387856165572 z final 0 3.291387856165572      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2913879 3.2597399 3.2597399) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2913879 3.2913879 3.2597399) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2913879 3.2913879 3.2913879) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2913879 3.2913879 3.2913879) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2913879 3.2913879 3.2913879) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2913879 3.2913879 3.2913879) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-17.141234     -277730.74      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 2 atoms

359.7% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 3.3230358163210107 y final 0 3.3230358163210107 z final 0 3.3230358163210107      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3230358 3.2913879 3.2913879) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3230358 3.3230358 3.2913879) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3230358 3.3230358 3.3230358) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3230358 3.3230358 3.3230358) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3230358 3.3230358 3.3230358) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3230358 3.3230358 3.3230358) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-16.941595     -338266.5       0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.3546837764764486 y final 0 3.3546837764764486 z final 0 3.3546837764764486      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3546838 3.3230358 3.3230358) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3546838 3.3546838 3.3230358) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3546838 3.3546838 3.3546838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3546838 3.3546838 3.3546838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3546838 3.3546838 3.3546838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3546838 3.3546838 3.3546838) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-16.698038     -398682.8       0              0            
Loop time of 1.291e-06 on 1 procs for 0 steps with 2 atoms

154.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.291e-06  |            |       |100.00

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
change_box all     x final 0 3.3863317366318872 y final 0 3.3863317366318872 z final 0 3.3863317366318872      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3863317 3.3546838 3.3546838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3863317 3.3863317 3.3546838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3863317 3.3863317 3.3863317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3863317 3.3863317 3.3863317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3863317 3.3863317 3.3863317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3863317 3.3863317 3.3863317) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-16.409832     -456798.56      0              0            
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
change_box all     x final 0 3.4179796967873246 y final 0 3.4179796967873246 z final 0 3.4179796967873246      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.4179797 3.3863317 3.3863317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4179797 3.4179797 3.3863317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4179797 3.4179797 3.4179797) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4179797 3.4179797 3.4179797) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4179797 3.4179797 3.4179797) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4179797 3.4179797 3.4179797) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-16.077679     -510684.73      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.4496276569427637 y final 0 3.4496276569427637 z final 0 3.4496276569427637      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.4496277 3.4179797 3.4179797) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4496277 3.4496277 3.4179797) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4496277 3.4496277 3.4496277) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4496277 3.4496277 3.4496277) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4496277 3.4496277 3.4496277) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4496277 3.4496277 3.4496277) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-15.703622     -558768.68      0              0            
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
change_box all     x final 0 3.481275617098201 y final 0 3.481275617098201 z final 0 3.481275617098201      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.4812756 3.4496277 3.4496277) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4812756 3.4812756 3.4496277) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4812756 3.4812756 3.4812756) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4812756 3.4812756 3.4812756) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4812756 3.4812756 3.4812756) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4812756 3.4812756 3.4812756) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-15.290817     -600014.54      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 9.494388046631457 y final 0 9.494388046631457 z final 0 9.494388046631457      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.494388 3.4812756 3.4812756) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494388 3.4812756) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
echo none
Created 52 atoms
  using lattice units in triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
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
-473.03968      157.21793      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 54 atoms

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
change_box all     x final 0 9.304500285698827 y final 0 9.494388046631457 z final 0 9.494388046631457      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.3045003 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3045003 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3045003 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3045003 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3045003 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3045003 9.494388 9.494388) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-472.46566      108212.71      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 54 atoms

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
change_box all     x final 0 9.399444166165143 y final 0 9.494388046631457 z final 0 9.494388046631457      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.3994442 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3994442 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3994442 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3994442 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3994442 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3994442 9.494388 9.494388) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-472.89724      53500.593      0              0            
Loop time of 6.66e-07 on 1 procs for 0 steps with 54 atoms

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
change_box all     x final 0 9.494388046631457 y final 0 9.494388046631457 z final 0 9.494388046631457      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-473.03968      157.21793      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 54 atoms

266.7% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 9.58933192709777 y final 0 9.494388046631457 z final 0 9.494388046631457      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.5893319 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.5893319 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.5893319 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.5893319 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.5893319 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.5893319 9.494388 9.494388) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-472.90346     -50632.976      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 54 atoms

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
change_box all     x final 0 9.684275807564086 y final 0 9.494388046631457 z final 0 9.494388046631457      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6842758 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.6842758 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.6842758 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.6842758 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.6842758 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.6842758 9.494388 9.494388) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-472.5053      -97739.065      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 54 atoms

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
change_box all     x final 0 9.494388046631457 y final 0 9.49440703538856 z final 0 9.49436905791233      xy final 0.0 xz final 0.0 yz final 0.01898873811582466 units box
Changing box ...
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494407 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494407 9.4943691) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494407 9.4943691) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494407 9.4943691) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.494388 9.494407 9.4943691) with tilt (0 0 0.018988738)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-473.03796      170.98165      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 54 atoms

423.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.08e-07   |            |       |100.00

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
change_box all     x final 0 9.494388046631457 y final 0 9.494588613460477 z final 0 9.494187484039276      xy final 0.0 xz final 0.0 yz final 0.0617122186462553 units box
Changing box ...
  triclinic box = (0 0 0) to (9.494388 9.494407 9.4943691) with tilt (0 0 0.018988738)
  triclinic box = (0 0 0) to (9.494388 9.4945886 9.4943691) with tilt (0 0 0.018988738)
  triclinic box = (0 0 0) to (9.494388 9.4945886 9.4941875) with tilt (0 0 0.018988738)
  triclinic box = (0 0 0) to (9.494388 9.4945886 9.4941875) with tilt (0 0 0.018988738)
  triclinic box = (0 0 0) to (9.494388 9.4945886 9.4941875) with tilt (0 0 0.018988738)
  triclinic box = (0 0 0) to (9.494388 9.4945886 9.4941875) with tilt (0 0 0.061712219)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-473.02153      302.55347      0              0            
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
change_box all     x final 0 9.494388046631457 y final 0 9.494962439733412 z final 0 9.49381368827713      xy final 0.0 xz final 0.0 yz final 0.10443195057104844 units box
Changing box ...
  triclinic box = (0 0 0) to (9.494388 9.4945886 9.4941875) with tilt (0 0 0.061712219)
  triclinic box = (0 0 0) to (9.494388 9.4949624 9.4941875) with tilt (0 0 0.061712219)
  triclinic box = (0 0 0) to (9.494388 9.4949624 9.4938137) with tilt (0 0 0.061712219)
  triclinic box = (0 0 0) to (9.494388 9.4949624 9.4938137) with tilt (0 0 0.061712219)
  triclinic box = (0 0 0) to (9.494388 9.4949624 9.4938137) with tilt (0 0 0.061712219)
  triclinic box = (0 0 0) to (9.494388 9.4949624 9.4938137) with tilt (0 0 0.10443195)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-472.98771      573.39047      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 54 atoms

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
change_box all     x final 0 9.494388046631457 y final 0 9.495528491501702 z final 0 9.493247738732473      xy final 0.0 xz final 0.0 yz final 0.14714533995035334 units box
Changing box ...
  triclinic box = (0 0 0) to (9.494388 9.4949624 9.4938137) with tilt (0 0 0.10443195)
  triclinic box = (0 0 0) to (9.494388 9.4955285 9.4938137) with tilt (0 0 0.10443195)
  triclinic box = (0 0 0) to (9.494388 9.4955285 9.4932477) with tilt (0 0 0.10443195)
  triclinic box = (0 0 0) to (9.494388 9.4955285 9.4932477) with tilt (0 0 0.10443195)
  triclinic box = (0 0 0) to (9.494388 9.4955285 9.4932477) with tilt (0 0 0.10443195)
  triclinic box = (0 0 0) to (9.494388 9.4955285 9.4932477) with tilt (0 0 0.14714534)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-472.93649      983.37517      0              0            
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
change_box all     x final 0 9.494388046631457 y final 0 9.49628673439099 z final 0 9.492489738495593      xy final 0.0 xz final 0.0 yz final 0.18984979476991184 units box
Changing box ...
  triclinic box = (0 0 0) to (9.494388 9.4955285 9.4932477) with tilt (0 0 0.14714534)
  triclinic box = (0 0 0) to (9.494388 9.4962867 9.4932477) with tilt (0 0 0.14714534)
  triclinic box = (0 0 0) to (9.494388 9.4962867 9.4924897) with tilt (0 0 0.14714534)
  triclinic box = (0 0 0) to (9.494388 9.4962867 9.4924897) with tilt (0 0 0.14714534)
  triclinic box = (0 0 0) to (9.494388 9.4962867 9.4924897) with tilt (0 0 0.14714534)
  triclinic box = (0 0 0) to (9.494388 9.4962867 9.4924897) with tilt (0 0 0.18984979)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-472.86786      1532.343       0              0            
Loop time of 5.83e-07 on 1 procs for 0 steps with 54 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.83e-07   |            |       |100.00

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
change_box all     x final 0 9.494388046631457 y final 0 9.494388046631457 z final 0 9.494388046631457      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.494388 9.4962867 9.4924897) with tilt (0 0 0.18984979)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.4924897) with tilt (0 0 0.18984979)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0.18984979)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0.18984979)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0.18984979)
  triclinic box = (0 0 0) to (9.494388 9.494388 9.494388) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-473.03874      160.13866      0              0            
Loop time of 1.667e-06 on 1 procs for 0 steps with 54 atoms

120.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.667e-06  |            |       |100.00

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
change_box all     x final 0 12.659184062175276 y final 0 15.823980077719096 z final 0 18.988776093262913      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.659184 9.494388 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 9.494388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
echo none
Created 185 atoms
  using lattice units in triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
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
-2089.7878     -714.05667      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.659184062175274 y final 0 15.823980077719096 z final 0 18.988776093262913      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2089.7878     -714.05667      0              0            
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
change_box all     x final 0 10.685153462906833 y final 0 13.356441828633544 z final 0 16.027730194360252      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.685153 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.685153 13.356442 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.685153 13.356442 16.02773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.685153 13.356442 16.02773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.685153 13.356442 16.02773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.685153 13.356442 16.02773) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-829.44075      4005419.3      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.657644768604847 y final 0 15.822055960756062 z final 0 18.986467152907274      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657645 13.356442 16.02773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657645 15.822056 16.02773) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657645 15.822056 18.986467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657645 15.822056 18.986467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657645 15.822056 18.986467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657645 15.822056 18.986467) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2089.7884      407.56509      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.657644768604847 y final 0 15.822055960756064 z final 0 18.986467152907274      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657645 15.822056 18.986467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657645 15.822056 18.986467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657645 15.822056 18.986467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657645 15.822056 18.986467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657645 15.822056 18.986467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657645 15.822056 18.986467) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2089.7884      407.56509      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.65839274133974 y final 0 15.822990926678218 z final 0 18.98758911201386      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.658393 15.822056 18.986467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658393 15.822991 18.986467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658393 15.822991 18.987589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658393 15.822991 18.987589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658393 15.822991 18.987589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658393 15.822991 18.987589) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2089.6907     -146.00508      0              0            
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
Neighs:          16208 ave       16208 max       16208 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16208
Ave neighs/atom = 67.8159
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.657987781035699 y final 0 15.822484726296251 z final 0 18.986981671555498      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657988 15.822991 18.987589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657988 15.822485 18.987589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657988 15.822485 18.986982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657988 15.822485 18.986982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657988 15.822485 18.986982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657988 15.822485 18.986982) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0357      71.38219       0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 239 atoms

384.2% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.657987781035699 y final 0 15.822484726296253 z final 0 18.986981671555498      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657988 15.822485 18.986982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657988 15.822485 18.986982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657988 15.822485 18.986982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657988 15.822485 18.986982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657988 15.822485 18.986982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657988 15.822485 18.986982) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0357      71.38219       0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 239 atoms

309.6% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.658054379501419 y final 0 15.822567974378732 z final 0 18.987081569254475      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.658054 15.822485 18.986982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658054 15.822568 18.986982) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658054 15.822568 18.987082) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658054 15.822568 18.987082) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658054 15.822568 18.987082) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658054 15.822568 18.987082) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0286      3.1106198      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.658018812524025 y final 0 15.822523515656815 z final 0 18.987028218788176      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.658019 15.822568 18.987082) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658019 15.822524 18.987082) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658019 15.822524 18.987028) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658019 15.822524 18.987028) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658019 15.822524 18.987028) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658019 15.822524 18.987028) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0589      14.460817      0              0            
Loop time of 6.66e-07 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.658018812524025 y final 0 15.822523515656817 z final 0 18.987028218788176      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.658019 15.822524 18.987028) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658019 15.822524 18.987028) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658019 15.822524 18.987028) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658019 15.822524 18.987028) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658019 15.822524 18.987028) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658019 15.822524 18.987028) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0589      14.460817      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.658099370982615 y final 0 15.822624213730082 z final 0 18.987149056476095      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.658099 15.822524 18.987028) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658099 15.822624 18.987028) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658099 15.822624 18.987149) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658099 15.822624 18.987149) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658099 15.822624 18.987149) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658099 15.822624 18.987149) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0593     -25.095463      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.658061469250834 y final 0 15.822576836565341 z final 0 18.987092203878404      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.658061 15.822624 18.987149) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658061 15.822577 18.987149) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658061 15.822577 18.987092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658061 15.822577 18.987092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658061 15.822577 18.987092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658061 15.822577 18.987092) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0611     -8.4824037      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.658061469250834 y final 0 15.822576836565341 z final 0 18.987092203878404      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.658061 15.822577 18.987092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658061 15.822577 18.987092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658061 15.822577 18.987092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658061 15.822577 18.987092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658061 15.822577 18.987092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658061 15.822577 18.987092) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0611     -8.4824037      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.658044615298975 y final 0 15.82255576912552 z final 0 18.987066922950618      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.658045 15.822577 18.987092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658045 15.822556 18.987092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658045 15.822556 18.987067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658045 15.822556 18.987067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658045 15.822556 18.987067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658045 15.822556 18.987067) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0612     -10.686445      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.658051892496765 y final 0 15.822564865622756 z final 0 18.9870778387473      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.658052 15.822556 18.987067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658052 15.822565 18.987067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658052 15.822565 18.987078) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658052 15.822565 18.987078) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658052 15.822565 18.987078) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658052 15.822565 18.987078) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0615     -10.016458      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.658051892496765 y final 0 15.822564865622757 z final 0 18.9870778387473      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.658052 15.822565 18.987078) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658052 15.822565 18.987078) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658052 15.822565 18.987078) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658052 15.822565 18.987078) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658052 15.822565 18.987078) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658052 15.822565 18.987078) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0615     -10.016458      0              0            
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
change_box all     x final 0 12.6580366413898 y final 0 15.82254580173905 z final 0 18.98705496208685      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.658037 15.822565 18.987078) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658037 15.822546 18.987078) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658037 15.822546 18.987055) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658037 15.822546 18.987055) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658037 15.822546 18.987055) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658037 15.822546 18.987055) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0616     -8.6474767      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.658041593563011 y final 0 15.822551991955564 z final 0 18.98706239034667      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.658042 15.822546 18.987055) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658042 15.822552 18.987055) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658042 15.822552 18.987062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658042 15.822552 18.987062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658042 15.822552 18.987062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.658042 15.822552 18.987062) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2090.0616     -9.1313474      0              0            
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
change_box all     x final 0 12.659184062175276 y final 0 15.823980077719096 z final 0 18.988776093262913      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.659184 15.822552 18.987062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.987062) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
echo none
1 atoms in group delatoms
Deleted 1 atoms, new total = 238
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.7122     -1099.1579      0              0            
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
change_box all     x final 0 12.659184062175274 y final 0 15.823980077719096 z final 0 18.988776093262913      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.7122     -1099.1579      0              0            
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
change_box all     x final 0 10.364463300577547 y final 0 12.955579125721936 z final 0 15.546694950866323      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.364463 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-237.11013      5588735        0              0            
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
change_box all     x final 0 12.657205727778404 y final 0 15.821507159723009 z final 0 18.98580859166761      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657206 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657206 15.821507 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657206 15.821507 18.985809) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657206 15.821507 18.985809) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657206 15.821507 18.985809) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657206 15.821507 18.985809) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.7132      329.77316      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.657205727778404 y final 0 15.82150715972301 z final 0 18.98580859166761      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657206 15.821507 18.985809) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657206 15.821507 18.985809) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657206 15.821507 18.985809) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657206 15.821507 18.985809) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657206 15.821507 18.985809) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657206 15.821507 18.985809) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.7132      329.77316      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 238 atoms

560.0% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.657904120589672 y final 0 15.822380150739562 z final 0 18.986856180887475      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657904 15.821507 18.985809) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657904 15.82238 18.985809) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657904 15.82238 18.986856) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657904 15.82238 18.986856) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657904 15.82238 18.986856) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657904 15.82238 18.986856) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.8209      85.080336      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
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
change_box all     x final 0 12.657588412990439 y final 0 15.821985516239407 z final 0 18.986382619487287      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657588 15.82238 18.986856) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657588 15.821986 18.986856) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657588 15.821986 18.986383) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657588 15.821986 18.986383) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657588 15.821986 18.986383) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657588 15.821986 18.986383) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0481      107.66935      0              0            
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
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657588412990439 y final 0 15.821985516239407 z final 0 18.986382619487287      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657588 15.821986 18.986383) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657588 15.821986 18.986383) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657588 15.821986 18.986383) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657588 15.821986 18.986383) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657588 15.821986 18.986383) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657588 15.821986 18.986383) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0481      107.66935      0              0            
Loop time of 1.542e-06 on 1 procs for 0 steps with 238 atoms

194.6% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657668794564183 y final 0 15.822085993206922 z final 0 18.986503191848303      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657669 15.821986 18.986383) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657669 15.822086 18.986383) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657669 15.822086 18.986503) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657669 15.822086 18.986503) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657669 15.822086 18.986503) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657669 15.822086 18.986503) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0371      20.184357      0              0            
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
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657626073906824 y final 0 15.822032592385046 z final 0 18.986439110862054      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657626 15.822086 18.986503) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657626 15.822033 18.986503) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657626 15.822033 18.986439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657626 15.822033 18.986439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657626 15.822033 18.986439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657626 15.822033 18.986439) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0869      26.253787      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657626073906822 y final 0 15.822032592385048 z final 0 18.986439110862054      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657626 15.822033 18.986439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657626 15.822033 18.986439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657626 15.822033 18.986439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657626 15.822033 18.986439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657626 15.822033 18.986439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657626 15.822033 18.986439) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0869      26.253787      0              0            
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
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.65767555639151 y final 0 15.822094445490936 z final 0 18.986513334589123      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657676 15.822033 18.986439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657676 15.822094 18.986439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657676 15.822094 18.986513) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657676 15.822094 18.986513) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657676 15.822094 18.986513) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657676 15.822094 18.986513) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0893     -51.726172      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657654926557901 y final 0 15.822068658198912 z final 0 18.98648238983869      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657655 15.822094 18.986513) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657655 15.822069 18.986513) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657655 15.822069 18.986482) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657655 15.822069 18.986482) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657655 15.822069 18.986482) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657655 15.822069 18.986482) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0917     -22.413729      0              0            
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
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657654926557901 y final 0 15.822068658198914 z final 0 18.98648238983869      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657655 15.822069 18.986482) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657655 15.822069 18.986482) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657655 15.822069 18.986482) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657655 15.822069 18.986482) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657655 15.822069 18.986482) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657655 15.822069 18.986482) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0917     -22.413729      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 238 atoms

521.9% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657594424914421 y final 0 15.82199303114457 z final 0 18.98639163737348      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657594 15.822069 18.986482) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657594 15.821993 18.986482) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657594 15.821993 18.986392) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657594 15.821993 18.986392) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657594 15.821993 18.986392) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657594 15.821993 18.986392) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0924     -32.301979      0              0            
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
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657617147561359 y final 0 15.82202143445324 z final 0 18.98642572134388      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657617 15.821993 18.986392) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657617 15.822021 18.986392) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657617 15.822021 18.986426) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657617 15.822021 18.986426) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657617 15.822021 18.986426) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657617 15.822021 18.986426) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0928     -29.138562      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657617147561357 y final 0 15.822021434453243 z final 0 18.98642572134388      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657617 15.822021 18.986426) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657617 15.822021 18.986426) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657617 15.822021 18.986426) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657617 15.822021 18.986426) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657617 15.822021 18.986426) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657617 15.822021 18.986426) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0928     -29.138562      0              0            
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
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657572665278693 y final 0 15.821965831599913 z final 0 18.986358997919883      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657573 15.822021 18.986426) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657573 15.821966 18.986426) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657573 15.821966 18.986359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657573 15.821966 18.986359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657573 15.821966 18.986359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657573 15.821966 18.986359) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0932     -31.555854      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657572665278693 y final 0 15.821965831599913 z final 0 18.986358997919883      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657573 15.821966 18.986359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657573 15.821966 18.986359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657573 15.821966 18.986359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657573 15.821966 18.986359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657573 15.821966 18.986359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657573 15.821966 18.986359) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0932     -31.555854      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657560929616158 y final 0 15.821951162021746 z final 0 18.98634139442608      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657561 15.821966 18.986359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657561 15.821951 18.986359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657561 15.821951 18.986341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657561 15.821951 18.986341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657561 15.821951 18.986341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657561 15.821951 18.986341) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0934     -34.655367      0              0            
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
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657560929616158 y final 0 15.821951162021747 z final 0 18.98634139442608      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657561 15.821951 18.986341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657561 15.821951 18.986341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657561 15.821951 18.986341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657561 15.821951 18.986341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657561 15.821951 18.986341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657561 15.821951 18.986341) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0934     -34.655367      0              0            
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
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.657566844435953 y final 0 15.821958555546493 z final 0 18.986350266655776      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.657567 15.821951 18.986341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657567 15.821959 18.986341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657567 15.821959 18.98635) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657567 15.821959 18.98635) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657567 15.821959 18.98635) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.657567 15.821959 18.98635) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.0935     -38.562226      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 238 atoms

312.8% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          16051 ave       16051 max       16051 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16051
Ave neighs/atom = 67.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.659184062175276 y final 0 15.823980077719096 z final 0 18.988776093262913      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.659184 15.821959 18.98635) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.98635) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6686     -2465.7884      0              0            
Loop time of 1.584e-06 on 1 procs for 0 steps with 238 atoms

252.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.584e-06  |            |       |100.00

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
change_box all     x final 0 12.659184062175274 y final 0 15.823980077719096 z final 0 18.988776093262913      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6686     -2465.7884      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 10.364463300577547 y final 0 12.955579125721936 z final 0 15.546694950866323      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.364463 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-239.8854       5582494.7      0              0            
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
change_box all     x final 0 12.656394579957496 y final 0 15.82049322494687 z final 0 18.984591869936242      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656395 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656395 15.820493 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656395 15.820493 18.984592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656395 15.820493 18.984592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656395 15.820493 18.984592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656395 15.820493 18.984592) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6705     -445.17409      0              0            
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
change_box all     x final 0 12.656394579957496 y final 0 15.82049322494687 z final 0 18.984591869936242      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656395 15.820493 18.984592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656395 15.820493 18.984592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656395 15.820493 18.984592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656395 15.820493 18.984592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656395 15.820493 18.984592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656395 15.820493 18.984592) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6705     -445.17409      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 238 atoms

436.2% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.655041698226698 y final 0 15.818802122783376 z final 0 18.982562547339146      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655042 15.820493 18.984592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655042 15.818802 18.984592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655042 15.818802 18.982563) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655042 15.818802 18.982563) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655042 15.818802 18.982563) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655042 15.818802 18.982563) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.9415     -869.06528      0              0            
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
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16081 ave       16081 max       16081 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16081
Ave neighs/atom = 67.567227
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.655566533368958 y final 0 15.8194581667112 z final 0 18.983349800052885      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655567 15.818802 18.982563) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655567 15.819458 18.982563) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655567 15.819458 18.98335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655567 15.819458 18.98335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655567 15.819458 18.98335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655567 15.819458 18.98335) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.1206     -845.43451      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.655566533368958 y final 0 15.819458166711202 z final 0 18.983349800052885      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655567 15.819458 18.98335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655567 15.819458 18.98335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655567 15.819458 18.98335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655567 15.819458 18.98335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655567 15.819458 18.98335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655567 15.819458 18.98335) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.1206     -845.43451      0              0            
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
change_box all     x final 0 12.655328913999162 y final 0 15.819161142498956 z final 0 18.98299337099807      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655329 15.819458 18.98335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655329 15.819161 18.98335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655329 15.819161 18.982993) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655329 15.819161 18.982993) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655329 15.819161 18.982993) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655329 15.819161 18.982993) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.1189     -1076.6144      0              0            
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
change_box all     x final 0 12.655448942546995 y final 0 15.819311178183748 z final 0 18.98317341381988      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655449 15.819161 18.982993) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655449 15.819311 18.982993) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655449 15.819311 18.983173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655449 15.819311 18.983173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655449 15.819311 18.983173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655449 15.819311 18.983173) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.1699     -1008.0501      0              0            
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
change_box all     x final 0 12.655448942546995 y final 0 15.81931117818375 z final 0 18.98317341381988      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655449 15.819311 18.983173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655449 15.819311 18.983173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655449 15.819311 18.983173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655449 15.819311 18.983173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655449 15.819311 18.983173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655449 15.819311 18.983173) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.1699     -1008.0501      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.655586415839943 y final 0 15.819483019799936 z final 0 18.98337962375929      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655586 15.819311 18.983173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655586 15.819483 18.983173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655586 15.819483 18.98338) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655586 15.819483 18.98338) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655586 15.819483 18.98338) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655586 15.819483 18.98338) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.1749     -1100.2232      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.655535581220951 y final 0 15.819419476526194 z final 0 18.983303371830807      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655536 15.819483 18.98338) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655536 15.819419 18.98338) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655536 15.819419 18.983303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655536 15.819419 18.983303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655536 15.819419 18.983303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655536 15.819419 18.983303) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.1775     -1070.7077      0              0            
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
change_box all     x final 0 12.65553558122095 y final 0 15.819419476526194 z final 0 18.983303371830807      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655536 15.819419 18.983303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655536 15.819419 18.983303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655536 15.819419 18.983303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655536 15.819419 18.983303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655536 15.819419 18.983303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655536 15.819419 18.983303) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.1775     -1070.7077      0              0            
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
change_box all     x final 0 12.655493480434762 y final 0 15.819366850543462 z final 0 18.983240220651528      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655493 15.819419 18.983303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655493 15.819367 18.983303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655493 15.819367 18.98324) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655493 15.819367 18.98324) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655493 15.819367 18.98324) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655493 15.819367 18.98324) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.179      -1094.6881      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.655508668374237 y final 0 15.819385835467804 z final 0 18.983263002560737      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655509 15.819367 18.98324) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655509 15.819386 18.98324) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655509 15.819386 18.983263) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655509 15.819386 18.983263) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655509 15.819386 18.983263) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655509 15.819386 18.983263) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.1796     -1087.0734      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 238 atoms

359.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.34e-07   |            |       |100.00

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
change_box all     x final 0 12.655508668374237 y final 0 15.819385835467804 z final 0 18.983263002560737      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655509 15.819386 18.983263) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655509 15.819386 18.983263) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655509 15.819386 18.983263) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655509 15.819386 18.983263) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655509 15.819386 18.983263) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655509 15.819386 18.983263) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.1796     -1087.0734      0              0            
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
change_box all     x final 0 12.655469153348392 y final 0 15.8193364416855 z final 0 18.98320373002197      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655469 15.819386 18.983263) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655469 15.819336 18.983263) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655469 15.819336 18.983204) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655469 15.819336 18.983204) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655469 15.819336 18.983204) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655469 15.819336 18.983204) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.18       -1072.8399      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.655484271203502 y final 0 15.819355339004387 z final 0 18.983226406804636      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.655484 15.819336 18.983204) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655484 15.819355 18.983204) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655484 15.819355 18.983226) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655484 15.819355 18.983226) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655484 15.819355 18.983226) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.655484 15.819355 18.983226) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2078.1802     -1078.5964      0              0            
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
change_box all     x final 0 12.659184062175276 y final 0 15.823980077719096 z final 0 18.988776093262913      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.659184 15.819355 18.983226) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.983226) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.1523     -1555.2946      0              0            
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
change_box all     x final 0 12.659184062175274 y final 0 15.823980077719096 z final 0 18.988776093262913      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.1523     -1555.2946      0              0            
Loop time of 1.625e-06 on 1 procs for 0 steps with 238 atoms

61.5% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 10.36446330057755 y final 0 12.955579125721936 z final 0 15.546694950866323      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.364463 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-239.18345      5580949.2      0              0            
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
change_box all     x final 0 12.656637106106801 y final 0 15.820796382633503 z final 0 18.984955659160203      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656637 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656637 15.820796 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656637 15.820796 18.984956) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656637 15.820796 18.984956) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656637 15.820796 18.984956) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656637 15.820796 18.984956) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
WARNING: Too many warnings: 101 vs 100. All future warnings will be suppressed (src/src/thermo.cpp:465)
-2077.1539      284.78702      0              0            
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
change_box all     x final 0 12.656637106106801 y final 0 15.820796382633503 z final 0 18.984955659160203      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656637 15.820796 18.984956) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656637 15.820796 18.984956) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656637 15.820796 18.984956) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656637 15.820796 18.984956) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656637 15.820796 18.984956) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656637 15.820796 18.984956) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.1539      284.78702      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 238 atoms

384.2% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.656672570223252 y final 0 15.820840712779066 z final 0 18.98500885533488      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656673 15.820796 18.984956) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656673 15.820841 18.984956) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656673 15.820841 18.985009) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656673 15.820841 18.985009) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656673 15.820841 18.985009) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656673 15.820841 18.985009) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.4057      358.18817      0              0            
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
change_box all     x final 0 12.65665830287965 y final 0 15.820822878599564 z final 0 18.984987454319477      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656658 15.820841 18.985009) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656658 15.820823 18.985009) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656658 15.820823 18.984987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656658 15.820823 18.984987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656658 15.820823 18.984987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656658 15.820823 18.984987) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6132      188.63847      0              0            
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
change_box all     x final 0 12.65665830287965 y final 0 15.820822878599566 z final 0 18.984987454319477      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656658 15.820823 18.984987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656658 15.820823 18.984987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656658 15.820823 18.984987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656658 15.820823 18.984987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656658 15.820823 18.984987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656658 15.820823 18.984987) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6132      188.63847      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 238 atoms

309.6% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.656771232922688 y final 0 15.820964041153365 z final 0 18.985156849384037      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656771 15.820823 18.984987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656771 15.820964 18.984987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656771 15.820964 18.985157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656771 15.820964 18.985157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656771 15.820964 18.985157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656771 15.820964 18.985157) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6014      80.571581      0              0            
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
change_box all     x final 0 12.656711284258234 y final 0 15.820889105322799 z final 0 18.985066926387358      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656711 15.820964 18.985157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656711 15.820889 18.985157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656711 15.820889 18.985067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656711 15.820889 18.985067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656711 15.820889 18.985067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656711 15.820889 18.985067) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6561      92.148625      0              0            
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
change_box all     x final 0 12.656711284258234 y final 0 15.8208891053228 z final 0 18.985066926387358      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656711 15.820889 18.985067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656711 15.820889 18.985067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656711 15.820889 18.985067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656711 15.820889 18.985067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656711 15.820889 18.985067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656711 15.820889 18.985067) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6561      92.148625      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.656886033350695 y final 0 15.821107541688377 z final 0 18.985329050026053      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656886 15.820889 18.985067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656886 15.821108 18.985067) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656886 15.821108 18.985329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656886 15.821108 18.985329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656886 15.821108 18.985329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656886 15.821108 18.985329) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6574     -4.216801       0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 238 atoms

369.3% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.656805867713226 y final 0 15.821007334641541 z final 0 18.985208801569847      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656806 15.821108 18.985329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656806 15.821007 18.985329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656806 15.821007 18.985209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656806 15.821007 18.985209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656806 15.821007 18.985209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656806 15.821007 18.985209) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6605      36.163603      0              0            
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
change_box all     x final 0 12.656805867713224 y final 0 15.821007334641541 z final 0 18.985208801569847      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656806 15.821007 18.985209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656806 15.821007 18.985209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656806 15.821007 18.985209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656806 15.821007 18.985209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656806 15.821007 18.985209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656806 15.821007 18.985209) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6605      36.163603      0              0            
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
change_box all     x final 0 12.656788626082575 y final 0 15.82098578260323 z final 0 18.985182939123874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656789 15.821007 18.985209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.661       21.880225      0              0            
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
change_box all     x final 0 12.656795383069168 y final 0 15.82099422883647 z final 0 18.98519307460376      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656795 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656795 15.820994 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656795 15.820994 18.985193) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656795 15.820994 18.985193) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656795 15.820994 18.985193) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656795 15.820994 18.985193) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6614      26.982276      0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 238 atoms

82.7% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.656795383069168 y final 0 15.82099422883647 z final 0 18.98519307460376      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656795 15.820994 18.985193) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656795 15.820994 18.985193) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656795 15.820994 18.985193) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656795 15.820994 18.985193) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656795 15.820994 18.985193) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656795 15.820994 18.985193) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6614      26.982276      0              0            
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
change_box all     x final 0 12.656763171698389 y final 0 15.820953964622998 z final 0 18.985144757547594      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656763 15.820994 18.985193) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656763 15.820954 18.985193) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656763 15.820954 18.985145) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656763 15.820954 18.985145) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656763 15.820954 18.985145) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656763 15.820954 18.985145) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6616      27.828413      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.656763171698389 y final 0 15.820953964623 z final 0 18.985144757547594      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656763 15.820954 18.985145) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656763 15.820954 18.985145) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656763 15.820954 18.985145) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656763 15.820954 18.985145) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656763 15.820954 18.985145) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656763 15.820954 18.985145) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6616      27.828413      0              0            
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
change_box all     x final 0 12.656753704788859 y final 0 15.820942130986088 z final 0 18.9851305571833      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656754 15.820954 18.985145) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656754 15.820942 18.985145) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656754 15.820942 18.985131) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656754 15.820942 18.985131) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656754 15.820942 18.985131) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656754 15.820942 18.985131) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6616      30.796532      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.659184062175276 y final 0 15.823980077719096 z final 0 18.988776093262913      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.659184 15.820942 18.985131) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.985131) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.1678     -1539.9029      0              0            
Loop time of 1.458e-06 on 1 procs for 0 steps with 238 atoms

205.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.458e-06  |            |       |100.00

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
change_box all     x final 0 12.659184062175274 y final 0 15.823980077719096 z final 0 18.988776093262913      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.1678     -1539.9029      0              0            
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
change_box all     x final 0 10.36446330057755 y final 0 12.955579125721936 z final 0 15.546694950866323      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.364463 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-239.8658       5581039.3      0              0            
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
change_box all     x final 0 12.656558496264301 y final 0 15.820698120330379 z final 0 18.984837744396454      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656558 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656558 15.820698 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656558 15.820698 18.984838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656558 15.820698 18.984838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656558 15.820698 18.984838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656558 15.820698 18.984838) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.1694      355.15057      0              0            
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
change_box all     x final 0 12.656558496264301 y final 0 15.820698120330379 z final 0 18.984837744396454      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656558 15.820698 18.984838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656558 15.820698 18.984838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656558 15.820698 18.984838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656558 15.820698 18.984838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656558 15.820698 18.984838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656558 15.820698 18.984838) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.1694      355.15057      0              0            
Loop time of 1.291e-06 on 1 procs for 0 steps with 238 atoms

697.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.291e-06  |            |       |100.00

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
change_box all     x final 0 12.65640365374524 y final 0 15.820504567182589 z final 0 18.984605480619106      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656404 15.820698 18.984838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656404 15.820505 18.984838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656404 15.820505 18.984605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656404 15.820505 18.984605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656404 15.820505 18.984605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656404 15.820505 18.984605) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.5585      339.70978      0              0            
Loop time of 1.541e-06 on 1 procs for 0 steps with 238 atoms

64.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.541e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16068 ave       16068 max       16068 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16068
Ave neighs/atom = 67.512605
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.656459742114397 y final 0 15.820574677643659 z final 0 18.98468961317239      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.65646 15.820505 18.984605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65646 15.820575 18.984605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65646 15.820575 18.98469) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65646 15.820575 18.98469) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65646 15.820575 18.98469) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65646 15.820575 18.98469) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.7445      344.26115      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 238 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.084e-06  |            |       |100.00

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
change_box all     x final 0 12.656459742114395 y final 0 15.82057467764366 z final 0 18.98468961317239      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.65646 15.820575 18.98469) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65646 15.820575 18.98469) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65646 15.820575 18.98469) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65646 15.820575 18.98469) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65646 15.820575 18.98469) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65646 15.820575 18.98469) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.7445      344.26115      0              0            
Loop time of 1.291e-06 on 1 procs for 0 steps with 238 atoms

309.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.291e-06  |            |       |100.00

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
change_box all     x final 0 12.6566886294108 y final 0 15.82086078676429 z final 0 18.985032944117144      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656689 15.820575 18.98469) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656689 15.820861 18.98469) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656689 15.820861 18.985033) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656689 15.820861 18.985033) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656689 15.820861 18.985033) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656689 15.820861 18.985033) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.7289      245.27374      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16059 ave       16059 max       16059 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16059
Ave neighs/atom = 67.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.656566354750566 y final 0 15.820707943438933 z final 0 18.984849532126717      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656566 15.820861 18.985033) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656566 15.820708 18.985033) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656566 15.820708 18.98485) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656566 15.820708 18.98485) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656566 15.820708 18.98485) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656566 15.820708 18.98485) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.7953      248.2252       0              0            
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
change_box all     x final 0 12.656566354750565 y final 0 15.820707943438933 z final 0 18.984849532126717      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656566 15.820708 18.98485) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656566 15.820708 18.98485) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656566 15.820708 18.98485) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656566 15.820708 18.98485) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656566 15.820708 18.98485) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656566 15.820708 18.98485) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.7953      248.2252       0              0            
Loop time of 1.666e-06 on 1 procs for 0 steps with 238 atoms

240.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.666e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
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
change_box all     x final 0 12.656747267506491 y final 0 15.820934084383852 z final 0 18.985120901260622      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656747 15.820708 18.98485) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656747 15.820934 18.98485) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656747 15.820934 18.985121) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656747 15.820934 18.985121) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656747 15.820934 18.985121) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656747 15.820934 18.985121) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.7986      188.5208       0              0            
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
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16055 ave       16055 max       16055 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16055
Ave neighs/atom = 67.457983
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.656671287431289 y final 0 15.820839109289846 z final 0 18.985006931147815      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656671 15.820934 18.985121) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656671 15.820839 18.985121) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656671 15.820839 18.985007) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656671 15.820839 18.985007) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656671 15.820839 18.985007) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656671 15.820839 18.985007) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.8023      209.33472      0              0            
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
Nghost:           2052 ave        2052 max        2052 min
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
change_box all     x final 0 12.656671287431287 y final 0 15.820839109289846 z final 0 18.985006931147815      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656671 15.820839 18.985007) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656671 15.820839 18.985007) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656671 15.820839 18.985007) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656671 15.820839 18.985007) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656671 15.820839 18.985007) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656671 15.820839 18.985007) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.8023      209.33472      0              0            
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
Nghost:           2052 ave        2052 max        2052 min
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
change_box all     x final 0 12.656586323753432 y final 0 15.820732904692527 z final 0 18.984879485631037      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656586 15.820839 18.985007) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656586 15.820733 18.985007) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656586 15.820733 18.984879) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656586 15.820733 18.984879) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656586 15.820733 18.984879) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656586 15.820733 18.984879) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.8033      232.96328      0              0            
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
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16059 ave       16059 max       16059 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16059
Ave neighs/atom = 67.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.656618753237613 y final 0 15.820773441547754 z final 0 18.984928129857305      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656619 15.820733 18.984879) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656619 15.820773 18.984879) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656619 15.820773 18.984928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656619 15.820773 18.984928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656619 15.820773 18.984928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656619 15.820773 18.984928) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.804       223.2233       0              0            
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
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16055 ave       16055 max       16055 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16055
Ave neighs/atom = 67.457983
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.656618753237613 y final 0 15.820773441547756 z final 0 18.984928129857305      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656619 15.820773 18.984928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656619 15.820773 18.984928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656619 15.820773 18.984928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656619 15.820773 18.984928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656619 15.820773 18.984928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656619 15.820773 18.984928) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.804       223.2233       0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16055 ave       16055 max       16055 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16055
Ave neighs/atom = 67.457983
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.65655318823906 y final 0 15.820691485299566 z final 0 18.984829782359476      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656553 15.820773 18.984928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656553 15.820691 18.984928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656553 15.820691 18.98483) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656553 15.820691 18.98483) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656553 15.820691 18.98483) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656553 15.820691 18.98483) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.8043      258.50181      0              0            
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
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16059 ave       16059 max       16059 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16059
Ave neighs/atom = 67.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.656574214614198 y final 0 15.820717768268487 z final 0 18.984861321922182      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656574 15.820691 18.98483) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656574 15.820718 18.98483) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656574 15.820718 18.984861) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656574 15.820718 18.984861) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656574 15.820718 18.984861) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656574 15.820718 18.984861) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.8044      247.00595      0              0            
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
Neighs:          16059 ave       16059 max       16059 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16059
Ave neighs/atom = 67.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.656574214614198 y final 0 15.820717768268489 z final 0 18.984861321922182      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656574 15.820718 18.984861) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656574 15.820718 18.984861) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656574 15.820718 18.984861) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656574 15.820718 18.984861) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656574 15.820718 18.984861) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656574 15.820718 18.984861) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.8044      247.00595      0              0            
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
Neighs:          16059 ave       16059 max       16059 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16059
Ave neighs/atom = 67.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.65654557299189 y final 0 15.820681966240604 z final 0 18.98481835948872      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656546 15.820718 18.984861) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656546 15.820682 18.984861) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656546 15.820682 18.984818) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656546 15.820682 18.984818) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656546 15.820682 18.984818) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656546 15.820682 18.984818) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.8045      267.42721      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2052 ave        2052 max        2052 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16059 ave       16059 max       16059 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16059
Ave neighs/atom = 67.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.656557064251645 y final 0 15.8206963303153 z final 0 18.984835596378353      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656557 15.820682 18.984818) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656557 15.820696 18.984818) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656557 15.820696 18.984836) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656557 15.820696 18.984836) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656557 15.820696 18.984836) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656557 15.820696 18.984836) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.8046      259.15204      0              0            
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
Neighs:          16059 ave       16059 max       16059 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16059
Ave neighs/atom = 67.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.659184062175276 y final 0 15.823980077719096 z final 0 18.988776093262913      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.659184 15.820696 18.984836) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.984836) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.1668     -1578.9588      0              0            
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
change_box all     x final 0 12.659184062175274 y final 0 15.823980077719096 z final 0 18.988776093262913      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.659184 15.82398 18.988776) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.1668     -1578.9588      0              0            
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
change_box all     x final 0 10.364463300577547 y final 0 12.955579125721936 z final 0 15.546694950866323      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.364463 15.82398 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 18.988776) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.364463 12.955579 15.546695) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-239.04527      5582103.3      0              0            
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
change_box all     x final 0 12.656630583813428 y final 0 15.820788229766785 z final 0 18.984945875720143      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656631 12.955579 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656631 15.820788 15.546695) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656631 15.820788 18.984946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656631 15.820788 18.984946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656631 15.820788 18.984946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656631 15.820788 18.984946) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.1684      264.83534      0              0            
Loop time of 6.67e-07 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.656630583813428 y final 0 15.820788229766785 z final 0 18.984945875720143      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656631 15.820788 18.984946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656631 15.820788 18.984946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656631 15.820788 18.984946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656631 15.820788 18.984946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656631 15.820788 18.984946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656631 15.820788 18.984946) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.1684      264.83534      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 238 atoms

257.3% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.656794422109156 y final 0 15.820993027636447 z final 0 18.98519163316482      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656794 15.820788 18.984946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656794 15.820993 18.984946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656794 15.820993 18.985192) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656794 15.820993 18.985192) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656794 15.820993 18.985192) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656794 15.820993 18.985192) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.5115      235.39006      0              0            
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
change_box all     x final 0 12.656736825996397 y final 0 15.820921032495498 z final 0 18.985105238995303      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656737 15.820993 18.985192) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656737 15.820921 18.985192) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656737 15.820921 18.985105) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656737 15.820921 18.985105) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656737 15.820921 18.985105) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656737 15.820921 18.985105) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6528      166.43449      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 238 atoms

92.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.084e-06  |            |       |100.00

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
change_box all     x final 0 12.656736825996395 y final 0 15.820921032495498 z final 0 18.985105238995303      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656737 15.820921 18.985105) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656737 15.820921 18.985105) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656737 15.820921 18.985105) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656737 15.820921 18.985105) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656737 15.820921 18.985105) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656737 15.820921 18.985105) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6528      166.43449      0              0            
Loop time of 1.458e-06 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.656807536207856 y final 0 15.821009420259823 z final 0 18.985211304312614      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656808 15.820921 18.985105) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656808 15.821009 18.985105) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656808 15.821009 18.985211) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656808 15.821009 18.985211) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656808 15.821009 18.985211) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656808 15.821009 18.985211) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6362     -10.233237      0              0            
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
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          16055 ave       16055 max       16055 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16055
Ave neighs/atom = 67.457983
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.656768891183262 y final 0 15.820961113979081 z final 0 18.985153336775657      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656769 15.821009 18.985211) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656769 15.820961 18.985211) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656769 15.820961 18.985153) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656769 15.820961 18.985153) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656769 15.820961 18.985153) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656769 15.820961 18.985153) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6899      43.518134      0              0            
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
change_box all     x final 0 12.65676889118326 y final 0 15.820961113979081 z final 0 18.985153336775657      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656769 15.820961 18.985153) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656769 15.820961 18.985153) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656769 15.820961 18.985153) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656769 15.820961 18.985153) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656769 15.820961 18.985153) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656769 15.820961 18.985153) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6899      43.518134      0              0            
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
change_box all     x final 0 12.656901641223813 y final 0 15.821127051529773 z final 0 18.9853524618365      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656902 15.820961 18.985153) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656902 15.821127 18.985153) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656902 15.821127 18.985352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656902 15.821127 18.985352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656902 15.821127 18.985352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656902 15.821127 18.985352) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6909     -57.673305      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 238 atoms

343.1% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.656839810870768 y final 0 15.821049763588466 z final 0 18.985259716306924      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.65684 15.821127 18.985352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65684 15.82105 18.985352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65684 15.82105 18.98526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65684 15.82105 18.98526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65684 15.82105 18.98526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65684 15.82105 18.98526) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6941     -14.222682      0              0            
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
change_box all     x final 0 12.656839810870768 y final 0 15.821049763588467 z final 0 18.985259716306924      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.65684 15.82105 18.98526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65684 15.82105 18.98526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65684 15.82105 18.98526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65684 15.82105 18.98526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65684 15.82105 18.98526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.65684 15.82105 18.98526) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6941     -14.222682      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 238 atoms

417.1% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.656811029006496 y final 0 15.82101378625813 z final 0 18.985216543510518      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656811 15.82105 18.98526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656811 15.821014 18.98526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656811 15.821014 18.985217) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656811 15.821014 18.985217) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656811 15.821014 18.985217) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656811 15.821014 18.985217) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6946     -16.589952      0              0            
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
change_box all     x final 0 12.656822578141723 y final 0 15.821028222677162 z final 0 18.985233867213356      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656823 15.821014 18.985217) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656823 15.821028 18.985217) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656823 15.821028 18.985234) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656823 15.821028 18.985234) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656823 15.821028 18.985234) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656823 15.821028 18.985234) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6949     -16.1602        0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.656822578141721 y final 0 15.821028222677164 z final 0 18.985233867213356      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656823 15.821028 18.985234) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656823 15.821028 18.985234) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656823 15.821028 18.985234) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656823 15.821028 18.985234) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656823 15.821028 18.985234) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656823 15.821028 18.985234) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6949     -16.1602        0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 238 atoms

479.8% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.656788959262773 y final 0 15.820986199078478 z final 0 18.985183438894932      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656789 15.821028 18.985234) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985234) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6952     -9.7436154      0              0            
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
change_box all     x final 0 12.656788959262773 y final 0 15.820986199078478 z final 0 18.985183438894932      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656789 15.820986 18.985183) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6952     -9.7436154      0              0            
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
change_box all     x final 0 12.65677889482443 y final 0 15.82097361853055 z final 0 18.98516834223742      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656779 15.820986 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656779 15.820974 18.985183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656779 15.820974 18.985168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656779 15.820974 18.985168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656779 15.820974 18.985168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656779 15.820974 18.985168) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6953     -8.4769171      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.65677889482443 y final 0 15.82097361853055 z final 0 18.98516834223742      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656779 15.820974 18.985168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656779 15.820974 18.985168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656779 15.820974 18.985168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656779 15.820974 18.985168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656779 15.820974 18.985168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656779 15.820974 18.985168) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6953     -8.4769171      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 238 atoms

436.4% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.656785808762884 y final 0 15.820982260953617 z final 0 18.9851787131451      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.656786 15.820974 18.985168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656786 15.820982 18.985168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656786 15.820982 18.985179) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656786 15.820982 18.985179) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656786 15.820982 18.985179) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.656786 15.820982 18.985179) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2077.6953     -11.923201      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 238 atoms

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
change_box all boundary p p f
Changing box ...
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 8.951394894652802 y final 0 6.329592031087638 z final 0 40.140638512968806      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9513949 15.820982 18.985179) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 6.329592 18.985179) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 6.329592 40.140639) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 6.329592 40.140639) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 6.329592 40.140639) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 6.329592 40.140639) with tilt (0 0 0)
echo none
158 atoms in group delatoms
Deleted 158 atoms, new total = 80
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.59366      2773.3873      0              0            
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
change_box all     x final 0 8.951394894652802 y final 0 6.329592031087637 z final 0 40.1406385129688      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9513949 6.329592 40.140639) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 6.329592 40.140639) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 6.329592 40.140639) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 6.329592 40.140639) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 6.329592 40.140639) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 6.329592 40.140639) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.59366      2773.3873      0              0            
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
change_box all     x final 0 7.328782283197491 y final 0 5.182231650288774 z final 0 32.86437519875399      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.3287823 6.329592 40.140639) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.3287823 5.1822317 40.140639) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.3287823 5.1822317 32.864375) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.3287823 5.1822317 32.864375) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.3287823 5.1822317 32.864375) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.3287823 5.1822317 32.864375) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.335 | 3.335 | 3.335 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-93.87292       3114229.1      0              0            
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
change_box all     x final 0 8.940153721925508 y final 0 6.321643321623679 z final 0 40.09022984971368      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9401537 5.1822317 32.864375) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9401537 6.3216433 32.864375) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9401537 6.3216433 40.09023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9401537 6.3216433 40.09023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9401537 6.3216433 40.09023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9401537 6.3216433 40.09023) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.61258      9403.0354      0              0            
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
change_box all     x final 0 8.940153721925508 y final 0 6.321643321623678 z final 0 40.09022984971368      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9401537 6.3216433 40.09023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9401537 6.3216433 40.09023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9401537 6.3216433 40.09023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9401537 6.3216433 40.09023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9401537 6.3216433 40.09023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9401537 6.3216433 40.09023) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.61258      9403.0354      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 80 atoms

369.0% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 8.946489710684023 y final 0 6.326123542240343 z final 0 40.11864225217953      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9464897 6.3216433 40.09023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9464897 6.3261235 40.09023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9464897 6.3261235 40.118642) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9464897 6.3261235 40.118642) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9464897 6.3261235 40.118642) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9464897 6.3261235 40.118642) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.56654      8853.4396      0              0            
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
change_box all     x final 0 8.942561495242298 y final 0 6.32334587446354 z final 0 40.10102700026595      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9425615 6.3261235 40.118642) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425615 6.3233459 40.118642) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425615 6.3233459 40.101027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425615 6.3233459 40.101027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425615 6.3233459 40.101027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425615 6.3233459 40.101027) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64011      9170.4756      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.942561495242298 y final 0 6.32334587446354 z final 0 40.10102700026595      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9425615 6.3233459 40.101027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425615 6.3233459 40.101027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425615 6.3233459 40.101027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425615 6.3233459 40.101027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425615 6.3233459 40.101027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425615 6.3233459 40.101027) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64011      9170.4756      0              0            
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
change_box all     x final 0 8.943576155384658 y final 0 6.324063347530801 z final 0 40.105577029224264      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9435762 6.3233459 40.101027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9435762 6.3240633 40.101027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9435762 6.3240633 40.105577) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9435762 6.3240633 40.105577) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9435762 6.3240633 40.105577) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9435762 6.3240633 40.105577) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64101      9133.6215      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 80 atoms

277.0% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 8.943106311886913 y final 0 6.32373111800745 z final 0 40.10347011536082      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9431063 6.3240633 40.105577) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9431063 6.3237311 40.105577) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9431063 6.3237311 40.10347) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9431063 6.3237311 40.10347) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9431063 6.3237311 40.10347) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9431063 6.3237311 40.10347) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64364      9147.7813      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 80 atoms

70.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.416e-06  |            |       |100.00

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
change_box all     x final 0 8.943106311886913 y final 0 6.32373111800745 z final 0 40.103470115360814      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9431063 6.3237311 40.10347) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9431063 6.3237311 40.10347) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9431063 6.3237311 40.10347) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9431063 6.3237311 40.10347) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9431063 6.3237311 40.10347) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9431063 6.3237311 40.10347) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64364      9147.7813      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.943316026178733 y final 0 6.323879408405306 z final 0 40.104410534779014      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.943316 6.3237311 40.10347) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.943316 6.3238794 40.10347) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.943316 6.3238794 40.104411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.943316 6.3238794 40.104411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.943316 6.3238794 40.104411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.943316 6.3238794 40.104411) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64436      9176.3884      0              0            
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
change_box all     x final 0 8.943244070060803 y final 0 6.32382852774637 z final 0 40.104087862776204      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9432441 6.3238794 40.104411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432441 6.3238285 40.104411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432441 6.3238285 40.104088) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432441 6.3238285 40.104088) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432441 6.3238285 40.104088) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432441 6.3238285 40.104088) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64464      9166.0595      0              0            
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
change_box all     x final 0 8.943244070060803 y final 0 6.32382852774637 z final 0 40.104087862776204      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9432441 6.3238285 40.104088) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432441 6.3238285 40.104088) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432441 6.3238285 40.104088) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432441 6.3238285 40.104088) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432441 6.3238285 40.104088) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432441 6.3238285 40.104088) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64464      9166.0595      0              0            
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
change_box all     x final 0 8.94321105665112 y final 0 6.323805183740511 z final 0 40.103939821117905      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9432111 6.3238285 40.104088) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432111 6.3238052 40.104088) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432111 6.3238052 40.10394) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432111 6.3238052 40.10394) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432111 6.3238052 40.10394) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432111 6.3238052 40.10394) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64449      9191.1211      0              0            
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
change_box all     x final 0 8.943229560636368 y final 0 6.3238182680339605 z final 0 40.10402279832891      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9432296 6.3238052 40.10394) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432296 6.3238183 40.10394) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432296 6.3238183 40.104023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432296 6.3238183 40.104023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432296 6.3238183 40.104023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432296 6.3238183 40.104023) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64488      9176.7743      0              0            
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
change_box all     x final 0 8.943229560636368 y final 0 6.3238182680339605 z final 0 40.104022798328906      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9432296 6.3238183 40.104023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432296 6.3238183 40.104023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432296 6.3238183 40.104023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432296 6.3238183 40.104023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432296 6.3238183 40.104023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9432296 6.3238183 40.104023) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64488      9176.7743      0              0            
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
change_box all     x final 0 8.943098497639545 y final 0 6.323725592500144 z final 0 40.103435074020034      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9430985 6.3238183 40.104023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9430985 6.3237256 40.104023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9430985 6.3237256 40.103435) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9430985 6.3237256 40.103435) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9430985 6.3237256 40.103435) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9430985 6.3237256 40.103435) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64497      9191.2158      0              0            
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
change_box all     x final 0 8.942574264859244 y final 0 6.323354903946273 z final 0 40.101084262914235      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9425743 6.3237256 40.103435) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425743 6.3233549 40.103435) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425743 6.3233549 40.101084) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425743 6.3233549 40.101084) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425743 6.3233549 40.101084) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9425743 6.3233549 40.101084) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64531      9249.0961      0              0            
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
change_box all     x final 0 8.940939879772822 y final 0 6.322199219168591 z final 0 40.093755208424135      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9409399 6.3233549 40.101084) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9409399 6.3221992 40.101084) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9409399 6.3221992 40.093755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9409399 6.3221992 40.093755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9409399 6.3221992 40.093755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9409399 6.3221992 40.093755) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64576      9430.7541      0              0            
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
change_box all     x final 0 8.940939879772822 y final 0 6.322199219168591 z final 0 40.09375520842413      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9409399 6.3221992 40.093755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9409399 6.3221992 40.093755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9409399 6.3221992 40.093755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9409399 6.3221992 40.093755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9409399 6.3221992 40.093755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9409399 6.3221992 40.093755) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64576      9430.7541      0              0            
Loop time of 1.291e-06 on 1 procs for 0 steps with 80 atoms

232.4% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 8.940347477923499 y final 0 6.321780327803747 z final 0 40.09109870753595      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9403475 6.3221992 40.093755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403475 6.3217803 40.093755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403475 6.3217803 40.091099) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403475 6.3217803 40.091099) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403475 6.3217803 40.091099) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403475 6.3217803 40.091099) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64581      9745.4105      0              0            
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
change_box all     x final 0 8.940347477923499 y final 0 6.321780327803747 z final 0 40.09109870753595      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9403475 6.3217803 40.091099) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403475 6.3217803 40.091099) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403475 6.3217803 40.091099) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403475 6.3217803 40.091099) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403475 6.3217803 40.091099) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403475 6.3217803 40.091099) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64581      9745.4105      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.940312454727037 y final 0 6.32175556266403 z final 0 40.09094165341362      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9403125 6.3217803 40.091099) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403125 6.3217556 40.091099) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403125 6.3217556 40.090942) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403125 6.3217556 40.090942) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403125 6.3217556 40.090942) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9403125 6.3217556 40.090942) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-682.64581      9764.0193      0              0            
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
change_box all     x final 0 6.329592031087638 y final 0 6.329592031087638 z final 0 50.065562147666284      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.329592 6.3217556 40.090942) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.329592 6.329592 40.090942) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.329592 6.329592 50.065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.329592 6.329592 50.065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.329592 6.329592 50.065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.329592 6.329592 50.065562) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.85504     -23128.376      0              0            
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
change_box all     x final 0 6.329592031087637 y final 0 6.329592031087637 z final 0 50.06556214766628      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.329592 6.329592 50.065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.329592 6.329592 50.065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.329592 6.329592 50.065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.329592 6.329592 50.065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.329592 6.329592 50.065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.329592 6.329592 50.065562) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.85504     -23128.376      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.166e-06  |            |       |100.00

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
change_box all     x final 0 5.182231650288774 y final 0 5.182231650288774 z final 0 40.99021540043131      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.1822317 6.329592 50.065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.1822317 5.1822317 50.065562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.1822317 5.1822317 40.990215) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.1822317 5.1822317 40.990215) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.1822317 5.1822317 40.990215) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.1822317 5.1822317 40.990215) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-96.485183      3431782        0              0            
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
change_box all     x final 0 6.312638932307709 y final 0 6.312638932307709 z final 0 49.93146718287339      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3126389 5.1822317 40.990215) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3126389 6.3126389 40.990215) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3126389 6.3126389 49.931467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3126389 6.3126389 49.931467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3126389 6.3126389 49.931467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3126389 6.3126389 49.931467) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.94578     -7088.9496      0              0            
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
change_box all     x final 0 6.312638932307709 y final 0 6.312638932307709 z final 0 49.93146718287339      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3126389 6.3126389 49.931467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3126389 6.3126389 49.931467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3126389 6.3126389 49.931467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3126389 6.3126389 49.931467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3126389 6.3126389 49.931467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3126389 6.3126389 49.931467) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.94578     -7088.9496      0              0            
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
change_box all     x final 0 6.312314411305147 y final 0 6.312314411305147 z final 0 49.92890029920345      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3123144 6.3126389 49.931467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3123144 6.3123144 49.931467) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3123144 6.3123144 49.9289) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3123144 6.3123144 49.9289) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3123144 6.3123144 49.9289) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3123144 6.3123144 49.9289) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.94768     -5959.5604      0              0            
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
change_box all     x final 0 6.312467545558538 y final 0 6.312467545558538 z final 0 49.93011155459596      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3124675 6.3123144 49.9289) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3124675 6.3124675 49.9289) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3124675 6.3124675 49.930112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3124675 6.3124675 49.930112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3124675 6.3124675 49.930112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3124675 6.3124675 49.930112) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.95519     -6511.523       0              0            
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
change_box all     x final 0 6.312467545558538 y final 0 6.312467545558538 z final 0 49.93011155459595      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3124675 6.3124675 49.930112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3124675 6.3124675 49.930112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3124675 6.3124675 49.930112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3124675 6.3124675 49.930112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3124675 6.3124675 49.930112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3124675 6.3124675 49.930112) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.95519     -6511.523       0              0            
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
change_box all     x final 0 6.311538976329234 y final 0 6.311538976329234 z final 0 49.92276679363351      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.311539 6.3124675 49.930112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.311539 6.311539 49.930112) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.311539 6.311539 49.922767) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.311539 6.311539 49.922767) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.311539 6.311539 49.922767) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.311539 6.311539 49.922767) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.95941     -5688.9985      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 80 atoms

369.3% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 6.311538976329234 y final 0 6.311538976329234 z final 0 49.92276679363351      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.311539 6.311539 49.922767) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.311539 6.311539 49.922767) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.311539 6.311539 49.922767) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.311539 6.311539 49.922767) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.311539 6.311539 49.922767) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.311539 6.311539 49.922767) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.95941     -5688.9985      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 80 atoms

70.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.416e-06  |            |       |100.00

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
change_box all     x final 0 6.309060402591013 y final 0 6.309060402591013 z final 0 49.90316186697802      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3090604 6.311539 49.922767) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3090604 6.3090604 49.922767) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3090604 6.3090604 49.903162) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3090604 6.3090604 49.903162) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3090604 6.3090604 49.903162) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3090604 6.3090604 49.903162) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.96896     -4213.6775      0              0            
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
change_box all     x final 0 6.305151002442131 y final 0 6.305151002442131 z final 0 49.872239444939936      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.305151 6.3090604 49.903162) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.305151 6.305151 49.903162) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.305151 6.305151 49.872239) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.305151 6.305151 49.872239) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.305151 6.305151 49.872239) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.305151 6.305151 49.872239) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.97468     -1866.1183      0              0            
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
change_box all     x final 0 6.3051510024421304 y final 0 6.3051510024421304 z final 0 49.872239444939936      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.305151 6.305151 49.872239) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.305151 6.305151 49.872239) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.305151 6.305151 49.872239) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.305151 6.305151 49.872239) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.305151 6.305151 49.872239) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.305151 6.305151 49.872239) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.97468     -1866.1183      0              0            
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
change_box all     x final 0 6.301442749185047 y final 0 6.301442749185048 z final 0 49.84290804680396      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3014427 6.305151 49.872239) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3014427 6.3014427 49.872239) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3014427 6.3014427 49.842908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3014427 6.3014427 49.842908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3014427 6.3014427 49.842908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.3014427 6.3014427 49.842908) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-685.99499     -896.75373      0              0            
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
change_box all     x final 0 6.297729966302651 y final 0 6.297729966302652 z final 0 49.81354082041295      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.29773 6.3014427 49.842908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.29773 6.29773 49.842908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.29773 6.29773 49.813541) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.29773 6.29773 49.813541) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.29773 6.29773 49.813541) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.29773 6.29773 49.813541) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-686.00177      97.970671      0              0            
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
change_box all     x final 0 6.297729966302651 y final 0 6.297729966302652 z final 0 49.813540820412946      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.29773 6.29773 49.813541) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.29773 6.29773 49.813541) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.29773 6.29773 49.813541) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.29773 6.29773 49.813541) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.29773 6.29773 49.813541) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.29773 6.29773 49.813541) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-686.00177      97.970671      0              0            
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
change_box all     x final 0 6.293336048485794 y final 0 6.293336048485796 z final 0 49.778785979271284      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.293336 6.29773 49.813541) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.293336 6.293336 49.813541) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.293336 6.293336 49.778786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.293336 6.293336 49.778786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.293336 6.293336 49.778786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.293336 6.293336 49.778786) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-686.02589     -68.024146      0              0            
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
change_box all     x final 0 6.293336048485793 y final 0 6.293336048485796 z final 0 49.77878597927128      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.293336 6.293336 49.778786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.293336 6.293336 49.778786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.293336 6.293336 49.778786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.293336 6.293336 49.778786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.293336 6.293336 49.778786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.293336 6.293336 49.778786) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-686.02589     -68.024146      0              0            
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
change_box all     x final 0 6.29259027176994 y final 0 6.292590271769941 z final 0 49.772887063459244      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2925903 6.293336 49.778786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925903 6.2925903 49.778786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925903 6.2925903 49.772887) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925903 6.2925903 49.772887) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925903 6.2925903 49.772887) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925903 6.2925903 49.772887) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-686.02927     -75.610371      0              0            
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
change_box all     x final 0 6.29259027176994 y final 0 6.2925902717699405 z final 0 49.77288706345924      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2925903 6.2925903 49.772887) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925903 6.2925903 49.772887) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925903 6.2925903 49.772887) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925903 6.2925903 49.772887) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925903 6.2925903 49.772887) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925903 6.2925903 49.772887) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-686.02927     -75.610371      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 6.292599739666229 y final 0 6.29259973966623 z final 0 49.77296195225898      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2925997 6.2925903 49.772887) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925997 6.2925997 49.772887) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925997 6.2925997 49.772962) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925997 6.2925997 49.772962) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925997 6.2925997 49.772962) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925997 6.2925997 49.772962) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-686.02957     -15.38577       0              0            
Loop time of 1.916e-06 on 1 procs for 0 steps with 80 atoms

261.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.916e-06  |            |       |100.00

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
change_box all     x final 0 6.2925965311594565 y final 0 6.292596531159458 z final 0 49.77293657373626      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2925965 6.2925997 49.772962) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925965 6.2925965 49.772962) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925965 6.2925965 49.772937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925965 6.2925965 49.772937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925965 6.2925965 49.772937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925965 6.2925965 49.772937) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-686.02968     -36.433906      0              0            
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
change_box all     x final 0 6.2925965311594565 y final 0 6.292596531159457 z final 0 49.77293657373625      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2925965 6.2925965 49.772937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925965 6.2925965 49.772937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925965 6.2925965 49.772937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925965 6.2925965 49.772937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925965 6.2925965 49.772937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2925965 6.2925965 49.772937) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-686.02968     -36.433906      0              0            
Loop time of 5.42e-07 on 1 procs for 0 steps with 80 atoms

369.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.42e-07   |            |       |100.00

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
change_box all     x final 0 6.292697618530733 y final 0 6.292697618530735 z final 0 49.773736150714385      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2926976 6.2925965 49.772937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2926976 6.2926976 49.772937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2926976 6.2926976 49.773736) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2926976 6.2926976 49.773736) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2926976 6.2926976 49.773736) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2926976 6.2926976 49.773736) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-686.02994      3.0137594      0              0            
Loop time of 1.583e-06 on 1 procs for 0 steps with 80 atoms

252.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.583e-06  |            |       |100.00

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
change_box all     x final 0 8.951394894652802 y final 0 7.752135378075655 z final 0 37.35836039929255      xy final 4.475697447326402 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9513949 6.2926976 49.773736) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 7.7521354 49.773736) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 7.7521354 37.35836) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9513949 7.7521354 37.35836) with tilt (4.4756974 0 0)
  triclinic box = (0 0 0) to (8.9513949 7.7521354 37.35836) with tilt (4.4756974 0 0)
  triclinic box = (0 0 0) to (8.9513949 7.7521354 37.35836) with tilt (4.4756974 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-670.95736     -11999.819      0              0            
Loop time of 1.5e-06 on 1 procs for 0 steps with 80 atoms

66.7% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 8.951394894652802 y final 0 7.752135378075655 z final 0 37.35836039929254      xy final 4.475697447326402 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9513949 7.7521354 37.35836) with tilt (4.4756974 0 0)
  triclinic box = (0 0 0) to (8.9513949 7.7521354 37.35836) with tilt (4.4756974 0 0)
  triclinic box = (0 0 0) to (8.9513949 7.7521354 37.35836) with tilt (4.4756974 0 0)
  triclinic box = (0 0 0) to (8.9513949 7.7521354 37.35836) with tilt (4.4756974 0 0)
  triclinic box = (0 0 0) to (8.9513949 7.7521354 37.35836) with tilt (4.4756974 0 0)
  triclinic box = (0 0 0) to (8.9513949 7.7521354 37.35836) with tilt (4.4756974 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-670.95736     -11999.819      0              0            
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
change_box all     x final 0 7.328782283197491 y final 0 6.3469116360543465 z final 0 30.586438543471438      xy final 3.664391141598746 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.3287823 7.7521354 37.35836) with tilt (4.4756974 0 0)
  triclinic box = (0 0 0) to (7.3287823 6.3469116 37.35836) with tilt (4.4756974 0 0)
  triclinic box = (0 0 0) to (7.3287823 6.3469116 30.586439) with tilt (4.4756974 0 0)
  triclinic box = (0 0 0) to (7.3287823 6.3469116 30.586439) with tilt (3.6643911 0 0)
  triclinic box = (0 0 0) to (7.3287823 6.3469116 30.586439) with tilt (3.6643911 0 0)
  triclinic box = (0 0 0) to (7.3287823 6.3469116 30.586439) with tilt (3.6643911 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-103.41013      2603066.2      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.923239073165128 y final 0 7.72775172140291 z final 0 37.240852978510226      xy final 4.461619536582565 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9232391 6.3469116 30.586439) with tilt (3.6643911 0 0)
  triclinic box = (0 0 0) to (8.9232391 7.7277517 30.586439) with tilt (3.6643911 0 0)
  triclinic box = (0 0 0) to (8.9232391 7.7277517 37.240853) with tilt (3.6643911 0 0)
  triclinic box = (0 0 0) to (8.9232391 7.7277517 37.240853) with tilt (4.4616195 0 0)
  triclinic box = (0 0 0) to (8.9232391 7.7277517 37.240853) with tilt (4.4616195 0 0)
  triclinic box = (0 0 0) to (8.9232391 7.7277517 37.240853) with tilt (4.4616195 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.07865      2033.2444      0              0            
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
change_box all     x final 0 8.923239073165128 y final 0 7.727751721402911 z final 0 37.240852978510226      xy final 4.461619536582565 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9232391 7.7277517 37.240853) with tilt (4.4616195 0 0)
  triclinic box = (0 0 0) to (8.9232391 7.7277517 37.240853) with tilt (4.4616195 0 0)
  triclinic box = (0 0 0) to (8.9232391 7.7277517 37.240853) with tilt (4.4616195 0 0)
  triclinic box = (0 0 0) to (8.9232391 7.7277517 37.240853) with tilt (4.4616195 0 0)
  triclinic box = (0 0 0) to (8.9232391 7.7277517 37.240853) with tilt (4.4616195 0 0)
  triclinic box = (0 0 0) to (8.9232391 7.7277517 37.240853) with tilt (4.4616195 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.07865      2033.2444      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.953985851916118 y final 0 7.754379212885832 z final 0 37.369173676593036      xy final 4.47699292595806 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9539859 7.7277517 37.240853) with tilt (4.4616195 0 0)
  triclinic box = (0 0 0) to (8.9539859 7.7543792 37.240853) with tilt (4.4616195 0 0)
  triclinic box = (0 0 0) to (8.9539859 7.7543792 37.369174) with tilt (4.4616195 0 0)
  triclinic box = (0 0 0) to (8.9539859 7.7543792 37.369174) with tilt (4.4769929 0 0)
  triclinic box = (0 0 0) to (8.9539859 7.7543792 37.369174) with tilt (4.4769929 0 0)
  triclinic box = (0 0 0) to (8.9539859 7.7543792 37.369174) with tilt (4.4769929 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-670.92367      2523.3753      0              0            
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
Nghost:            731 ave         731 max         731 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4584 ave        4584 max        4584 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4584
Ave neighs/atom = 57.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.937963737827726 y final 0 7.74050365506294 z final 0 37.30230589570335      xy final 4.468981868913864 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9379637 7.7543792 37.369174) with tilt (4.4769929 0 0)
  triclinic box = (0 0 0) to (8.9379637 7.7405037 37.369174) with tilt (4.4769929 0 0)
  triclinic box = (0 0 0) to (8.9379637 7.7405037 37.302306) with tilt (4.4769929 0 0)
  triclinic box = (0 0 0) to (8.9379637 7.7405037 37.302306) with tilt (4.4689819 0 0)
  triclinic box = (0 0 0) to (8.9379637 7.7405037 37.302306) with tilt (4.4689819 0 0)
  triclinic box = (0 0 0) to (8.9379637 7.7405037 37.302306) with tilt (4.4689819 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.84043      2181.7468      0              0            
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
change_box all     x final 0 8.937963737827726 y final 0 7.740503655062941 z final 0 37.30230589570335      xy final 4.468981868913864 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9379637 7.7405037 37.302306) with tilt (4.4689819 0 0)
  triclinic box = (0 0 0) to (8.9379637 7.7405037 37.302306) with tilt (4.4689819 0 0)
  triclinic box = (0 0 0) to (8.9379637 7.7405037 37.302306) with tilt (4.4689819 0 0)
  triclinic box = (0 0 0) to (8.9379637 7.7405037 37.302306) with tilt (4.4689819 0 0)
  triclinic box = (0 0 0) to (8.9379637 7.7405037 37.302306) with tilt (4.4689819 0 0)
  triclinic box = (0 0 0) to (8.9379637 7.7405037 37.302306) with tilt (4.4689819 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.84043      2181.7468      0              0            
Loop time of 1.458e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.458e-06  |            |       |100.00

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
change_box all     x final 0 8.943667051461835 y final 0 7.745442869555832 z final 0 37.32610849280925      xy final 4.4718335257309185 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9436671 7.7405037 37.302306) with tilt (4.4689819 0 0)
  triclinic box = (0 0 0) to (8.9436671 7.7454429 37.302306) with tilt (4.4689819 0 0)
  triclinic box = (0 0 0) to (8.9436671 7.7454429 37.326108) with tilt (4.4689819 0 0)
  triclinic box = (0 0 0) to (8.9436671 7.7454429 37.326108) with tilt (4.4718335 0 0)
  triclinic box = (0 0 0) to (8.9436671 7.7454429 37.326108) with tilt (4.4718335 0 0)
  triclinic box = (0 0 0) to (8.9436671 7.7454429 37.326108) with tilt (4.4718335 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.88264      1939.7097      0              0            
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
change_box all     x final 0 8.941243493004883 y final 0 7.743344006364553 z final 0 37.31599384907504      xy final 4.4706217465024425 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9412435 7.7454429 37.326108) with tilt (4.4718335 0 0)
  triclinic box = (0 0 0) to (8.9412435 7.743344 37.326108) with tilt (4.4718335 0 0)
  triclinic box = (0 0 0) to (8.9412435 7.743344 37.315994) with tilt (4.4718335 0 0)
  triclinic box = (0 0 0) to (8.9412435 7.743344 37.315994) with tilt (4.4706217 0 0)
  triclinic box = (0 0 0) to (8.9412435 7.743344 37.315994) with tilt (4.4706217 0 0)
  triclinic box = (0 0 0) to (8.9412435 7.743344 37.315994) with tilt (4.4706217 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.93199      1995.9043      0              0            
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
change_box all     x final 0 8.941243493004883 y final 0 7.743344006364553 z final 0 37.31599384907504      xy final 4.4706217465024425 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9412435 7.743344 37.315994) with tilt (4.4706217 0 0)
  triclinic box = (0 0 0) to (8.9412435 7.743344 37.315994) with tilt (4.4706217 0 0)
  triclinic box = (0 0 0) to (8.9412435 7.743344 37.315994) with tilt (4.4706217 0 0)
  triclinic box = (0 0 0) to (8.9412435 7.743344 37.315994) with tilt (4.4706217 0 0)
  triclinic box = (0 0 0) to (8.9412435 7.743344 37.315994) with tilt (4.4706217 0 0)
  triclinic box = (0 0 0) to (8.9412435 7.743344 37.315994) with tilt (4.4706217 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.93199      1995.9043      0              0            
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
change_box all     x final 0 8.943494974469253 y final 0 7.745293846508848 z final 0 37.32539033498234      xy final 4.4717474872346274 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.943495 7.743344 37.315994) with tilt (4.4706217 0 0)
  triclinic box = (0 0 0) to (8.943495 7.7452938 37.315994) with tilt (4.4706217 0 0)
  triclinic box = (0 0 0) to (8.943495 7.7452938 37.32539) with tilt (4.4706217 0 0)
  triclinic box = (0 0 0) to (8.943495 7.7452938 37.32539) with tilt (4.4717475 0 0)
  triclinic box = (0 0 0) to (8.943495 7.7452938 37.32539) with tilt (4.4717475 0 0)
  triclinic box = (0 0 0) to (8.943495 7.7452938 37.32539) with tilt (4.4717475 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.93506      1819.5202      0              0            
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
change_box all     x final 0 8.942471009274339 y final 0 7.744407066637461 z final 0 37.32111684897953      xy final 4.47123550463717 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.942471 7.7452938 37.32539) with tilt (4.4717475 0 0)
  triclinic box = (0 0 0) to (8.942471 7.7444071 37.32539) with tilt (4.4717475 0 0)
  triclinic box = (0 0 0) to (8.942471 7.7444071 37.321117) with tilt (4.4717475 0 0)
  triclinic box = (0 0 0) to (8.942471 7.7444071 37.321117) with tilt (4.4712355 0 0)
  triclinic box = (0 0 0) to (8.942471 7.7444071 37.321117) with tilt (4.4712355 0 0)
  triclinic box = (0 0 0) to (8.942471 7.7444071 37.321117) with tilt (4.4712355 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.94188      1894.1775      0              0            
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
change_box all     x final 0 8.942471009274339 y final 0 7.744407066637462 z final 0 37.32111684897952      xy final 4.47123550463717 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.942471 7.7444071 37.321117) with tilt (4.4712355 0 0)
  triclinic box = (0 0 0) to (8.942471 7.7444071 37.321117) with tilt (4.4712355 0 0)
  triclinic box = (0 0 0) to (8.942471 7.7444071 37.321117) with tilt (4.4712355 0 0)
  triclinic box = (0 0 0) to (8.942471 7.7444071 37.321117) with tilt (4.4712355 0 0)
  triclinic box = (0 0 0) to (8.942471 7.7444071 37.321117) with tilt (4.4712355 0 0)
  triclinic box = (0 0 0) to (8.942471 7.7444071 37.321117) with tilt (4.4712355 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.94188      1894.1775      0              0            
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
Nghost:            731 ave         731 max         731 min
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
change_box all     x final 0 8.943743166187392 y final 0 7.745508786841765 z final 0 37.326426155182816      xy final 4.471871583093697 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9437432 7.7444071 37.321117) with tilt (4.4712355 0 0)
  triclinic box = (0 0 0) to (8.9437432 7.7455088 37.321117) with tilt (4.4712355 0 0)
  triclinic box = (0 0 0) to (8.9437432 7.7455088 37.326426) with tilt (4.4712355 0 0)
  triclinic box = (0 0 0) to (8.9437432 7.7455088 37.326426) with tilt (4.4718716 0 0)
  triclinic box = (0 0 0) to (8.9437432 7.7455088 37.326426) with tilt (4.4718716 0 0)
  triclinic box = (0 0 0) to (8.9437432 7.7455088 37.326426) with tilt (4.4718716 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.94352      1760.0351      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
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
change_box all     x final 0 8.943743166187392 y final 0 7.745508786841765 z final 0 37.326426155182816      xy final 4.471871583093697 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9437432 7.7455088 37.326426) with tilt (4.4718716 0 0)
  triclinic box = (0 0 0) to (8.9437432 7.7455088 37.326426) with tilt (4.4718716 0 0)
  triclinic box = (0 0 0) to (8.9437432 7.7455088 37.326426) with tilt (4.4718716 0 0)
  triclinic box = (0 0 0) to (8.9437432 7.7455088 37.326426) with tilt (4.4718716 0 0)
  triclinic box = (0 0 0) to (8.9437432 7.7455088 37.326426) with tilt (4.4718716 0 0)
  triclinic box = (0 0 0) to (8.9437432 7.7455088 37.326426) with tilt (4.4718716 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.94352      1760.0351      0              0            
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
change_box all     x final 0 8.944708130067147 y final 0 7.746344470075368 z final 0 37.3304534010832      xy final 4.472354065033574 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9447081 7.7455088 37.326426) with tilt (4.4718716 0 0)
  triclinic box = (0 0 0) to (8.9447081 7.7463445 37.326426) with tilt (4.4718716 0 0)
  triclinic box = (0 0 0) to (8.9447081 7.7463445 37.330453) with tilt (4.4718716 0 0)
  triclinic box = (0 0 0) to (8.9447081 7.7463445 37.330453) with tilt (4.4723541 0 0)
  triclinic box = (0 0 0) to (8.9447081 7.7463445 37.330453) with tilt (4.4723541 0 0)
  triclinic box = (0 0 0) to (8.9447081 7.7463445 37.330453) with tilt (4.4723541 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.94613      1602.5153      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 80 atoms

266.7% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 8.946224039947298 y final 0 7.747657286541425 z final 0 37.336780002501406      xy final 4.47311201997365 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.946224 7.7463445 37.330453) with tilt (4.4723541 0 0)
  triclinic box = (0 0 0) to (8.946224 7.7476573 37.330453) with tilt (4.4723541 0 0)
  triclinic box = (0 0 0) to (8.946224 7.7476573 37.33678) with tilt (4.4723541 0 0)
  triclinic box = (0 0 0) to (8.946224 7.7476573 37.33678) with tilt (4.473112 0 0)
  triclinic box = (0 0 0) to (8.946224 7.7476573 37.33678) with tilt (4.473112 0 0)
  triclinic box = (0 0 0) to (8.946224 7.7476573 37.33678) with tilt (4.473112 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.94768      1357.417       0              0            
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
change_box all     x final 0 8.946224039947298 y final 0 7.7476572865414255 z final 0 37.336780002501406      xy final 4.47311201997365 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.946224 7.7476573 37.33678) with tilt (4.473112 0 0)
  triclinic box = (0 0 0) to (8.946224 7.7476573 37.33678) with tilt (4.473112 0 0)
  triclinic box = (0 0 0) to (8.946224 7.7476573 37.33678) with tilt (4.473112 0 0)
  triclinic box = (0 0 0) to (8.946224 7.7476573 37.33678) with tilt (4.473112 0 0)
  triclinic box = (0 0 0) to (8.946224 7.7476573 37.33678) with tilt (4.473112 0 0)
  triclinic box = (0 0 0) to (8.946224 7.7476573 37.33678) with tilt (4.473112 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.94768      1357.417       0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 80 atoms

369.3% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 8.94913603232882 y final 0 7.750179145919449 z final 0 37.34893310960361      xy final 4.474568016164411 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.949136 7.7476573 37.33678) with tilt (4.473112 0 0)
  triclinic box = (0 0 0) to (8.949136 7.7501791 37.33678) with tilt (4.473112 0 0)
  triclinic box = (0 0 0) to (8.949136 7.7501791 37.348933) with tilt (4.473112 0 0)
  triclinic box = (0 0 0) to (8.949136 7.7501791 37.348933) with tilt (4.474568 0 0)
  triclinic box = (0 0 0) to (8.949136 7.7501791 37.348933) with tilt (4.474568 0 0)
  triclinic box = (0 0 0) to (8.949136 7.7501791 37.348933) with tilt (4.474568 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.95215      844.62791      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 80 atoms

312.8% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 8.953761958098566 y final 0 7.754185315152072 z final 0 37.36823926290378      xy final 4.476880979049284 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.953762 7.7501791 37.348933) with tilt (4.474568 0 0)
  triclinic box = (0 0 0) to (8.953762 7.7541853 37.348933) with tilt (4.474568 0 0)
  triclinic box = (0 0 0) to (8.953762 7.7541853 37.368239) with tilt (4.474568 0 0)
  triclinic box = (0 0 0) to (8.953762 7.7541853 37.368239) with tilt (4.476881 0 0)
  triclinic box = (0 0 0) to (8.953762 7.7541853 37.368239) with tilt (4.476881 0 0)
  triclinic box = (0 0 0) to (8.953762 7.7541853 37.368239) with tilt (4.476881 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.95522      66.450693      0              0            
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
Nghost:            731 ave         731 max         731 min
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
change_box all     x final 0 8.953761958098566 y final 0 7.754185315152072 z final 0 37.36823926290378      xy final 4.476880979049284 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.953762 7.7541853 37.368239) with tilt (4.476881 0 0)
  triclinic box = (0 0 0) to (8.953762 7.7541853 37.368239) with tilt (4.476881 0 0)
  triclinic box = (0 0 0) to (8.953762 7.7541853 37.368239) with tilt (4.476881 0 0)
  triclinic box = (0 0 0) to (8.953762 7.7541853 37.368239) with tilt (4.476881 0 0)
  triclinic box = (0 0 0) to (8.953762 7.7541853 37.368239) with tilt (4.476881 0 0)
  triclinic box = (0 0 0) to (8.953762 7.7541853 37.368239) with tilt (4.476881 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.95522      66.450693      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            731 ave         731 max         731 min
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
change_box all     x final 0 8.954662980500055 y final 0 7.754965623441141 z final 0 37.37199965108895      xy final 4.4773314902500285 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.954663 7.7541853 37.368239) with tilt (4.476881 0 0)
  triclinic box = (0 0 0) to (8.954663 7.7549656 37.368239) with tilt (4.476881 0 0)
  triclinic box = (0 0 0) to (8.954663 7.7549656 37.372) with tilt (4.476881 0 0)
  triclinic box = (0 0 0) to (8.954663 7.7549656 37.372) with tilt (4.4773315 0 0)
  triclinic box = (0 0 0) to (8.954663 7.7549656 37.372) with tilt (4.4773315 0 0)
  triclinic box = (0 0 0) to (8.954663 7.7549656 37.372) with tilt (4.4773315 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.95551     -57.745417      0              0            
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
Nghost:            731 ave         731 max         731 min
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
change_box all     x final 0 8.954368096972937 y final 0 7.754710246815497 z final 0 37.370768964116515      xy final 4.477184048486469 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9543681 7.7549656 37.372) with tilt (4.4773315 0 0)
  triclinic box = (0 0 0) to (8.9543681 7.7547102 37.372) with tilt (4.4773315 0 0)
  triclinic box = (0 0 0) to (8.9543681 7.7547102 37.370769) with tilt (4.4773315 0 0)
  triclinic box = (0 0 0) to (8.9543681 7.7547102 37.370769) with tilt (4.477184 0 0)
  triclinic box = (0 0 0) to (8.9543681 7.7547102 37.370769) with tilt (4.477184 0 0)
  triclinic box = (0 0 0) to (8.9543681 7.7547102 37.370769) with tilt (4.477184 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-671.9556      -17.246723      0              0            
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
change_box all     x final 0 8.951394894652802 y final 0 10.963174989026873 z final 0 31.628203067113486      xy final -8.951394894652804 xz final 0.0 yz final -0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9513949 7.7547102 37.370769) with tilt (4.477184 0 0)
  triclinic box = (0 0 0) to (8.9513949 10.963175 37.370769) with tilt (4.477184 0 0)
  triclinic box = (0 0 0) to (8.9513949 10.963175 31.628203) with tilt (4.477184 0 0)
  triclinic box = (0 0 0) to (8.9513949 10.963175 31.628203) with tilt (-8.9513949 0 0)
  triclinic box = (0 0 0) to (8.9513949 10.963175 31.628203) with tilt (-8.9513949 0 0)
  triclinic box = (0 0 0) to (8.9513949 10.963175 31.628203) with tilt (-8.9513949 0 -0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-662.55523     -16160.808      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 7.328782283197491 y final 0 8.975888514891667 z final 0 25.894982515641157      xy final -7.328782283197492 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.3287823 10.963175 31.628203) with tilt (-8.9513949 0 -0)
  triclinic box = (0 0 0) to (7.3287823 8.9758885 31.628203) with tilt (-8.9513949 0 -0)
  triclinic box = (0 0 0) to (7.3287823 8.9758885 25.894983) with tilt (-8.9513949 0 -0)
  triclinic box = (0 0 0) to (7.3287823 8.9758885 25.894983) with tilt (-7.3287823 0 -0)
  triclinic box = (0 0 0) to (7.3287823 8.9758885 25.894983) with tilt (-7.3287823 0 -0)
  triclinic box = (0 0 0) to (7.3287823 8.9758885 25.894983) with tilt (-7.3287823 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-113.30633      2073359.2      0              0            
Loop time of 1.334e-06 on 1 procs for 0 steps with 80 atoms

149.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.334e-06  |            |       |100.00

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
change_box all     x final 0 8.91618710097918 y final 0 10.92005442429209 z final 0 31.503802427776307      xy final -8.916187100979181 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9161871 8.9758885 25.894983) with tilt (-7.3287823 0 0)
  triclinic box = (0 0 0) to (8.9161871 10.920054 25.894983) with tilt (-7.3287823 0 0)
  triclinic box = (0 0 0) to (8.9161871 10.920054 31.503802) with tilt (-7.3287823 0 0)
  triclinic box = (0 0 0) to (8.9161871 10.920054 31.503802) with tilt (-8.9161871 0 0)
  triclinic box = (0 0 0) to (8.9161871 10.920054 31.503802) with tilt (-8.9161871 0 0)
  triclinic box = (0 0 0) to (8.9161871 10.920054 31.503802) with tilt (-8.9161871 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-662.73842     -1867.684       0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.209e-06  |            |       |100.00

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
change_box all     x final 0 8.91618710097918 y final 0 10.92005442429209 z final 0 31.503802427776307      xy final -8.916187100979181 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9161871 10.920054 31.503802) with tilt (-8.9161871 0 0)
  triclinic box = (0 0 0) to (8.9161871 10.920054 31.503802) with tilt (-8.9161871 0 0)
  triclinic box = (0 0 0) to (8.9161871 10.920054 31.503802) with tilt (-8.9161871 0 0)
  triclinic box = (0 0 0) to (8.9161871 10.920054 31.503802) with tilt (-8.9161871 0 0)
  triclinic box = (0 0 0) to (8.9161871 10.920054 31.503802) with tilt (-8.9161871 0 0)
  triclinic box = (0 0 0) to (8.9161871 10.920054 31.503802) with tilt (-8.9161871 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-662.73842     -1867.684       0              0            
Loop time of 1.625e-06 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.940089659868727 y final 0 10.94932896070519 z final 0 31.588257978590388      xy final -8.940089659868729 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9400897 10.920054 31.503802) with tilt (-8.9161871 0 0)
  triclinic box = (0 0 0) to (8.9400897 10.949329 31.503802) with tilt (-8.9161871 0 0)
  triclinic box = (0 0 0) to (8.9400897 10.949329 31.588258) with tilt (-8.9161871 0 0)
  triclinic box = (0 0 0) to (8.9400897 10.949329 31.588258) with tilt (-8.9400897 0 0)
  triclinic box = (0 0 0) to (8.9400897 10.949329 31.588258) with tilt (-8.9400897 0 0)
  triclinic box = (0 0 0) to (8.9400897 10.949329 31.588258) with tilt (-8.9400897 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.30613     -1536.9032      0              0            
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
Nghost:            574 ave         574 max         574 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4208 ave        4208 max        4208 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4208
Ave neighs/atom = 52.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.93124623349912 y final 0 10.938498019613489 z final 0 31.55701126360141      xy final -8.931246233499122 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9312462 10.949329 31.588258) with tilt (-8.9400897 0 0)
  triclinic box = (0 0 0) to (8.9312462 10.938498 31.588258) with tilt (-8.9400897 0 0)
  triclinic box = (0 0 0) to (8.9312462 10.938498 31.557011) with tilt (-8.9400897 0 0)
  triclinic box = (0 0 0) to (8.9312462 10.938498 31.557011) with tilt (-8.9312462 0 0)
  triclinic box = (0 0 0) to (8.9312462 10.938498 31.557011) with tilt (-8.9312462 0 0)
  triclinic box = (0 0 0) to (8.9312462 10.938498 31.557011) with tilt (-8.9312462 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.57899     -1536.9909      0              0            
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
change_box all     x final 0 8.93124623349912 y final 0 10.938498019613489 z final 0 31.557011263601407      xy final -8.931246233499122 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9312462 10.938498 31.557011) with tilt (-8.9312462 0 0)
  triclinic box = (0 0 0) to (8.9312462 10.938498 31.557011) with tilt (-8.9312462 0 0)
  triclinic box = (0 0 0) to (8.9312462 10.938498 31.557011) with tilt (-8.9312462 0 0)
  triclinic box = (0 0 0) to (8.9312462 10.938498 31.557011) with tilt (-8.9312462 0 0)
  triclinic box = (0 0 0) to (8.9312462 10.938498 31.557011) with tilt (-8.9312462 0 0)
  triclinic box = (0 0 0) to (8.9312462 10.938498 31.557011) with tilt (-8.9312462 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.57899     -1536.9909      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 80 atoms

384.2% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 8.931247447686236 y final 0 10.938499506682927 z final 0 31.55701555372122      xy final -8.931247447686237 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9312474 10.938498 31.557011) with tilt (-8.9312462 0 0)
  triclinic box = (0 0 0) to (8.9312474 10.9385 31.557011) with tilt (-8.9312462 0 0)
  triclinic box = (0 0 0) to (8.9312474 10.9385 31.557016) with tilt (-8.9312462 0 0)
  triclinic box = (0 0 0) to (8.9312474 10.9385 31.557016) with tilt (-8.9312474 0 0)
  triclinic box = (0 0 0) to (8.9312474 10.9385 31.557016) with tilt (-8.9312474 0 0)
  triclinic box = (0 0 0) to (8.9312474 10.9385 31.557016) with tilt (-8.9312474 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.60836     -1737.9914      0              0            
Loop time of 7.09e-07 on 1 procs for 0 steps with 80 atoms

141.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.09e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4272 ave        4272 max        4272 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4272
Ave neighs/atom = 53.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.931246869356048 y final 0 10.938498798375997 z final 0 31.5570135102917      xy final -8.93124686935605 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9312469 10.9385 31.557016) with tilt (-8.9312474 0 0)
  triclinic box = (0 0 0) to (8.9312469 10.938499 31.557016) with tilt (-8.9312474 0 0)
  triclinic box = (0 0 0) to (8.9312469 10.938499 31.557014) with tilt (-8.9312474 0 0)
  triclinic box = (0 0 0) to (8.9312469 10.938499 31.557014) with tilt (-8.9312469 0 0)
  triclinic box = (0 0 0) to (8.9312469 10.938499 31.557014) with tilt (-8.9312469 0 0)
  triclinic box = (0 0 0) to (8.9312469 10.938499 31.557014) with tilt (-8.9312469 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.74214     -1718.8186      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            574 ave         574 max         574 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4208 ave        4208 max        4208 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4208
Ave neighs/atom = 52.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.929530637014508 y final 0 10.936396851617577 z final 0 31.550949500643213      xy final -8.92953063701451 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9295306 10.938499 31.557014) with tilt (-8.9312469 0 0)
  triclinic box = (0 0 0) to (8.9295306 10.936397 31.557014) with tilt (-8.9312469 0 0)
  triclinic box = (0 0 0) to (8.9295306 10.936397 31.55095) with tilt (-8.9312469 0 0)
  triclinic box = (0 0 0) to (8.9295306 10.936397 31.55095) with tilt (-8.9295306 0 0)
  triclinic box = (0 0 0) to (8.9295306 10.936397 31.55095) with tilt (-8.9295306 0 0)
  triclinic box = (0 0 0) to (8.9295306 10.936397 31.55095) with tilt (-8.9295306 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.77226     -1586.2885      0              0            
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
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4240 ave        4240 max        4240 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4240
Ave neighs/atom = 53
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.929530637014508 y final 0 10.936396851617577 z final 0 31.550949500643213      xy final -8.92953063701451 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9295306 10.936397 31.55095) with tilt (-8.9295306 0 0)
  triclinic box = (0 0 0) to (8.9295306 10.936397 31.55095) with tilt (-8.9295306 0 0)
  triclinic box = (0 0 0) to (8.9295306 10.936397 31.55095) with tilt (-8.9295306 0 0)
  triclinic box = (0 0 0) to (8.9295306 10.936397 31.55095) with tilt (-8.9295306 0 0)
  triclinic box = (0 0 0) to (8.9295306 10.936397 31.55095) with tilt (-8.9295306 0 0)
  triclinic box = (0 0 0) to (8.9295306 10.936397 31.55095) with tilt (-8.9295306 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.77226     -1586.2885      0              0            
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
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4240 ave        4240 max        4240 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4240
Ave neighs/atom = 53
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.930263280194097 y final 0 10.937294152594335 z final 0 31.553538168390805      xy final -8.930263280194099 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9302633 10.936397 31.55095) with tilt (-8.9295306 0 0)
  triclinic box = (0 0 0) to (8.9302633 10.937294 31.55095) with tilt (-8.9295306 0 0)
  triclinic box = (0 0 0) to (8.9302633 10.937294 31.553538) with tilt (-8.9295306 0 0)
  triclinic box = (0 0 0) to (8.9302633 10.937294 31.553538) with tilt (-8.9302633 0 0)
  triclinic box = (0 0 0) to (8.9302633 10.937294 31.553538) with tilt (-8.9302633 0 0)
  triclinic box = (0 0 0) to (8.9302633 10.937294 31.553538) with tilt (-8.9302633 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.78527     -1545.5301      0              0            
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
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4240 ave        4240 max        4240 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4240
Ave neighs/atom = 53
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.930263280194097 y final 0 10.937294152594335 z final 0 31.553538168390805      xy final -8.930263280194099 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9302633 10.937294 31.553538) with tilt (-8.9302633 0 0)
  triclinic box = (0 0 0) to (8.9302633 10.937294 31.553538) with tilt (-8.9302633 0 0)
  triclinic box = (0 0 0) to (8.9302633 10.937294 31.553538) with tilt (-8.9302633 0 0)
  triclinic box = (0 0 0) to (8.9302633 10.937294 31.553538) with tilt (-8.9302633 0 0)
  triclinic box = (0 0 0) to (8.9302633 10.937294 31.553538) with tilt (-8.9302633 0 0)
  triclinic box = (0 0 0) to (8.9302633 10.937294 31.553538) with tilt (-8.9302633 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.78527     -1545.5301      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4240 ave        4240 max        4240 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4240
Ave neighs/atom = 53
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.931121941854123 y final 0 10.938345794058712 z final 0 31.556572100606825      xy final -8.931121941854125 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9311219 10.937294 31.553538) with tilt (-8.9302633 0 0)
  triclinic box = (0 0 0) to (8.9311219 10.938346 31.553538) with tilt (-8.9302633 0 0)
  triclinic box = (0 0 0) to (8.9311219 10.938346 31.556572) with tilt (-8.9302633 0 0)
  triclinic box = (0 0 0) to (8.9311219 10.938346 31.556572) with tilt (-8.9311219 0 0)
  triclinic box = (0 0 0) to (8.9311219 10.938346 31.556572) with tilt (-8.9311219 0 0)
  triclinic box = (0 0 0) to (8.9311219 10.938346 31.556572) with tilt (-8.9311219 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.78798     -1698.7922      0              0            
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
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4272 ave        4272 max        4272 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4272
Ave neighs/atom = 53.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.932552894851352 y final 0 10.940098346403268 z final 0 31.561628125115682      xy final -8.932552894851353 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9325529 10.938346 31.556572) with tilt (-8.9311219 0 0)
  triclinic box = (0 0 0) to (8.9325529 10.940098 31.556572) with tilt (-8.9311219 0 0)
  triclinic box = (0 0 0) to (8.9325529 10.940098 31.561628) with tilt (-8.9311219 0 0)
  triclinic box = (0 0 0) to (8.9325529 10.940098 31.561628) with tilt (-8.9325529 0 0)
  triclinic box = (0 0 0) to (8.9325529 10.940098 31.561628) with tilt (-8.9325529 0 0)
  triclinic box = (0 0 0) to (8.9325529 10.940098 31.561628) with tilt (-8.9325529 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.78971     -1952.9033      0              0            
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
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4272 ave        4272 max        4272 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4272
Ave neighs/atom = 53.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.932552894851352 y final 0 10.940098346403268 z final 0 31.561628125115682      xy final -8.932552894851353 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9325529 10.940098 31.561628) with tilt (-8.9325529 0 0)
  triclinic box = (0 0 0) to (8.9325529 10.940098 31.561628) with tilt (-8.9325529 0 0)
  triclinic box = (0 0 0) to (8.9325529 10.940098 31.561628) with tilt (-8.9325529 0 0)
  triclinic box = (0 0 0) to (8.9325529 10.940098 31.561628) with tilt (-8.9325529 0 0)
  triclinic box = (0 0 0) to (8.9325529 10.940098 31.561628) with tilt (-8.9325529 0 0)
  triclinic box = (0 0 0) to (8.9325529 10.940098 31.561628) with tilt (-8.9325529 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.78971     -1952.9033      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4272 ave        4272 max        4272 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4272
Ave neighs/atom = 53.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.93368816293828 y final 0 10.941488760170387 z final 0 31.565639398220213      xy final -8.933688162938282 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9336882 10.940098 31.561628) with tilt (-8.9325529 0 0)
  triclinic box = (0 0 0) to (8.9336882 10.941489 31.561628) with tilt (-8.9325529 0 0)
  triclinic box = (0 0 0) to (8.9336882 10.941489 31.565639) with tilt (-8.9325529 0 0)
  triclinic box = (0 0 0) to (8.9336882 10.941489 31.565639) with tilt (-8.9336882 0 0)
  triclinic box = (0 0 0) to (8.9336882 10.941489 31.565639) with tilt (-8.9336882 0 0)
  triclinic box = (0 0 0) to (8.9336882 10.941489 31.565639) with tilt (-8.9336882 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.7939      -2132.7879      0              0            
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
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4272 ave        4272 max        4272 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4272
Ave neighs/atom = 53.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.93368816293828 y final 0 10.941488760170387 z final 0 31.565639398220213      xy final -8.933688162938282 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9336882 10.941489 31.565639) with tilt (-8.9336882 0 0)
  triclinic box = (0 0 0) to (8.9336882 10.941489 31.565639) with tilt (-8.9336882 0 0)
  triclinic box = (0 0 0) to (8.9336882 10.941489 31.565639) with tilt (-8.9336882 0 0)
  triclinic box = (0 0 0) to (8.9336882 10.941489 31.565639) with tilt (-8.9336882 0 0)
  triclinic box = (0 0 0) to (8.9336882 10.941489 31.565639) with tilt (-8.9336882 0 0)
  triclinic box = (0 0 0) to (8.9336882 10.941489 31.565639) with tilt (-8.9336882 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.7939      -2132.7879      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

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
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4272 ave        4272 max        4272 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4272
Ave neighs/atom = 53.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.935377463654863 y final 0 10.943557722559259 z final 0 31.571608249637702      xy final -8.935377463654865 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9353775 10.941489 31.565639) with tilt (-8.9336882 0 0)
  triclinic box = (0 0 0) to (8.9353775 10.943558 31.565639) with tilt (-8.9336882 0 0)
  triclinic box = (0 0 0) to (8.9353775 10.943558 31.571608) with tilt (-8.9336882 0 0)
  triclinic box = (0 0 0) to (8.9353775 10.943558 31.571608) with tilt (-8.9353775 0 0)
  triclinic box = (0 0 0) to (8.9353775 10.943558 31.571608) with tilt (-8.9353775 0 0)
  triclinic box = (0 0 0) to (8.9353775 10.943558 31.571608) with tilt (-8.9353775 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.79688     -2303.2048      0              0            
Loop time of 1.792e-06 on 1 procs for 0 steps with 80 atoms

167.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.792e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4272 ave        4272 max        4272 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4272
Ave neighs/atom = 53.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.935377463654863 y final 0 10.943557722559259 z final 0 31.571608249637702      xy final -8.935377463654865 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9353775 10.943558 31.571608) with tilt (-8.9353775 0 0)
  triclinic box = (0 0 0) to (8.9353775 10.943558 31.571608) with tilt (-8.9353775 0 0)
  triclinic box = (0 0 0) to (8.9353775 10.943558 31.571608) with tilt (-8.9353775 0 0)
  triclinic box = (0 0 0) to (8.9353775 10.943558 31.571608) with tilt (-8.9353775 0 0)
  triclinic box = (0 0 0) to (8.9353775 10.943558 31.571608) with tilt (-8.9353775 0 0)
  triclinic box = (0 0 0) to (8.9353775 10.943558 31.571608) with tilt (-8.9353775 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.79688     -2303.2048      0              0            
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
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4272 ave        4272 max        4272 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4272
Ave neighs/atom = 53.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.935030158702777 y final 0 10.94313236260038 z final 0 31.570381107758685      xy final -8.935030158702778 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9350302 10.943558 31.571608) with tilt (-8.9353775 0 0)
  triclinic box = (0 0 0) to (8.9350302 10.943132 31.571608) with tilt (-8.9353775 0 0)
  triclinic box = (0 0 0) to (8.9350302 10.943132 31.570381) with tilt (-8.9353775 0 0)
  triclinic box = (0 0 0) to (8.9350302 10.943132 31.570381) with tilt (-8.9350302 0 0)
  triclinic box = (0 0 0) to (8.9350302 10.943132 31.570381) with tilt (-8.9350302 0 0)
  triclinic box = (0 0 0) to (8.9350302 10.943132 31.570381) with tilt (-8.9350302 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-663.79705     -2229.0629      0              0            
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
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4272 ave        4272 max        4272 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4272
Ave neighs/atom = 53.4
Neighbor list builds = 0
Dangerous builds = 0
Total wall time: 0:00:09
