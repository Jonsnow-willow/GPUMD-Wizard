LAMMPS (23 Jun 2022 - Update 4)
units metal
atom_style atomic
atom_modify map array sort 0 0
boundary p p p
box tilt large
region cell prism    0 3.631 0 3.631 0 3.631     0.0 0.0 0.0     units box
create_box 1 cell
Created triclinic box = (0 0 0) to (3.631 3.631 3.631) with tilt (0 0 0)
  1 by 1 by 1 MPI processor grid
echo none
Created 4 atoms
  using lattice units in triclinic box = (0 0 0) to (3.631 3.631 3.631) with tilt (0 0 0)
  create_atoms CPU = 0.000 seconds
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
pair_style eam/alloy
pair_coeff * * potentials/Cu.eam.alloy Cu
Reading eam/alloy potential file potentials/Cu.eam.alloy with DATE: 2018-03-30
mass 1 63.545999983653153719842521240935
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
  master list distance cutoff = 7.715752
  ghost atom cutoff = 7.715752
  binsize = 3.857876, bins = 1 1 1
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair eam/alloy, perpetual
      attributes: half, newton on
      pair build: half/bin/newton/tri
      stencil: half/bin/3d/tri
      bin: standard
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-14.156391     -17751.076      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            352 ave         352 max         352 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 352
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.443434489050725 y final 0 3.443434489050725 z final 0 3.443434489050725      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.4434345 3.631 3.631) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4434345 3.4434345 3.631) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4434345 3.4434345 3.4434345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4434345 3.4434345 3.4434345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4434345 3.4434345 3.4434345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4434345 3.4434345 3.4434345) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.690664      275289.86      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            400 ave         400 max         400 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 400
Ave neighs/atom = 100
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.6148818759761285 y final 0 3.6148818759761285 z final 0 3.6148818759761285      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.6148819 3.4434345 3.4434345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6148819 3.6148819 3.4434345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6148819 3.6148819 3.6148819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6148819 3.6148819 3.6148819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6148819 3.6148819 3.6148819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6148819 3.6148819 3.6148819) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-14.159979      89.665773      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 4 atoms

505.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.92e-07   |            |       |100.00

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            352 ave         352 max         352 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 352
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.253393688378516 y final 0 3.253393688378516 z final 0 3.253393688378516      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2533937 3.6148819 3.6148819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2533937 3.2533937 3.6148819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2533937 3.2533937 3.2533937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2533937 3.2533937 3.2533937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2533937 3.2533937 3.2533937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2533937 3.2533937 3.2533937) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-11.556158      851681.59      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            448 ave         448 max         448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 448
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.289542507138277 y final 0 3.289542507138277 z final 0 3.289542507138277      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2895425 3.2533937 3.2533937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2895425 3.2895425 3.2533937) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2895425 3.2895425 3.2895425) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2895425 3.2895425 3.2895425) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2895425 3.2895425 3.2895425) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2895425 3.2895425 3.2895425) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-12.124409      719753         0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            448 ave         448 max         448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 448
Ave neighs/atom = 112
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.3256913258980383 y final 0 3.3256913258980383 z final 0 3.3256913258980383      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3256913 3.2895425 3.2895425) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3256913 3.3256913 3.2895425) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3256913 3.3256913 3.3256913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3256913 3.3256913 3.3256913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3256913 3.3256913 3.3256913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3256913 3.3256913 3.3256913) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-12.612012      599407.45      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            400 ave         400 max         400 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 400
Ave neighs/atom = 100
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.3618401446578 y final 0 3.3618401446578 z final 0 3.3618401446578      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3618401 3.3256913 3.3256913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3618401 3.3618401 3.3256913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3618401 3.3618401 3.3618401) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3618401 3.3618401 3.3618401) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3618401 3.3618401 3.3618401) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3618401 3.3618401 3.3618401) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.023163      489157.95      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            400 ave         400 max         400 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 400
Ave neighs/atom = 100
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.397988963417561 y final 0 3.397988963417561 z final 0 3.397988963417561      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.397989 3.3618401 3.3618401) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.397989 3.397989 3.3618401) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.397989 3.397989 3.397989) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.397989 3.397989 3.397989) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.397989 3.397989 3.397989) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.397989 3.397989 3.397989) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.36172       388445.24      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            400 ave         400 max         400 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 400
Ave neighs/atom = 100
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.4341377821773222 y final 0 3.4341377821773222 z final 0 3.4341377821773222      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.4341378 3.397989 3.397989) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4341378 3.4341378 3.397989) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4341378 3.4341378 3.4341378) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4341378 3.4341378 3.4341378) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4341378 3.4341378 3.4341378) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4341378 3.4341378 3.4341378) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.631754      297200.8       0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            400 ave         400 max         400 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 400
Ave neighs/atom = 100
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.470286600937084 y final 0 3.470286600937084 z final 0 3.470286600937084      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.4702866 3.4341378 3.4341378) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4702866 3.4702866 3.4341378) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4702866 3.4702866 3.4702866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4702866 3.4702866 3.4702866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4702866 3.4702866 3.4702866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4702866 3.4702866 3.4702866) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.837838      215682.65      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            352 ave         352 max         352 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 352
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.506435419696845 y final 0 3.506435419696845 z final 0 3.506435419696845      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.5064354 3.4702866 3.4702866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5064354 3.5064354 3.4702866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5064354 3.5064354 3.5064354) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5064354 3.5064354 3.5064354) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5064354 3.5064354 3.5064354) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5064354 3.5064354 3.5064354) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.985321      144453.46      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            352 ave         352 max         352 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 352
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.542584238456606 y final 0 3.542584238456606 z final 0 3.542584238456606      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.5425842 3.5064354 3.5064354) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5425842 3.5425842 3.5064354) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5425842 3.5425842 3.5425842) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5425842 3.5425842 3.5425842) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5425842 3.5425842 3.5425842) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5425842 3.5425842 3.5425842) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-14.083026      91595.106      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            352 ave         352 max         352 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 352
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.578733057216368 y final 0 3.578733057216368 z final 0 3.578733057216368      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.5787331 3.5425842 3.5425842) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5787331 3.5787331 3.5425842) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5787331 3.5787331 3.5787331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5787331 3.5787331 3.5787331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5787331 3.5787331 3.5787331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.5787331 3.5787331 3.5787331) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-14.141049      44106.624      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            352 ave         352 max         352 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 352
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.6148818759761285 y final 0 3.6148818759761285 z final 0 3.6148818759761285      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.6148819 3.5787331 3.5787331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6148819 3.6148819 3.5787331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6148819 3.6148819 3.6148819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6148819 3.6148819 3.6148819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6148819 3.6148819 3.6148819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6148819 3.6148819 3.6148819) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-14.159979      89.665773      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            352 ave         352 max         352 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 352
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.6510306947358906 y final 0 3.6510306947358906 z final 0 3.6510306947358906      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.6510307 3.6148819 3.6148819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6510307 3.6510307 3.6148819) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6510307 3.6510307 3.6510307) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6510307 3.6510307 3.6510307) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6510307 3.6510307 3.6510307) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6510307 3.6510307 3.6510307) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-14.142392     -38185.124      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            280 ave         280 max         280 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 280
Ave neighs/atom = 70
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.6871795134956513 y final 0 3.6871795134956513 z final 0 3.6871795134956513      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.6871795 3.6510307 3.6510307) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6871795 3.6871795 3.6510307) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6871795 3.6871795 3.6871795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6871795 3.6871795 3.6871795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6871795 3.6871795 3.6871795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.6871795 3.6871795 3.6871795) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-14.092556     -70036.376      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            280 ave         280 max         280 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 280
Ave neighs/atom = 70
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.7233283322554134 y final 0 3.7233283322554134 z final 0 3.7233283322554134      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.7233283 3.6871795 3.6871795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.7233283 3.7233283 3.6871795) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.7233283 3.7233283 3.7233283) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.7233283 3.7233283 3.7233283) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.7233283 3.7233283 3.7233283) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.7233283 3.7233283 3.7233283) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-14.014929     -96090.39       0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            280 ave         280 max         280 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 280
Ave neighs/atom = 70
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.7594771510151737 y final 0 3.7594771510151737 z final 0 3.7594771510151737      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.7594772 3.7233283 3.7233283) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.7594772 3.7594772 3.7233283) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.7594772 3.7594772 3.7594772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.7594772 3.7594772 3.7594772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.7594772 3.7594772 3.7594772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.7594772 3.7594772 3.7594772) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.913442     -117253.11      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            280 ave         280 max         280 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 280
Ave neighs/atom = 70
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.7956259697749357 y final 0 3.7956259697749357 z final 0 3.7956259697749357      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.795626 3.7594772 3.7594772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.795626 3.795626 3.7594772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.795626 3.795626 3.795626) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.795626 3.795626 3.795626) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.795626 3.795626 3.795626) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.795626 3.795626 3.795626) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.791589     -134498.99      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            280 ave         280 max         280 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 280
Ave neighs/atom = 70
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.8317747885346964 y final 0 3.8317747885346964 z final 0 3.8317747885346964      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.8317748 3.795626 3.795626) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.8317748 3.8317748 3.795626) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.8317748 3.8317748 3.8317748) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.8317748 3.8317748 3.8317748) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.8317748 3.8317748 3.8317748) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.8317748 3.8317748 3.8317748) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.245 | 3.245 | 3.245 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.65169      -149359.67      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            662 ave         662 max         662 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            280 ave         280 max         280 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 280
Ave neighs/atom = 70
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.8679236072944585 y final 0 3.8679236072944585 z final 0 3.8679236072944585      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.8679236 3.8317748 3.8317748) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.8679236 3.8679236 3.8317748) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.8679236 3.8679236 3.8679236) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.8679236 3.8679236 3.8679236) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.8679236 3.8679236 3.8679236) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.8679236 3.8679236 3.8679236) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.221 | 3.221 | 3.221 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.495032     -162694.83      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            360 ave         360 max         360 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            268 ave         268 max         268 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 268
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.9040724260542192 y final 0 3.9040724260542192 z final 0 3.9040724260542192      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.9040724 3.8679236 3.8679236) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9040724 3.9040724 3.8679236) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9040724 3.9040724 3.9040724) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9040724 3.9040724 3.9040724) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9040724 3.9040724 3.9040724) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9040724 3.9040724 3.9040724) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.221 | 3.221 | 3.221 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.322413     -174836.9       0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            360 ave         360 max         360 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            268 ave         268 max         268 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 268
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.9402212448139813 y final 0 3.9402212448139813 z final 0 3.9402212448139813      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.9402212 3.9040724 3.9040724) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9402212 3.9402212 3.9040724) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9402212 3.9402212 3.9402212) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9402212 3.9402212 3.9402212) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9402212 3.9402212 3.9402212) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9402212 3.9402212 3.9402212) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.221 | 3.221 | 3.221 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.134525     -185818.67      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            360 ave         360 max         360 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            268 ave         268 max         268 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 268
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 3.9763700635737416 y final 0 3.9763700635737416 z final 0 3.9763700635737416      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.9763701 3.9402212 3.9402212) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9763701 3.9763701 3.9402212) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9763701 3.9763701 3.9763701) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9763701 3.9763701 3.9763701) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9763701 3.9763701 3.9763701) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.9763701 3.9763701 3.9763701) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.221 | 3.221 | 3.221 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-12.932164     -195558.05      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 4 atoms

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

Nlocal:              4 ave           4 max           4 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            360 ave         360 max         360 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            268 ave         268 max         268 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 268
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  4 settings made for x
  4 settings made for y
  4 settings made for z
change_box all     x final 0 10.844645627928386 y final 0 10.844645627928386 z final 0 10.844645627928386      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.844646 3.9763701 3.9763701) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844646 3.9763701) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
echo none
Created 104 atoms
  using lattice units in triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
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
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-382.31944      89.665773      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 108 atoms

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

Nlocal:            108 ave         108 max         108 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1580 ave        1580 max        1580 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9504 ave        9504 max        9504 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9504
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  108 settings made for x
  108 settings made for y
  108 settings made for z
change_box all     x final 0 10.627752715369818 y final 0 10.844645627928386 z final 0 10.844645627928386      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.627753 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.627753 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.627753 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.627753 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.627753 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.627753 10.844646 10.844646) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-382.02286      38707.872      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 108 atoms

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

Nlocal:            108 ave         108 max         108 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1580 ave        1580 max        1580 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9504 ave        9504 max        9504 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9504
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  108 settings made for x
  108 settings made for y
  108 settings made for z
change_box all     x final 0 10.736199171649103 y final 0 10.844645627928386 z final 0 10.844645627928386      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.736199 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.736199 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.736199 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.736199 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.736199 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.736199 10.844646 10.844646) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-382.24804      18241.958      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 108 atoms

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

Nlocal:            108 ave         108 max         108 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1580 ave        1580 max        1580 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9504 ave        9504 max        9504 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9504
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  108 settings made for x
  108 settings made for y
  108 settings made for z
change_box all     x final 0 10.844645627928386 y final 0 10.844645627928386 z final 0 10.844645627928386      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-382.31944      89.665773      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 108 atoms

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

Nlocal:            108 ave         108 max         108 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1580 ave        1580 max        1580 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9504 ave        9504 max        9504 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9504
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  108 settings made for x
  108 settings made for y
  108 settings made for z
change_box all     x final 0 10.95309208420767 y final 0 10.844645627928386 z final 0 10.844645627928386      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.953092 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.953092 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.953092 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.953092 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.953092 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.953092 10.844646 10.844646) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-382.25498     -15701.193      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 108 atoms

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

Nlocal:            108 ave         108 max         108 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1580 ave        1580 max        1580 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9072 ave        9072 max        9072 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9072
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  108 settings made for x
  108 settings made for y
  108 settings made for z
change_box all     x final 0 11.061538540486955 y final 0 10.844645627928386 z final 0 10.844645627928386      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.061539 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.061539 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.061539 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.061539 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.061539 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.061539 10.844646 10.844646) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-382.07445     -29309.094      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 108 atoms

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

Nlocal:            108 ave         108 max         108 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1580 ave        1580 max        1580 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           8640 ave        8640 max        8640 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 8640
Ave neighs/atom = 80
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  108 settings made for x
  108 settings made for y
  108 settings made for z
change_box all     x final 0 10.844645627928386 y final 0 10.844667317197953 z final 0 10.844623938702199      xy final 0.0 xz final 0.0 yz final 0.021689247877404397 units box
Changing box ...
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844667 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844667 10.844624) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844667 10.844624) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844667 10.844624) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.844646 10.844667 10.844624) with tilt (0 0 0.021689248)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-382.31802      93.815005      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 108 atoms

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

Nlocal:            108 ave         108 max         108 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1580 ave        1580 max        1580 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9504 ave        9504 max        9504 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9504
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  108 settings made for x
  108 settings made for y
  108 settings made for z
change_box all     x final 0 10.844645627928386 y final 0 10.844874718647532 z final 0 10.844416542048629      xy final 0.0 xz final 0.0 yz final 0.0704887075233161 units box
Changing box ...
  triclinic box = (0 0 0) to (10.844646 10.844667 10.844624) with tilt (0 0 0.021689248)
  triclinic box = (0 0 0) to (10.844646 10.844875 10.844624) with tilt (0 0 0.021689248)
  triclinic box = (0 0 0) to (10.844646 10.844875 10.844417) with tilt (0 0 0.021689248)
  triclinic box = (0 0 0) to (10.844646 10.844875 10.844417) with tilt (0 0 0.021689248)
  triclinic box = (0 0 0) to (10.844646 10.844875 10.844417) with tilt (0 0 0.021689248)
  triclinic box = (0 0 0) to (10.844646 10.844875 10.844417) with tilt (0 0 0.070488708)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-382.30649      134.80861      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 108 atoms

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

Nlocal:            108 ave         108 max         108 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1580 ave        1580 max        1580 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9504 ave        9504 max        9504 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9504
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  108 settings made for x
  108 settings made for y
  108 settings made for z
change_box all     x final 0 10.844645627928386 y final 0 10.84530170914302 z final 0 10.843989586403064      xy final 0.0 xz final 0.0 yz final 0.11928388545043371 units box
Changing box ...
  triclinic box = (0 0 0) to (10.844646 10.844875 10.844417) with tilt (0 0 0.070488708)
  triclinic box = (0 0 0) to (10.844646 10.845302 10.844417) with tilt (0 0 0.070488708)
  triclinic box = (0 0 0) to (10.844646 10.845302 10.84399) with tilt (0 0 0.070488708)
  triclinic box = (0 0 0) to (10.844646 10.845302 10.84399) with tilt (0 0 0.070488708)
  triclinic box = (0 0 0) to (10.844646 10.845302 10.84399) with tilt (0 0 0.070488708)
  triclinic box = (0 0 0) to (10.844646 10.845302 10.84399) with tilt (0 0 0.11928389)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-382.28253      219.90718      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 108 atoms

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

Nlocal:            108 ave         108 max         108 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1580 ave        1580 max        1580 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9504 ave        9504 max        9504 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9504
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  108 settings made for x
  108 settings made for y
  108 settings made for z
change_box all     x final 0 10.844645627928386 y final 0 10.845948262749637 z final 0 10.843343149557956      xy final 0.0 xz final 0.0 yz final 0.1680718188181483 units box
Changing box ...
  triclinic box = (0 0 0) to (10.844646 10.845302 10.84399) with tilt (0 0 0.11928389)
  triclinic box = (0 0 0) to (10.844646 10.845948 10.84399) with tilt (0 0 0.11928389)
  triclinic box = (0 0 0) to (10.844646 10.845948 10.843343) with tilt (0 0 0.11928389)
  triclinic box = (0 0 0) to (10.844646 10.845948 10.843343) with tilt (0 0 0.11928389)
  triclinic box = (0 0 0) to (10.844646 10.845948 10.843343) with tilt (0 0 0.11928389)
  triclinic box = (0 0 0) to (10.844646 10.845948 10.843343) with tilt (0 0 0.16807182)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-382.24658      346.6195       0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 108 atoms

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

Nlocal:            108 ave         108 max         108 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1580 ave        1580 max        1580 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9396 ave        9396 max        9396 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9396
Ave neighs/atom = 87
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  108 settings made for x
  108 settings made for y
  108 settings made for z
change_box all     x final 0 10.844645627928386 y final 0 10.846814340204428 z final 0 10.84247734926472      xy final 0.0 xz final 0.0 yz final 0.21684954698529443 units box
Changing box ...
  triclinic box = (0 0 0) to (10.844646 10.845948 10.843343) with tilt (0 0 0.16807182)
  triclinic box = (0 0 0) to (10.844646 10.846814 10.843343) with tilt (0 0 0.16807182)
  triclinic box = (0 0 0) to (10.844646 10.846814 10.842477) with tilt (0 0 0.16807182)
  triclinic box = (0 0 0) to (10.844646 10.846814 10.842477) with tilt (0 0 0.16807182)
  triclinic box = (0 0 0) to (10.844646 10.846814 10.842477) with tilt (0 0 0.16807182)
  triclinic box = (0 0 0) to (10.844646 10.846814 10.842477) with tilt (0 0 0.21684955)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-382.19843      515.09603      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 108 atoms

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

Nlocal:            108 ave         108 max         108 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1580 ave        1580 max        1580 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9396 ave        9396 max        9396 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9396
Ave neighs/atom = 87
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  108 settings made for x
  108 settings made for y
  108 settings made for z
change_box all     x final 0 10.844645627928386 y final 0 10.844645627928386 z final 0 10.844645627928386      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.844646 10.846814 10.842477) with tilt (0 0 0.21684955)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.842477) with tilt (0 0 0.21684955)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0.21684955)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0.21684955)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0.21684955)
  triclinic box = (0 0 0) to (10.844646 10.844646 10.844646) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-382.31904      92.90063       0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 108 atoms

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

Nlocal:            108 ave         108 max         108 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1580 ave        1580 max        1580 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9504 ave        9504 max        9504 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9504
Ave neighs/atom = 88
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  108 settings made for x
  108 settings made for y
  108 settings made for z
change_box all     x final 0 14.459527503904514 y final 0 18.074409379880642 z final 0 21.689291255856773      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.459528 10.844646 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 10.844646) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
echo none
Created 371 atoms
  using lattice units in triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
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
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
Setting atom values ...
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
Per MPI rank memory allocation (min/avg/max) = 3.296 | 3.296 | 3.296 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.2608     -625.57714      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42064 ave       42064 max       42064 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42064
Ave neighs/atom = 87.816284
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 11.838459842423534 y final 0 14.798074803029417 z final 0 17.7576897636353      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.83846 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-339.83164      2837660.3      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           4101 ave        4101 max        4101 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          76480 ave       76480 max       76480 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 76480
Ave neighs/atom = 159.66597
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.457344483859142 y final 0 18.07168060482393 z final 0 21.686016725788715      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.457344 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.457344 18.071681 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.457344 18.071681 21.686017) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.457344 18.071681 21.686017) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.457344 18.071681 21.686017) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.457344 18.071681 21.686017) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.2614      0.74674082     0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42064 ave       42064 max       42064 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42064
Ave neighs/atom = 87.816284
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45734448385914 y final 0 18.07168060482393 z final 0 21.686016725788715      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.457344 18.071681 21.686017) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.457344 18.071681 21.686017) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.457344 18.071681 21.686017) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.457344 18.071681 21.686017) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.457344 18.071681 21.686017) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.457344 18.071681 21.686017) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.2614      0.74674082     0              0            
Loop time of 1.459e-06 on 1 procs for 0 steps with 479 atoms

548.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.459e-06  |            |       |100.00

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42064 ave       42064 max       42064 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42064
Ave neighs/atom = 87.816284
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456224871603903 y final 0 18.070281089505528 z final 0 21.684337307406246      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456225 18.071681 21.686017) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456225 18.070281 21.686017) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456225 18.070281 21.684337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456225 18.070281 21.684337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456225 18.070281 21.684337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456225 18.070281 21.684337) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3557      149.85617      0              0            
Loop time of 1.333e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42064 ave       42064 max       42064 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42064
Ave neighs/atom = 87.816284
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456224871603903 y final 0 18.070281089505528 z final 0 21.68433730740625      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456225 18.070281 21.684337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456225 18.070281 21.684337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456225 18.070281 21.684337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456225 18.070281 21.684337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456225 18.070281 21.684337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456225 18.070281 21.684337) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3557      149.85617      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 479 atoms

353.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.416e-06  |            |       |100.00

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42064 ave       42064 max       42064 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42064
Ave neighs/atom = 87.816284
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456566208427159 y final 0 18.07070776053473 z final 0 21.684849312641212      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456566 18.070281 21.684337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456566 18.070708 21.684337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456566 18.070708 21.684849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456566 18.070708 21.684849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456566 18.070708 21.684849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456566 18.070708 21.684849) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3735      61.920463      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42064 ave       42064 max       42064 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42064
Ave neighs/atom = 87.816284
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456566208427159 y final 0 18.07070776053473 z final 0 21.684849312641216      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456566 18.070708 21.684849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456566 18.070708 21.684849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456566 18.070708 21.684849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456566 18.070708 21.684849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456566 18.070708 21.684849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456566 18.070708 21.684849) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3735      61.920463      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 479 atoms

959.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42064 ave       42064 max       42064 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42064
Ave neighs/atom = 87.816284
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456904891493544 y final 0 18.071131114367752 z final 0 21.685357337240816      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456905 18.070708 21.684849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456905 18.071131 21.684849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456905 18.071131 21.685357) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456905 18.071131 21.685357) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456905 18.071131 21.685357) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456905 18.071131 21.685357) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3784     -19.445721      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456904891493544 y final 0 18.071131114367752 z final 0 21.685357337240816      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456905 18.071131 21.685357) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456905 18.071131 21.685357) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456905 18.071131 21.685357) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456905 18.071131 21.685357) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456905 18.071131 21.685357) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456905 18.071131 21.685357) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3784     -19.445721      0              0            
Loop time of 1.333e-06 on 1 procs for 0 steps with 479 atoms

450.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.333e-06  |            |       |100.00

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45694690869041 y final 0 18.071183635863846 z final 0 21.685420363036126      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456947 18.071131 21.685357) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456947 18.071184 21.685357) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456947 18.071184 21.68542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456947 18.071184 21.68542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456947 18.071184 21.68542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456947 18.071184 21.68542) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3799     -21.364183      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45694690869041 y final 0 18.071183635863846 z final 0 21.685420363036126      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456947 18.071184 21.68542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456947 18.071184 21.68542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456947 18.071184 21.68542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456947 18.071184 21.68542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456947 18.071184 21.68542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456947 18.071184 21.68542) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3799     -21.364183      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456901373139459 y final 0 18.071126716425162 z final 0 21.685352059709704      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456901 18.071184 21.68542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456901 18.071127 21.68542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456901 18.071127 21.685352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456901 18.071127 21.685352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456901 18.071127 21.685352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456901 18.071127 21.685352) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3804     -4.4674038      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456901373139459 y final 0 18.071126716425162 z final 0 21.685352059709704      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456901 18.071127 21.685352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456901 18.071127 21.685352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456901 18.071127 21.685352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456901 18.071127 21.685352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456901 18.071127 21.685352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456901 18.071127 21.685352) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3804     -4.4674038      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877386321642 y final 0 18.071096732902895 z final 0 21.68531607948298      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071127 21.685352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685352) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8089301      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456781439448363 y final 0 18.070976799311307 z final 0 21.68517215917307      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456781 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456781 18.070977 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456781 18.070977 21.685172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456781 18.070977 21.685172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456781 18.070977 21.685172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456781 18.070977 21.685172) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3772      43.27548       0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42064 ave       42064 max       42064 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42064
Ave neighs/atom = 87.816284
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456870489132212 y final 0 18.07108811141611 z final 0 21.68530573369884      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45687 18.070977 21.685172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45687 18.071088 21.685172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45687 18.071088 21.685306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45687 18.071088 21.685306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45687 18.071088 21.685306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45687 18.071088 21.685306) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3804      6.4442551      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456876813187378 y final 0 18.071096016485065 z final 0 21.685315219781586      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071088 21.685306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685315) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685315) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685315) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685315) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3804      4.0034642      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877329035681 y final 0 18.071096661295446 z final 0 21.685315993554042      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685315) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685315) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.827546       0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456876988575798 y final 0 18.071096235720592 z final 0 21.68531548286422      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685315) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685315) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685315) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685315) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3804      3.9463633      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877309787526 y final 0 18.07109663723525 z final 0 21.685315964681806      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685315) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685315) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8338019      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877097787785 y final 0 18.071096372235573 z final 0 21.6853156466822      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3804      3.910826       0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877302500759 y final 0 18.071096628126792 z final 0 21.685315953751658      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8361702      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877167390195 y final 0 18.07109645923859 z final 0 21.685315751085813      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3804      3.8881864      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687729958056 y final 0 18.071096624476542 z final 0 21.685315949371358      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071096 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8371194      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877256601542 y final 0 18.071096570752772 z final 0 21.685315884902835      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8510901      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687721199587 y final 0 18.07109651499568 z final 0 21.685315817994326      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3804      3.8736809      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877256289259 y final 0 18.071096570362418 z final 0 21.685315884434406      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8511917      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877227055621 y final 0 18.07109653382037 z final 0 21.685315840583954      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.860696       0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877217116185 y final 0 18.071096521396075 z final 0 21.6853158256748      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3804      3.872016       0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877227040245 y final 0 18.071096533801146 z final 0 21.685315840560886      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.860701       0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877220490366 y final 0 18.0710965256138 z final 0 21.685315830736066      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3804      3.8709188      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 479 atoms

479.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.34e-07   |            |       |100.00

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877227033571 y final 0 18.071096533792804 z final 0 21.685315840550874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8607031      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877227032528 y final 0 18.071096533791504 z final 0 21.68531584054931      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8607035      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687722703349 y final 0 18.071096533792705 z final 0 21.685315840550754      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8607032      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687722703357 y final 0 18.071096533792804 z final 0 21.685315840550874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8607031      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456877227033571 y final 0 18.071096533792804 z final 0 21.685315840550874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8607031      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687722703357 y final 0 18.071096533792804 z final 0 21.685315840550874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8607031      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687722703357 y final 0 18.071096533792804 z final 0 21.685315840550874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8607031      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687722703357 y final 0 18.071096533792804 z final 0 21.685315840550874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8607031      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687722703357 y final 0 18.071096533792804 z final 0 21.685315840550874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8607031      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687722703357 y final 0 18.071096533792804 z final 0 21.685315840550874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8607031      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 479 atoms

378.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.92e-07   |            |       |100.00

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687722703357 y final 0 18.071096533792804 z final 0 21.685315840550874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8607031      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687722703357 y final 0 18.071096533792804 z final 0 21.685315840550874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456877 18.071097 21.685316) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.8607031      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879013629058 y final 0 18.071098767037167 z final 0 21.68531852044411      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071097 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685316) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.4051461      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456886160013221 y final 0 18.071107700017382 z final 0 21.68532924002037      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456886 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456886 18.071108 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456886 18.071108 21.685329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456886 18.071108 21.685329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456886 18.071108 21.685329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456886 18.071108 21.685329) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3788      4.7255854      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42064 ave       42064 max       42064 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42064
Ave neighs/atom = 87.816284
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879526587185 y final 0 18.07109940823483 z final 0 21.685319289881303      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45688 18.071108 21.685329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45688 18.071099 21.685329) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45688 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45688 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45688 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45688 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3804      3.2756251      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42056 ave       42056 max       42056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42056
Ave neighs/atom = 87.799582
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879046869982 y final 0 18.071098808588324 z final 0 21.685318570305498      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3968234      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879014901325 y final 0 18.071098768627504 z final 0 21.685318522352514      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.4045174      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 479 atoms

327.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.16e-07   |            |       |100.00

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879036000638 y final 0 18.071098795001646 z final 0 21.685318554001483      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.4021634      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687901540606 y final 0 18.07109876925842 z final 0 21.68531852310961      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.4042681      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687902899848 y final 0 18.07109878624895 z final 0 21.685318543498248      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3975628      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879033619906 y final 0 18.071098792025726 z final 0 21.68531855043038      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.4033346      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879029021835 y final 0 18.07109878627814 z final 0 21.685318543533278      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3975513      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879029368725 y final 0 18.07109878671175 z final 0 21.68531854405361      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973804      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 479 atoms

457.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879031494315 y final 0 18.07109878936874 z final 0 21.685318547241998      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.4043809      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687902937365 y final 0 18.071098786717908 z final 0 21.685318544060998      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.397378       0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879029378422 y final 0 18.071098786723876 z final 0 21.68531854406816      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973757      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 479 atoms

378.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.92e-07   |            |       |100.00

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879030436369 y final 0 18.07109878804631 z final 0 21.68531854565508      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.4049018      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687902937964 y final 0 18.0710987867254 z final 0 21.68531854406999      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973751      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879029379836 y final 0 18.07109878672564 z final 0 21.685318544070277      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.397375       0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879029379655 y final 0 18.071098786725415 z final 0 21.685318544070007      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973751      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687902937964 y final 0 18.0710987867254 z final 0 21.68531854406999      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973751      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687902937964 y final 0 18.0710987867254 z final 0 21.68531854406999      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973751      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687902937964 y final 0 18.0710987867254 z final 0 21.68531854406999      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
WARNING: Too many warnings: 101 vs 100. All future warnings will be suppressed (src/src/thermo.cpp:465)
-1694.3805      3.3973751      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687902937964 y final 0 18.0710987867254 z final 0 21.68531854406999      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973751      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687902937964 y final 0 18.0710987867254 z final 0 21.68531854406999      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973751      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687902937964 y final 0 18.0710987867254 z final 0 21.68531854406999      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973751      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687902937964 y final 0 18.0710987867254 z final 0 21.68531854406999      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973751      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687902937964 y final 0 18.0710987867254 z final 0 21.68531854406999      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973751      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 479 atoms

505.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.92e-07   |            |       |100.00

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45687902937964 y final 0 18.0710987867254 z final 0 21.68531854406999      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973751      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456879029379639 y final 0 18.0710987867254 z final 0 21.68531854406999      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456879 18.071099 21.685319) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      3.3973751      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 479 atoms

700.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1e-06      |            |       |100.00

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456891504380948 y final 0 18.071114380477045 z final 0 21.685337256571966      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456892 18.071099 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456892 18.071114 21.685319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456892 18.071114 21.685337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456892 18.071114 21.685337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456892 18.071114 21.685337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456892 18.071114 21.685337) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805     -0.3369582      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45688249264689 y final 0 18.071103115809468 z final 0 21.68532373897087      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456882 18.071114 21.685337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456882 18.071103 21.685337) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456882 18.071103 21.685324) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456882 18.071103 21.685324) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456882 18.071103 21.685324) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456882 18.071103 21.685324) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      2.1125462      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888440390738 y final 0 18.07111055048928 z final 0 21.685332660586646      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456888 18.071103 21.685324) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456888 18.071111 21.685324) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456888 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456888 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456888 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456888 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.50621166     0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456890462624205 y final 0 18.071113078281115 z final 0 21.685335693936846      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45689 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45689 18.071113 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45689 18.071113 21.685336) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45689 18.071113 21.685336) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45689 18.071113 21.685336) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45689 18.071113 21.685336) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805     -0.017247393    0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888552368987 y final 0 18.071110690462092 z final 0 21.685332828554017      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071113 21.685336) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685336) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.47175602     0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456889813137403 y final 0 18.071112266422613 z final 0 21.685334719706642      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45689 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45689 18.071112 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45689 18.071112 21.685335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45689 18.071112 21.685335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45689 18.071112 21.685335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45689 18.071112 21.685335) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.090586466    0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888624405448 y final 0 18.071110780507667 z final 0 21.68533293660871      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071112 21.685335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685335) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.45244258     0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888552819192 y final 0 18.071110691024845 z final 0 21.685332829229324      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.47161753     0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888555838233 y final 0 18.071110694798648 z final 0 21.685332833757887      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.47068879     0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45688859012184 y final 0 18.071110737653157 z final 0 21.685332885183296      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.46298684     0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888555939829 y final 0 18.071110694925643 z final 0 21.685332833910284      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.47065754     0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888578499958 y final 0 18.071110723125802 z final 0 21.68533286775047      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.46777329     0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45688855607222 y final 0 18.071110695091132 z final 0 21.685332834108866      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.47061681     0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888556500354 y final 0 18.071110695626302 z final 0 21.68533283475107      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.4704851      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888567500156 y final 0 18.07111070937605 z final 0 21.68533285125077      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.47115686     0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45688855653173 y final 0 18.071110695665517 z final 0 21.685332834798132      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.47047545     0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.45688856377089 y final 0 18.07111070471447 z final 0 21.685332845656873      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.46824855     0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888566232205 y final 0 18.071110707791114 z final 0 21.685332849348846      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.47154689     0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888563772457 y final 0 18.07111070471643 z final 0 21.68533284565922      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.46824806     0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888563770953 y final 0 18.07111070471455 z final 0 21.685332845656966      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.46824853     0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888563770953 y final 0 18.07111070471455 z final 0 21.68533284565697      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.46824853     0              0            
Loop time of 6.67e-07 on 1 procs for 0 steps with 479 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.67e-07   |            |       |100.00

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888563771706 y final 0 18.07111070471549 z final 0 21.685332845658095      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.4682483      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888563770972 y final 0 18.07111070471457 z final 0 21.685332845656994      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.46824852     0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888563770953 y final 0 18.07111070471455 z final 0 21.68533284565697      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.46824853     0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888563770967 y final 0 18.071110704714567 z final 0 21.685332845656987      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.46824852     0              0            
Loop time of 7.09e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888563770953 y final 0 18.07111070471455 z final 0 21.68533284565697      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.46824853     0              0            
Loop time of 7.09e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.456888563770953 y final 0 18.07111070471455 z final 0 21.68533284565697      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.456889 18.071111 21.685333) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1694.3805      0.46824853     0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 479 atoms

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

Nlocal:            479 ave         479 max         479 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2905 ave        2905 max        2905 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          42052 ave       42052 max       42052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 42052
Ave neighs/atom = 87.791232
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  479 settings made for x
  479 settings made for y
  479 settings made for z
change_box all     x final 0 14.459527503904514 y final 0 18.074409379880642 z final 0 21.689291255856773      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.459528 18.071111 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.685333) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
echo none
1 atoms in group delatoms
Deleted 1 atoms, new total = 478
run 0
Per MPI rank memory allocation (min/avg/max) = 3.398 | 3.398 | 3.398 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.4315     -1231.2445      0              0            
Loop time of 1.417e-06 on 1 procs for 0 steps with 478 atoms

282.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.417e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 11.838459842423534 y final 0 14.798074803029417 z final 0 17.7576897636353      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.83846 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-340.26655      2827610.5      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 478 atoms

240.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.25e-06   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           4094 ave        4094 max        4094 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          76161 ave       76161 max       76161 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 76161
Ave neighs/atom = 159.33264
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.455375844126161 y final 0 18.0692198051577 z final 0 21.683063766189242      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.455376 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455376 18.06922 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455376 18.06922 21.683064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455376 18.06922 21.683064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455376 18.06922 21.683064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455376 18.06922 21.683064) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.4337     -38.915167      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.455375844126161 y final 0 18.0692198051577 z final 0 21.683063766189242      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.455376 18.06922 21.683064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455376 18.06922 21.683064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455376 18.06922 21.683064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455376 18.06922 21.683064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455376 18.06922 21.683064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455376 18.06922 21.683064) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.4337     -38.915167      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453029469386774 y final 0 18.066286836733465 z final 0 21.679544204080162      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453029 18.06922 21.683064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453029 18.066287 21.683064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453029 18.066287 21.679544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453029 18.066287 21.679544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453029 18.066287 21.679544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453029 18.066287 21.679544) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.615       288.27331      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41877 ave       41877 max       41877 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41877
Ave neighs/atom = 87.608787
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453029469386774 y final 0 18.066286836733465 z final 0 21.679544204080162      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453029 18.066287 21.679544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453029 18.066287 21.679544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453029 18.066287 21.679544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453029 18.066287 21.679544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453029 18.066287 21.679544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453029 18.066287 21.679544) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.615       288.27331      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 478 atoms

457.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41877 ave       41877 max       41877 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41877
Ave neighs/atom = 87.608787
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.452753395727632 y final 0 18.065941744659533 z final 0 21.679130093591446      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.452753 18.066287 21.679544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452753 18.065942 21.679544) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452753 18.065942 21.67913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452753 18.065942 21.67913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452753 18.065942 21.67913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452753 18.065942 21.67913) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6814      366.18481      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 478 atoms

378.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.92e-07   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41843 ave       41843 max       41843 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41843
Ave neighs/atom = 87.537657
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.452753395727632 y final 0 18.065941744659533 z final 0 21.679130093591446      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.452753 18.065942 21.67913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452753 18.065942 21.67913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452753 18.065942 21.67913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452753 18.065942 21.67913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452753 18.065942 21.67913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452753 18.065942 21.67913) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6814      366.18481      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41843 ave       41843 max       41843 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41843
Ave neighs/atom = 87.537657
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453707501036984 y final 0 18.067134376296224 z final 0 21.680561251555478      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453708 18.065942 21.67913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453708 18.067134 21.67913) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453708 18.067134 21.680561) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453708 18.067134 21.680561) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453708 18.067134 21.680561) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453708 18.067134 21.680561) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.7029      129.42403      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41823 ave       41823 max       41823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41823
Ave neighs/atom = 87.495816
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453707501036984 y final 0 18.067134376296224 z final 0 21.680561251555478      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453708 18.067134 21.680561) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453708 18.067134 21.680561) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453708 18.067134 21.680561) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453708 18.067134 21.680561) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453708 18.067134 21.680561) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453708 18.067134 21.680561) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.7029      129.42403      0              0            
Loop time of 1.459e-06 on 1 procs for 0 steps with 478 atoms

205.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.459e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41823 ave       41823 max       41823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41823
Ave neighs/atom = 87.495816
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.454221213426374 y final 0 18.067776516782963 z final 0 21.68133182013956      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.454221 18.067134 21.680561) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454221 18.067777 21.680561) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454221 18.067777 21.681332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454221 18.067777 21.681332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454221 18.067777 21.681332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454221 18.067777 21.681332) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.715       29.947311      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41811 ave       41811 max       41811 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41811
Ave neighs/atom = 87.470711
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.454221213426374 y final 0 18.067776516782963 z final 0 21.68133182013956      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.454221 18.067777 21.681332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454221 18.067777 21.681332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454221 18.067777 21.681332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454221 18.067777 21.681332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454221 18.067777 21.681332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454221 18.067777 21.681332) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.715       29.947311      0              0            
Loop time of 1.5e-06 on 1 procs for 0 steps with 478 atoms

533.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.5e-06    |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41811 ave       41811 max       41811 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41811
Ave neighs/atom = 87.470711
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45406161015805 y final 0 18.067577012697562 z final 0 21.681092415237078      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.454062 18.067777 21.681332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454062 18.067577 21.681332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454062 18.067577 21.681092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454062 18.067577 21.681092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454062 18.067577 21.681092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454062 18.067577 21.681092) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.7193      90.121061      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41811 ave       41811 max       41811 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41811
Ave neighs/atom = 87.470711
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.454061610158048 y final 0 18.067577012697562 z final 0 21.681092415237078      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.454062 18.067577 21.681092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454062 18.067577 21.681092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454062 18.067577 21.681092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454062 18.067577 21.681092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454062 18.067577 21.681092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454062 18.067577 21.681092) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.7193      90.121061      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41811 ave       41811 max       41811 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41811
Ave neighs/atom = 87.470711
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453881500701932 y final 0 18.06735187587742 z final 0 21.68082225105291      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453882 18.067577 21.681092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453882 18.067352 21.681092) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453882 18.067352 21.680822) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453882 18.067352 21.680822) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453882 18.067352 21.680822) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453882 18.067352 21.680822) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.722       148.88782      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41815 ave       41815 max       41815 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41815
Ave neighs/atom = 87.479079
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453881500701932 y final 0 18.06735187587742 z final 0 21.68082225105291      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453882 18.067352 21.680822) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453882 18.067352 21.680822) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453882 18.067352 21.680822) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453882 18.067352 21.680822) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453882 18.067352 21.680822) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453882 18.067352 21.680822) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.722       148.88782      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 478 atoms

444.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41815 ave       41815 max       41815 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41815
Ave neighs/atom = 87.479079
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453838947497216 y final 0 18.067298684371526 z final 0 21.680758421245837      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453839 18.067352 21.680822) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453839 18.067299 21.680822) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453839 18.067299 21.680758) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453839 18.067299 21.680758) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453839 18.067299 21.680758) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453839 18.067299 21.680758) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.7231      159.67446      0              0            
Loop time of 7.09e-07 on 1 procs for 0 steps with 478 atoms

282.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.09e-07   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41811 ave       41811 max       41811 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41811
Ave neighs/atom = 87.470711
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453838947497214 y final 0 18.067298684371526 z final 0 21.680758421245837      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453839 18.067299 21.680758) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453839 18.067299 21.680758) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453839 18.067299 21.680758) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453839 18.067299 21.680758) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453839 18.067299 21.680758) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453839 18.067299 21.680758) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.7231      159.67446      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 478 atoms

600.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.166e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41811 ave       41811 max       41811 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41811
Ave neighs/atom = 87.470711
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453857169227168 y final 0 18.067321461533968 z final 0 21.68078575384077      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453857 18.067299 21.680758) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453857 18.067321 21.680758) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453857 18.067321 21.680786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453857 18.067321 21.680786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453857 18.067321 21.680786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453857 18.067321 21.680786) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.7237      154.44306      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41807 ave       41807 max       41807 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41807
Ave neighs/atom = 87.462343
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453857169227168 y final 0 18.067321461533968 z final 0 21.68078575384077      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453857 18.067321 21.680786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453857 18.067321 21.680786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453857 18.067321 21.680786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453857 18.067321 21.680786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453857 18.067321 21.680786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453857 18.067321 21.680786) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.7237      154.44306      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41807 ave       41807 max       41807 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41807
Ave neighs/atom = 87.462343
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453905265401339 y final 0 18.067381581751683 z final 0 21.68085789810203      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453905 18.067321 21.680786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453905 18.067382 21.680786) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453905 18.067382 21.680858) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453905 18.067382 21.680858) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453905 18.067382 21.680858) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453905 18.067382 21.680858) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.7241      144.3756       0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41803 ave       41803 max       41803 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41803
Ave neighs/atom = 87.453975
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.459527503904514 y final 0 18.074409379880642 z final 0 21.689291255856773      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.459528 18.067382 21.680858) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.680858) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.3677     -1536.9482      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 11.838459842423534 y final 0 14.798074803029417 z final 0 17.7576897636353      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.83846 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-340.86753      2825538.6      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           4094 ave        4094 max        4094 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          76161 ave       76161 max       76161 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 76161
Ave neighs/atom = 159.33264
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45473902589412 y final 0 18.06842378236765 z final 0 21.68210853884118      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.454739 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454739 18.068424 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454739 18.068424 21.682109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454739 18.068424 21.682109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454739 18.068424 21.682109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454739 18.068424 21.682109) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.3706     -161.61108      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45473902589412 y final 0 18.06842378236765 z final 0 21.68210853884118      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.454739 18.068424 21.682109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454739 18.068424 21.682109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454739 18.068424 21.682109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454739 18.068424 21.682109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454739 18.068424 21.682109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454739 18.068424 21.682109) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.3706     -161.61108      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 478 atoms

457.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.452392126059037 y final 0 18.065490157573795 z final 0 21.678588189088558      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.452392 18.068424 21.682109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452392 18.06549 21.682109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452392 18.06549 21.678588) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452392 18.06549 21.678588) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452392 18.06549 21.678588) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452392 18.06549 21.678588) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5491      129.85519      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.452392126059037 y final 0 18.065490157573795 z final 0 21.67858818908856      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.452392 18.06549 21.678588) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452392 18.06549 21.678588) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452392 18.06549 21.678588) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452392 18.06549 21.678588) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452392 18.06549 21.678588) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452392 18.06549 21.678588) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5491      129.85519      0              0            
Loop time of 1.958e-06 on 1 procs for 0 steps with 478 atoms

357.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.958e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.452806428151924 y final 0 18.066008035189903 z final 0 21.67920964222789      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.452806 18.06549 21.678588) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452806 18.066008 21.678588) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452806 18.066008 21.67921) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452806 18.066008 21.67921) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452806 18.066008 21.67921) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452806 18.066008 21.67921) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5886      15.546947      0              0            
Loop time of 1.333e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41857 ave       41857 max       41857 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41857
Ave neighs/atom = 87.566946
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.452806428151922 y final 0 18.066008035189903 z final 0 21.67920964222789      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.452806 18.066008 21.67921) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452806 18.066008 21.67921) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452806 18.066008 21.67921) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452806 18.066008 21.67921) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452806 18.066008 21.67921) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452806 18.066008 21.67921) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5886      15.546947      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 478 atoms

428.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.166e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41857 ave       41857 max       41857 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41857
Ave neighs/atom = 87.566946
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453553930097028 y final 0 18.066942412621284 z final 0 21.68033089514555      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453554 18.066008 21.67921) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453554 18.066942 21.67921) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453554 18.066942 21.680331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453554 18.066942 21.680331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453554 18.066942 21.680331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453554 18.066942 21.680331) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.601      -134.57027      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 478 atoms

417.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41833 ave       41833 max       41833 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41833
Ave neighs/atom = 87.516736
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453553930097028 y final 0 18.066942412621284 z final 0 21.68033089514555      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453554 18.066942 21.680331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453554 18.066942 21.680331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453554 18.066942 21.680331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453554 18.066942 21.680331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453554 18.066942 21.680331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453554 18.066942 21.680331) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.601      -134.57027      0              0            
Loop time of 1.458e-06 on 1 procs for 0 steps with 478 atoms

480.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.458e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41833 ave       41833 max       41833 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41833
Ave neighs/atom = 87.516736
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453737962609006 y final 0 18.067172453261257 z final 0 21.68060694391352      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453738 18.066942 21.680331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453738 18.067172 21.680331) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453738 18.067172 21.680607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453738 18.067172 21.680607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453738 18.067172 21.680607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453738 18.067172 21.680607) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6061     -135.95021      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41801 ave       41801 max       41801 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41801
Ave neighs/atom = 87.449791
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453737962609006 y final 0 18.067172453261257 z final 0 21.68060694391352      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453738 18.067172 21.680607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453738 18.067172 21.680607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453738 18.067172 21.680607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453738 18.067172 21.680607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453738 18.067172 21.680607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453738 18.067172 21.680607) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6061     -135.95021      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41801 ave       41801 max       41801 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41801
Ave neighs/atom = 87.449791
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453637089154434 y final 0 18.067046361443047 z final 0 21.68045563373167      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453637 18.067172 21.680607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453637 18.067046 21.680607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453637 18.067046 21.680456) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453637 18.067046 21.680456) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453637 18.067046 21.680456) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453637 18.067046 21.680456) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6084     -82.572658      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41801 ave       41801 max       41801 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41801
Ave neighs/atom = 87.449791
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453637089154432 y final 0 18.067046361443047 z final 0 21.68045563373167      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453637 18.067046 21.680456) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453637 18.067046 21.680456) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453637 18.067046 21.680456) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453637 18.067046 21.680456) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453637 18.067046 21.680456) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453637 18.067046 21.680456) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6084     -82.572658      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 478 atoms

622.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41801 ave       41801 max       41801 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41801
Ave neighs/atom = 87.449791
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453560577624222 y final 0 18.066950722030292 z final 0 21.68034086643636      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453561 18.067046 21.680456) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453561 18.066951 21.680456) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453561 18.066951 21.680341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453561 18.066951 21.680341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453561 18.066951 21.680341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453561 18.066951 21.680341) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6096     -53.662494      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 478 atoms

417.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41801 ave       41801 max       41801 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41801
Ave neighs/atom = 87.449791
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453560577624222 y final 0 18.066950722030292 z final 0 21.68034086643636      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453561 18.066951 21.680341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453561 18.066951 21.680341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453561 18.066951 21.680341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453561 18.066951 21.680341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453561 18.066951 21.680341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453561 18.066951 21.680341) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6096     -53.662494      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 478 atoms

480.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.25e-06   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41801 ave       41801 max       41801 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41801
Ave neighs/atom = 87.449791
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45354274559131 y final 0 18.066928431989158 z final 0 21.680314118387      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453543 18.066951 21.680341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453543 18.066928 21.680341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453543 18.066928 21.680314) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453543 18.066928 21.680314) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453543 18.066928 21.680314) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453543 18.066928 21.680314) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6102     -51.688615      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41809 ave       41809 max       41809 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41809
Ave neighs/atom = 87.466527
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45354274559131 y final 0 18.066928431989158 z final 0 21.680314118387003      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453543 18.066928 21.680314) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453543 18.066928 21.680314) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453543 18.066928 21.680314) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453543 18.066928 21.680314) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453543 18.066928 21.680314) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453543 18.066928 21.680314) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6102     -51.688615      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 478 atoms

423.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.416e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41809 ave       41809 max       41809 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41809
Ave neighs/atom = 87.466527
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45356270463119 y final 0 18.066953380789013 z final 0 21.680344056946826      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453563 18.066928 21.680314) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453563 18.066953 21.680314) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453563 18.066953 21.680344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453563 18.066953 21.680344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453563 18.066953 21.680344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453563 18.066953 21.680344) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6104     -61.17885       0              0            
Loop time of 1.458e-06 on 1 procs for 0 steps with 478 atoms

342.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.458e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41809 ave       41809 max       41809 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41809
Ave neighs/atom = 87.466527
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45356270463119 y final 0 18.066953380789013 z final 0 21.680344056946826      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453563 18.066953 21.680344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453563 18.066953 21.680344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453563 18.066953 21.680344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453563 18.066953 21.680344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453563 18.066953 21.680344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453563 18.066953 21.680344) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6104     -61.17885       0              0            
Loop time of 1.334e-06 on 1 procs for 0 steps with 478 atoms

449.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.334e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41809 ave       41809 max       41809 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41809
Ave neighs/atom = 87.466527
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453591700827298 y final 0 18.06698962603415 z final 0 21.68038755124099      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453592 18.066953 21.680344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453592 18.06699 21.680344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453592 18.06699 21.680388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453592 18.06699 21.680388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453592 18.06699 21.680388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453592 18.06699 21.680388) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.6105     -71.333493      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41809 ave       41809 max       41809 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41809
Ave neighs/atom = 87.466527
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.459527503904514 y final 0 18.074409379880642 z final 0 21.689291255856773      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.459528 18.06699 21.680388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.680388) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.3123     -1413.2538      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 478 atoms

417.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.58e-07   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 11.838459842423534 y final 0 14.798074803029417 z final 0 17.7576897636353      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.83846 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-340.76971      2825773.4      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           4094 ave        4094 max        4094 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          76161 ave       76161 max       76161 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 76161
Ave neighs/atom = 159.33264
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.454878228611516 y final 0 18.068597785764393 z final 0 21.682317342917276      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.454878 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454878 18.068598 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454878 18.068598 21.682317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454878 18.068598 21.682317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454878 18.068598 21.682317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454878 18.068598 21.682317) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.3151     -77.133312      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.454878228611516 y final 0 18.068597785764393 z final 0 21.68231734291728      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.454878 18.068598 21.682317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454878 18.068598 21.682317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454878 18.068598 21.682317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454878 18.068598 21.682317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454878 18.068598 21.682317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454878 18.068598 21.682317) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.3151     -77.133312      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 478 atoms

464.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.292e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45265547773404 y final 0 18.06581934716742 z final 0 21.678983216601214      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.452655 18.068598 21.682317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452655 18.065819 21.682317) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452655 18.065819 21.678983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452655 18.065819 21.678983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452655 18.065819 21.678983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452655 18.065819 21.678983) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.4927      243.25336      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45265547773404 y final 0 18.06581934716742 z final 0 21.678983216601218      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.452655 18.065819 21.678983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452655 18.065819 21.678983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452655 18.065819 21.678983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452655 18.065819 21.678983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452655 18.065819 21.678983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452655 18.065819 21.678983) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.4927      243.25336      0              0            
Loop time of 1.5e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453028175458744 y final 0 18.06628521932326 z final 0 21.67954226318832      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453028 18.065819 21.678983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453028 18.066285 21.678983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453028 18.066285 21.679542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453028 18.066285 21.679542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453028 18.066285 21.679542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453028 18.066285 21.679542) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5332      146.63494      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41873 ave       41873 max       41873 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41873
Ave neighs/atom = 87.600418
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453028175458744 y final 0 18.06628521932326 z final 0 21.67954226318832      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453028 18.066285 21.679542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453028 18.066285 21.679542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453028 18.066285 21.679542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453028 18.066285 21.679542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453028 18.066285 21.679542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453028 18.066285 21.679542) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5332      146.63494      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41873 ave       41873 max       41873 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41873
Ave neighs/atom = 87.600418
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453739592950207 y final 0 18.06717449118758 z final 0 21.68060938942553      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45374 18.066285 21.679542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45374 18.067174 21.679542) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45374 18.067174 21.680609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45374 18.067174 21.680609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45374 18.067174 21.680609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45374 18.067174 21.680609) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5451     -30.577767      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 478 atoms

327.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.16e-07   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41851 ave       41851 max       41851 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41851
Ave neighs/atom = 87.554393
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453739592950205 y final 0 18.06717449118758 z final 0 21.68060938942553      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45374 18.067174 21.680609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45374 18.067174 21.680609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45374 18.067174 21.680609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45374 18.067174 21.680609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45374 18.067174 21.680609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45374 18.067174 21.680609) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5451     -30.577767      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 478 atoms

575.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.042e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41851 ave       41851 max       41851 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41851
Ave neighs/atom = 87.554393
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453938898593616 y final 0 18.067423623241844 z final 0 21.680908347890654      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453939 18.067174 21.680609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453939 18.067424 21.680609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453939 18.067424 21.680908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453939 18.067424 21.680908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453939 18.067424 21.680908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453939 18.067424 21.680908) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5495     -67.987577      0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 478 atoms

165.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.209e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41849 ave       41849 max       41849 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41849
Ave neighs/atom = 87.550209
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453938898593616 y final 0 18.067423623241844 z final 0 21.680908347890654      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453939 18.067424 21.680908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453939 18.067424 21.680908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453939 18.067424 21.680908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453939 18.067424 21.680908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453939 18.067424 21.680908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453939 18.067424 21.680908) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5495     -67.987577      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41849 ave       41849 max       41849 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41849
Ave neighs/atom = 87.550209
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453851637387059 y final 0 18.067314546733652 z final 0 21.680777456080822      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453852 18.067424 21.680908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453852 18.067315 21.680908) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453852 18.067315 21.680777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453852 18.067315 21.680777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453852 18.067315 21.680777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453852 18.067315 21.680777) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5514     -37.132069      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41837 ave       41837 max       41837 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41837
Ave neighs/atom = 87.525105
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453851637387059 y final 0 18.067314546733652 z final 0 21.680777456080826      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453852 18.067315 21.680777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453852 18.067315 21.680777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453852 18.067315 21.680777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453852 18.067315 21.680777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453852 18.067315 21.680777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453852 18.067315 21.680777) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5514     -37.132069      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41837 ave       41837 max       41837 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41837
Ave neighs/atom = 87.525105
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453775853003346 y final 0 18.067219816254017 z final 0 21.680663779505256      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453776 18.067315 21.680777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453776 18.06722 21.680777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453776 18.06722 21.680664) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453776 18.06722 21.680664) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453776 18.06722 21.680664) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453776 18.06722 21.680664) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5524     -15.829577      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41833 ave       41833 max       41833 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41833
Ave neighs/atom = 87.516736
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453775853003346 y final 0 18.067219816254017 z final 0 21.680663779505256      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453776 18.06722 21.680664) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453776 18.06722 21.680664) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453776 18.06722 21.680664) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453776 18.06722 21.680664) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453776 18.06722 21.680664) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453776 18.06722 21.680664) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5524     -15.829577      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 478 atoms

514.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.167e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41833 ave       41833 max       41833 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41833
Ave neighs/atom = 87.516736
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453759824416741 y final 0 18.067199780520763 z final 0 21.68063973662535      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45376 18.06722 21.680664) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45376 18.0672 21.680664) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45376 18.0672 21.68064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45376 18.0672 21.68064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45376 18.0672 21.68064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45376 18.0672 21.68064) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5529     -13.930923      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41837 ave       41837 max       41837 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41837
Ave neighs/atom = 87.525105
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453759824416741 y final 0 18.067199780520763 z final 0 21.68063973662535      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45376 18.0672 21.68064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45376 18.0672 21.68064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45376 18.0672 21.68064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45376 18.0672 21.68064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45376 18.0672 21.68064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45376 18.0672 21.68064) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5529     -13.930923      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 478 atoms

553.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.084e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41837 ave       41837 max       41837 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41837
Ave neighs/atom = 87.525105
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45378044693803 y final 0 18.067225558672376 z final 0 21.68067067040728      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45378 18.0672 21.68064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45378 18.067226 21.68064) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45378 18.067226 21.680671) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45378 18.067226 21.680671) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45378 18.067226 21.680671) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45378 18.067226 21.680671) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5532     -21.852801      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41839 ave       41839 max       41839 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41839
Ave neighs/atom = 87.529289
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45378044693803 y final 0 18.067225558672376 z final 0 21.68067067040728      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45378 18.067226 21.680671) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45378 18.067226 21.680671) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45378 18.067226 21.680671) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45378 18.067226 21.680671) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45378 18.067226 21.680671) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45378 18.067226 21.680671) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5532     -21.852801      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41839 ave       41839 max       41839 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41839
Ave neighs/atom = 87.529289
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453810092690498 y final 0 18.06726261586296 z final 0 21.680715139035986      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45381 18.067226 21.680671) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45381 18.067263 21.680671) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45381 18.067263 21.680715) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45381 18.067263 21.680715) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45381 18.067263 21.680715) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45381 18.067263 21.680715) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5533     -29.881921      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41837 ave       41837 max       41837 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41837
Ave neighs/atom = 87.525105
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.459527503904514 y final 0 18.074409379880642 z final 0 21.689291255856773      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.459528 18.067263 21.680715) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.680715) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.311      -1361.2387      0              0            
Loop time of 1.333e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 11.838459842423534 y final 0 14.798074803029417 z final 0 17.7576897636353      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.83846 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-340.74854      2825783.6      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           4094 ave        4094 max        4094 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          76161 ave       76161 max       76161 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 76161
Ave neighs/atom = 159.33264
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.455038891989329 y final 0 18.06879861498666 z final 0 21.682558337983995      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.455039 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455039 18.068799 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455039 18.068799 21.682558) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455039 18.068799 21.682558) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455039 18.068799 21.682558) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455039 18.068799 21.682558) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.3136     -71.15404       0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.455038891989329 y final 0 18.06879861498666 z final 0 21.682558337983995      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.455039 18.068799 21.682558) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455039 18.068799 21.682558) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455039 18.068799 21.682558) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455039 18.068799 21.682558) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455039 18.068799 21.682558) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.455039 18.068799 21.682558) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.3136     -71.15404       0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 478 atoms

480.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.041e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.452823434228815 y final 0 18.066029292786155 z final 0 21.679235151343224      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.452823 18.068799 21.682558) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452823 18.066029 21.682558) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452823 18.066029 21.679235) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452823 18.066029 21.679235) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452823 18.066029 21.679235) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452823 18.066029 21.679235) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.4891      275.26293      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.452823434228815 y final 0 18.066029292786155 z final 0 21.679235151343224      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.452823 18.066029 21.679235) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452823 18.066029 21.679235) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452823 18.066029 21.679235) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452823 18.066029 21.679235) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452823 18.066029 21.679235) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452823 18.066029 21.679235) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.4891      275.26293      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 478 atoms

745.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.208e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453222176960935 y final 0 18.066527721201346 z final 0 21.679833265441406      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453222 18.066029 21.679235) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453222 18.066528 21.679235) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453222 18.066528 21.679833) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453222 18.066528 21.679833) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453222 18.066528 21.679833) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453222 18.066528 21.679833) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.529       179.84597      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 478 atoms

428.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.166e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41877 ave       41877 max       41877 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41877
Ave neighs/atom = 87.608787
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453222176960935 y final 0 18.066527721201346 z final 0 21.679833265441406      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453222 18.066528 21.679833) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453222 18.066528 21.679833) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453222 18.066528 21.679833) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453222 18.066528 21.679833) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453222 18.066528 21.679833) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453222 18.066528 21.679833) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.529       179.84597      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 478 atoms

533.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.125e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41877 ave       41877 max       41877 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41877
Ave neighs/atom = 87.608787
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453916259941538 y final 0 18.067395324927112 z final 0 21.68087438991231      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453916 18.066528 21.679833) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453916 18.067395 21.679833) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453916 18.067395 21.680874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453916 18.067395 21.680874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453916 18.067395 21.680874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453916 18.067395 21.680874) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.541       16.067068      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41855 ave       41855 max       41855 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41855
Ave neighs/atom = 87.562762
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453916259941536 y final 0 18.067395324927112 z final 0 21.68087438991231      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453916 18.067395 21.680874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453916 18.067395 21.680874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453916 18.067395 21.680874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453916 18.067395 21.680874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453916 18.067395 21.680874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453916 18.067395 21.680874) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.541       16.067068      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41855 ave       41855 max       41855 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41855
Ave neighs/atom = 87.562762
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.454090391474468 y final 0 18.06761298934328 z final 0 21.68113558721171      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45409 18.067395 21.680874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45409 18.067613 21.680874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45409 18.067613 21.681136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45409 18.067613 21.681136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45409 18.067613 21.681136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45409 18.067613 21.681136) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5453     -8.2476934      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41843 ave       41843 max       41843 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41843
Ave neighs/atom = 87.537657
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.454090391474468 y final 0 18.06761298934328 z final 0 21.681135587211713      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45409 18.067613 21.681136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45409 18.067613 21.681136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45409 18.067613 21.681136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45409 18.067613 21.681136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45409 18.067613 21.681136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45409 18.067613 21.681136) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5453     -8.2476934      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41843 ave       41843 max       41843 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41843
Ave neighs/atom = 87.537657
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.454015131796716 y final 0 18.067518914746092 z final 0 21.681022697695088      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.454015 18.067613 21.681136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454015 18.067519 21.681136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454015 18.067519 21.681023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454015 18.067519 21.681023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454015 18.067519 21.681023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454015 18.067519 21.681023) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5472      26.134277      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41847 ave       41847 max       41847 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41847
Ave neighs/atom = 87.546025
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.454015131796716 y final 0 18.067518914746092 z final 0 21.681022697695088      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.454015 18.067519 21.681023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454015 18.067519 21.681023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454015 18.067519 21.681023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454015 18.067519 21.681023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454015 18.067519 21.681023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454015 18.067519 21.681023) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5472      26.134277      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41847 ave       41847 max       41847 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41847
Ave neighs/atom = 87.546025
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453954107979223 y final 0 18.067442634974228 z final 0 21.680931161968854      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453954 18.067519 21.681023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453954 18.067443 21.681023) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453954 18.067443 21.680931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453954 18.067443 21.680931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453954 18.067443 21.680931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453954 18.067443 21.680931) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5481      47.090955      0              0            
Loop time of 1.417e-06 on 1 procs for 0 steps with 478 atoms

141.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.417e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41855 ave       41855 max       41855 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41855
Ave neighs/atom = 87.562762
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453954107979223 y final 0 18.067442634974228 z final 0 21.680931161968854      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453954 18.067443 21.680931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453954 18.067443 21.680931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453954 18.067443 21.680931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453954 18.067443 21.680931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453954 18.067443 21.680931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453954 18.067443 21.680931) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5481      47.090955      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 478 atoms

763.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.17e-07   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41855 ave       41855 max       41855 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41855
Ave neighs/atom = 87.562762
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453940094875579 y final 0 18.067425118594674 z final 0 21.68091014231339      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45394 18.067443 21.680931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45394 18.067425 21.680931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45394 18.067425 21.68091) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45394 18.067425 21.68091) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45394 18.067425 21.68091) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45394 18.067425 21.68091) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5485      50.456398      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41855 ave       41855 max       41855 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41855
Ave neighs/atom = 87.562762
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453940094875579 y final 0 18.067425118594674 z final 0 21.68091014231339      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45394 18.067425 21.68091) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45394 18.067425 21.68091) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45394 18.067425 21.68091) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45394 18.067425 21.68091) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45394 18.067425 21.68091) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45394 18.067425 21.68091) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5485      50.456398      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 478 atoms

625.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 9.59e-07   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41855 ave       41855 max       41855 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41855
Ave neighs/atom = 87.562762
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453953401072477 y final 0 18.067441751340798 z final 0 21.68093010160874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453953 18.067425 21.68091) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453953 18.067442 21.68091) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453953 18.067442 21.68093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453953 18.067442 21.68093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453953 18.067442 21.68093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453953 18.067442 21.68093) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5487      45.690167      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 478 atoms

457.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41855 ave       41855 max       41855 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41855
Ave neighs/atom = 87.562762
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453953401072477 y final 0 18.067441751340798 z final 0 21.68093010160874      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453953 18.067442 21.68093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453953 18.067442 21.68093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453953 18.067442 21.68093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453953 18.067442 21.68093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453953 18.067442 21.68093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453953 18.067442 21.68093) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5487      45.690167      0              0            
Loop time of 1.417e-06 on 1 procs for 0 steps with 478 atoms

352.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.417e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41855 ave       41855 max       41855 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41855
Ave neighs/atom = 87.562762
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453975522448982 y final 0 18.06746940306143 z final 0 21.680963283673503      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453976 18.067442 21.68093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453976 18.067469 21.68093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453976 18.067469 21.680963) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453976 18.067469 21.680963) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453976 18.067469 21.680963) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453976 18.067469 21.680963) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5488      40.254842      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41857 ave       41857 max       41857 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41857
Ave neighs/atom = 87.566946
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.459527503904514 y final 0 18.074409379880642 z final 0 21.689291255856773      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.459528 18.067469 21.680963) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.680963) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.459528 18.074409 21.689291) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.3202     -1352.2122      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 478 atoms

320.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.25e-06   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 11.838459842423534 y final 0 14.798074803029417 z final 0 17.7576897636353      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.83846 18.074409 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 21.689291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.83846 14.798075 17.75769) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-340.82533      2825647        0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           4094 ave        4094 max        4094 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          76161 ave       76161 max       76161 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 76161
Ave neighs/atom = 159.33264
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.455069624584114 y final 0 18.06883703073014 z final 0 21.682604436876172      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45507 14.798075 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45507 18.068837 17.75769) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45507 18.068837 21.682604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45507 18.068837 21.682604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45507 18.068837 21.682604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45507 18.068837 21.682604) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.3228     -70.788037      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.455069624584114 y final 0 18.06883703073014 z final 0 21.682604436876172      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45507 18.068837 21.682604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45507 18.068837 21.682604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45507 18.068837 21.682604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45507 18.068837 21.682604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45507 18.068837 21.682604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45507 18.068837 21.682604) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.3228     -70.788037      0              0            
Loop time of 1.667e-06 on 1 procs for 0 steps with 478 atoms

539.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.667e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.452632029377577 y final 0 18.06579003672197 z final 0 21.678948044066367      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.452632 18.068837 21.682604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452632 18.06579 21.682604) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452632 18.06579 21.678948) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452632 18.06579 21.678948) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452632 18.06579 21.678948) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452632 18.06579 21.678948) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5175      278.01046      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.452632029377577 y final 0 18.06579003672197 z final 0 21.678948044066367      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.452632 18.06579 21.678948) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452632 18.06579 21.678948) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452632 18.06579 21.678948) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452632 18.06579 21.678948) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452632 18.06579 21.678948) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.452632 18.06579 21.678948) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5175      278.01046      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41889 ave       41889 max       41889 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41889
Ave neighs/atom = 87.633891
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45317994234968 y final 0 18.066474927937097 z final 0 21.67976991352452      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45318 18.06579 21.678948) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45318 18.066475 21.678948) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45318 18.066475 21.67977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45318 18.066475 21.67977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45318 18.066475 21.67977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45318 18.066475 21.67977) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5553      137.80243      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41883 ave       41883 max       41883 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41883
Ave neighs/atom = 87.621339
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45317994234968 y final 0 18.066474927937097 z final 0 21.67976991352452      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.45318 18.066475 21.67977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45318 18.066475 21.67977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45318 18.066475 21.67977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45318 18.066475 21.67977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45318 18.066475 21.67977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.45318 18.066475 21.67977) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5553      137.80243      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 478 atoms

457.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 8.75e-07   |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41883 ave       41883 max       41883 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41883
Ave neighs/atom = 87.621339
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.4538750215923 y final 0 18.067343776990374 z final 0 21.68081253238845      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453875 18.066475 21.67977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453875 18.067344 21.67977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453875 18.067344 21.680813) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453875 18.067344 21.680813) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453875 18.067344 21.680813) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453875 18.067344 21.680813) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.566      -32.198775      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41865 ave       41865 max       41865 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41865
Ave neighs/atom = 87.583682
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.4538750215923 y final 0 18.067343776990374 z final 0 21.68081253238845      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453875 18.067344 21.680813) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453875 18.067344 21.680813) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453875 18.067344 21.680813) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453875 18.067344 21.680813) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453875 18.067344 21.680813) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453875 18.067344 21.680813) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.566      -32.198775      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 478 atoms

464.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.292e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41865 ave       41865 max       41865 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41865
Ave neighs/atom = 87.583682
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.454004154925597 y final 0 18.067505193657 z final 0 21.681006232388402      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.454004 18.067344 21.680813) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454004 18.067505 21.680813) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454004 18.067505 21.681006) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454004 18.067505 21.681006) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454004 18.067505 21.681006) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454004 18.067505 21.681006) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.57       -48.707442      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41861 ave       41861 max       41861 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41861
Ave neighs/atom = 87.575314
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.454004154925597 y final 0 18.067505193657 z final 0 21.681006232388402      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.454004 18.067505 21.681006) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454004 18.067505 21.681006) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454004 18.067505 21.681006) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454004 18.067505 21.681006) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454004 18.067505 21.681006) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.454004 18.067505 21.681006) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.57       -48.707442      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 478 atoms

514.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.167e-06  |            |       |100.00

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41861 ave       41861 max       41861 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41861
Ave neighs/atom = 87.575314
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45389922227326 y final 0 18.067374027841577 z final 0 21.680848833409897      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453899 18.067505 21.681006) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453899 18.067374 21.681006) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453899 18.067374 21.680849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453899 18.067374 21.680849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453899 18.067374 21.680849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453899 18.067374 21.680849) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5715     -11.81702       0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41857 ave       41857 max       41857 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41857
Ave neighs/atom = 87.566946
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.45389922227326 y final 0 18.067374027841577 z final 0 21.680848833409897      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453899 18.067374 21.680849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453899 18.067374 21.680849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453899 18.067374 21.680849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453899 18.067374 21.680849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453899 18.067374 21.680849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453899 18.067374 21.680849) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5715     -11.81702       0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41857 ave       41857 max       41857 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41857
Ave neighs/atom = 87.566946
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453840819876955 y final 0 18.0673010248462 z final 0 21.680761229815445      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453841 18.067374 21.680849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453841 18.067301 21.680849) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453841 18.067301 21.680761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453841 18.067301 21.680761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453841 18.067301 21.680761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453841 18.067301 21.680761) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5722      3.6790406      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41857 ave       41857 max       41857 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41857
Ave neighs/atom = 87.566946
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453840819876953 y final 0 18.0673010248462 z final 0 21.68076122981545      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453841 18.067301 21.680761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453841 18.067301 21.680761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453841 18.067301 21.680761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453841 18.067301 21.680761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453841 18.067301 21.680761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453841 18.067301 21.680761) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5722      3.6790406      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41857 ave       41857 max       41857 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41857
Ave neighs/atom = 87.566946
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453834953897776 y final 0 18.06729369237223 z final 0 21.680752430846685      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453835 18.067301 21.680761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453835 18.067294 21.680761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453835 18.067294 21.680752) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453835 18.067294 21.680752) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453835 18.067294 21.680752) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453835 18.067294 21.680752) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5725      1.2653692      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41857 ave       41857 max       41857 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41857
Ave neighs/atom = 87.566946
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453834953897776 y final 0 18.06729369237223 z final 0 21.680752430846685      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453835 18.067294 21.680752) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453835 18.067294 21.680752) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453835 18.067294 21.680752) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453835 18.067294 21.680752) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453835 18.067294 21.680752) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453835 18.067294 21.680752) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5725      1.2653692      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41857 ave       41857 max       41857 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41857
Ave neighs/atom = 87.566946
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453854284271307 y final 0 18.06731785533914 z final 0 21.680781426406988      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453854 18.067294 21.680752) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453854 18.067318 21.680752) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453854 18.067318 21.680781) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453854 18.067318 21.680781) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453854 18.067318 21.680781) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453854 18.067318 21.680781) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5727     -7.1834451      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41857 ave       41857 max       41857 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41857
Ave neighs/atom = 87.566946
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453854284271307 y final 0 18.06731785533914 z final 0 21.680781426406988      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453854 18.067318 21.680781) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453854 18.067318 21.680781) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453854 18.067318 21.680781) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453854 18.067318 21.680781) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453854 18.067318 21.680781) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453854 18.067318 21.680781) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5727     -7.1834451      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41857 ave       41857 max       41857 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41857
Ave neighs/atom = 87.566946
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 14.453878895411723 y final 0 18.067348619264664 z final 0 21.680818343117615      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (14.453879 18.067318 21.680781) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453879 18.067349 21.680781) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453879 18.067349 21.680818) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453879 18.067349 21.680818) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453879 18.067349 21.680818) with tilt (0 0 0)
  triclinic box = (0 0 0) to (14.453879 18.067349 21.680818) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1689.5728     -15.021529      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 478 atoms

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

Nlocal:            478 ave         478 max         478 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2898 ave        2898 max        2898 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          41857 ave       41857 max       41857 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 41857
Ave neighs/atom = 87.566946
Neighbor list builds = 0
Dangerous builds = 0
change_box all boundary p p f
Changing box ...
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  478 settings made for x
  478 settings made for y
  478 settings made for z
change_box all     x final 0 10.224429950764275 y final 0 7.229763751952257 z final 0 44.28302113306516      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.22443 18.067349 21.680818) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.22443 7.2297638 21.680818) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.22443 7.2297638 44.283021) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.22443 7.2297638 44.283021) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.22443 7.2297638 44.283021) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.22443 7.2297638 44.283021) with tilt (0 0 0)
echo none
318 atoms in group delatoms
Deleted 318 atoms, new total = 160
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.19308     -10361.281      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 8.371055233382307 y final 0 5.919229921211767 z final 0 36.25587124084262      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.3710552 7.2297638 44.283021) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.3710552 5.9192299 44.283021) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.3710552 5.9192299 36.255871) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.3710552 5.9192299 36.255871) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.3710552 5.9192299 36.255871) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.3710552 5.9192299 36.255871) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.54213      1620865.4      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1570 ave        1570 max        1570 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          21984 ave       21984 max       21984 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 21984
Ave neighs/atom = 137.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.188315865057024 y final 0 7.204227237052308 z final 0 44.12660744269057      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.188316 5.9192299 36.255871) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.188316 7.2042272 36.255871) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.188316 7.2042272 44.126607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.188316 7.2042272 44.126607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.188316 7.2042272 44.126607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.188316 7.2042272 44.126607) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.30987     -1574.2661      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.188315865057024 y final 0 7.204227237052308 z final 0 44.12660744269057      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.188316 7.2042272 44.126607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.188316 7.2042272 44.126607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.188316 7.2042272 44.126607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.188316 7.2042272 44.126607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.188316 7.2042272 44.126607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.188316 7.2042272 44.126607) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.30987     -1574.2661      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.187418197359827 y final 0 7.2035924901363675 z final 0 44.12271955477928      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.187418 7.2042272 44.126607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187418 7.2035925 44.126607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187418 7.2035925 44.12272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187418 7.2035925 44.12272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187418 7.2035925 44.12272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187418 7.2035925 44.12272) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.37231     -446.44918      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.187418197359827 y final 0 7.2035924901363675 z final 0 44.12271955477928      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.187418 7.2035925 44.12272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187418 7.2035925 44.12272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187418 7.2035925 44.12272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187418 7.2035925 44.12272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187418 7.2035925 44.12272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187418 7.2035925 44.12272) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.37231     -446.44918      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.189439277784516 y final 0 7.205021609809988 z final 0 44.1314730547386      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.189439 7.2035925 44.12272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.189439 7.2050216 44.12272) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.189439 7.2050216 44.131473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.189439 7.2050216 44.131473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.189439 7.2050216 44.131473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.189439 7.2050216 44.131473) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.38237     -503.10164      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.189439277784516 y final 0 7.205021609809988 z final 0 44.1314730547386      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.189439 7.2050216 44.131473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.189439 7.2050216 44.131473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.189439 7.2050216 44.131473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.189439 7.2050216 44.131473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.189439 7.2050216 44.131473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.189439 7.2050216 44.131473) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.38237     -503.10164      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.190142982767311 y final 0 7.205519204375276 z final 0 44.134520870878426      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.190143 7.2050216 44.131473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.190143 7.2055192 44.131473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.190143 7.2055192 44.134521) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.190143 7.2055192 44.134521) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.190143 7.2055192 44.134521) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.190143 7.2055192 44.134521) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.38457     -616.89129      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.19141749352903 y final 0 7.206420419577584 z final 0 44.140040903512755      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.191417 7.2055192 44.134521) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.191417 7.2064204 44.134521) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.191417 7.2064204 44.140041) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.191417 7.2064204 44.140041) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.191417 7.2064204 44.140041) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.191417 7.2064204 44.140041) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.38609     -821.87689      0              0            
Loop time of 1.291e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.19141749352903 y final 0 7.206420419577584 z final 0 44.140040903512755      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.191417 7.2064204 44.140041) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.191417 7.2064204 44.140041) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.191417 7.2064204 44.140041) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.191417 7.2064204 44.140041) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.191417 7.2064204 44.140041) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.191417 7.2064204 44.140041) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.38609     -821.87689      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.190527780457389 y final 0 7.205791297431315 z final 0 44.13618746787442      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.190528 7.2064204 44.140041) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.190528 7.2057913 44.140041) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.190528 7.2057913 44.136187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.190528 7.2057913 44.136187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.190528 7.2057913 44.136187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.190528 7.2057913 44.136187) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.3888      -713.63049      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.187424300575735 y final 0 7.203596805761721 z final 0 44.12274598841326      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.187424 7.2057913 44.136187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187424 7.2035968 44.136187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187424 7.2035968 44.122746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187424 7.2035968 44.122746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187424 7.2035968 44.122746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187424 7.2035968 44.122746) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.39287     -329.73496      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.187424300575735 y final 0 7.20359680576172 z final 0 44.12274598841326      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.187424 7.2035968 44.122746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187424 7.2035968 44.122746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187424 7.2035968 44.122746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187424 7.2035968 44.122746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187424 7.2035968 44.122746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.187424 7.2035968 44.122746) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.39287     -329.73496      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.186192103346107 y final 0 7.20272551074489 z final 0 44.117409219887044      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.186192 7.2035968 44.122746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.186192 7.2027255 44.122746) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.186192 7.2027255 44.117409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.186192 7.2027255 44.117409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.186192 7.2027255 44.117409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.186192 7.2027255 44.117409) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.39699     -146.12896      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.18260580563695 y final 0 7.20018961531539 z final 0 44.101876608483884      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.182606 7.2027255 44.117409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.182606 7.2001896 44.117409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.182606 7.2001896 44.101877) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.182606 7.2001896 44.101877) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.182606 7.2001896 44.101877) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.182606 7.2001896 44.101877) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.40194      395.9856       0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.18260580563695 y final 0 7.20018961531539 z final 0 44.101876608483884      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.182606 7.2001896 44.101877) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.182606 7.2001896 44.101877) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.182606 7.2001896 44.101877) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.182606 7.2001896 44.101877) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.182606 7.2001896 44.101877) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.182606 7.2001896 44.101877) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.40194      395.9856       0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.180447995166181 y final 0 7.198663812898993 z final 0 44.0925309171213      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.180448 7.2001896 44.101877) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.180448 7.1986638 44.101877) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.180448 7.1986638 44.092531) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.180448 7.1986638 44.092531) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.180448 7.1986638 44.092531) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.180448 7.1986638 44.092531) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.40737      637.04846      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.171821324961055 y final 0 7.192563835897883 z final 0 44.055167952062      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.171821 7.1986638 44.092531) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.171821 7.1925638 44.092531) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.171821 7.1925638 44.055168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.171821 7.1925638 44.055168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.171821 7.1925638 44.055168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.171821 7.1925638 44.055168) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.40807      1637.7255      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12432 ave       12432 max       12432 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12432
Ave neighs/atom = 77.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.17598961228999 y final 0 7.1955112601341105 z final 0 44.07322121828466      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.17599 7.1925638 44.055168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.17599 7.1955113 44.055168) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.17599 7.1955113 44.073221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.17599 7.1955113 44.073221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.17599 7.1955113 44.073221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.17599 7.1955113 44.073221) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.41186      1146.9954      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.17598961228999 y final 0 7.1955112601341105 z final 0 44.07322121828466      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.17599 7.1955113 44.073221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.17599 7.1955113 44.073221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.17599 7.1955113 44.073221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.17599 7.1955113 44.073221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.17599 7.1955113 44.073221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.17599 7.1955113 44.073221) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.41186      1146.9954      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.17140009466725 y final 0 7.192265981100694 z final 0 44.05334356184249      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.1714 7.1955113 44.073221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.1714 7.192266 44.073221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.1714 7.192266 44.053344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.1714 7.192266 44.053344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.1714 7.192266 44.053344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.1714 7.192266 44.053344) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.42111      1683.6041      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12448 ave       12448 max       12448 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12448
Ave neighs/atom = 77.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.153062714227024 y final 0 7.179299495042206 z final 0 43.97392254673749      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.153063 7.192266 44.053344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.153063 7.1792995 44.053344) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.153063 7.1792995 43.973923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.153063 7.1792995 43.973923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.153063 7.1792995 43.973923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.153063 7.1792995 43.973923) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.426        3922.9003      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12368 ave       12368 max       12368 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12368
Ave neighs/atom = 77.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.153062714227024 y final 0 7.179299495042206 z final 0 43.97392254673749      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.153063 7.1792995 43.973923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.153063 7.1792995 43.973923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.153063 7.1792995 43.973923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.153063 7.1792995 43.973923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.153063 7.1792995 43.973923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.153063 7.1792995 43.973923) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.426        3922.9003      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12368 ave       12368 max       12368 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12368
Ave neighs/atom = 77.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.159552548422301 y final 0 7.183888500810465 z final 0 44.00203066290665      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.159553 7.1792995 43.973923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159553 7.1838885 43.973923) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159553 7.1838885 44.002031) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159553 7.1838885 44.002031) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159553 7.1838885 44.002031) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159553 7.1838885 44.002031) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.43134      3187.9902      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12432 ave       12432 max       12432 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12432
Ave neighs/atom = 77.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.159552548422301 y final 0 7.183888500810465 z final 0 44.00203066290665      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.159553 7.1838885 44.002031) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159553 7.1838885 44.002031) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159553 7.1838885 44.002031) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159553 7.1838885 44.002031) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159553 7.1838885 44.002031) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159553 7.1838885 44.002031) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.43134      3187.9902      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12432 ave       12432 max       12432 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12432
Ave neighs/atom = 77.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.159753615750512 y final 0 7.184030676881718 z final 0 44.002901505465815      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.159754 7.1838885 44.002031) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159754 7.1840307 44.002031) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159754 7.1840307 44.002902) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159754 7.1840307 44.002902) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159754 7.1840307 44.002902) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.159754 7.1840307 44.002902) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.4 | 3.4 | 3.4 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-550.43143      3196.3201      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1210 ave        1210 max        1210 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12432 ave       12432 max       12432 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12432
Ave neighs/atom = 77.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.229763751952257 y final 0 7.229763751952257 z final 0 54.34137782177322      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2297638 7.1840307 44.002902) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2297638 7.2297638 44.002902) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2297638 7.2297638 54.341378) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2297638 7.2297638 54.341378) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2297638 7.2297638 54.341378) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2297638 7.2297638 54.341378) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.18473     -9009.2828      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 5.919229921211767 y final 0 5.919229921211767 z final 0 44.49095718731553      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.9192299 7.2297638 54.341378) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.9192299 5.9192299 54.341378) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.9192299 5.9192299 44.490957) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.9192299 5.9192299 44.490957) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.9192299 5.9192299 44.490957) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.9192299 5.9192299 44.490957) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-114.4325       1860603.7      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2090 ave        2090 max        2090 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          23056 ave       23056 max       23056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 23056
Ave neighs/atom = 144.1
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.214277418657984 y final 0 7.214277418657984 z final 0 54.22497724251629      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2142774 5.9192299 44.490957) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2142774 7.2142774 44.490957) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2142774 7.2142774 54.224977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2142774 7.2142774 54.224977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2142774 7.2142774 54.224977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2142774 7.2142774 54.224977) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.22734     -3017.2147      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.212649287094084 y final 0 7.212649287094084 z final 0 54.2127396486622      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2126493 7.2142774 54.224977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2126493 7.2126493 54.224977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2126493 7.2126493 54.21274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2126493 7.2126493 54.21274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2126493 7.2126493 54.21274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2126493 7.2126493 54.21274) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.23138     -2246.7351      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.212649287094084 y final 0 7.212649287094084 z final 0 54.2127396486622      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2126493 7.2126493 54.21274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2126493 7.2126493 54.21274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2126493 7.2126493 54.21274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2126493 7.2126493 54.21274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2126493 7.2126493 54.21274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2126493 7.2126493 54.21274) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.23138     -2246.7351      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.212397391033064 y final 0 7.212397391033064 z final 0 54.210846311688364      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2123974 7.2126493 54.21274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2123974 7.2123974 54.21274) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2123974 7.2123974 54.210846) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2123974 7.2123974 54.210846) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2123974 7.2123974 54.210846) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2123974 7.2123974 54.210846) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.23297     -1999.9131      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.212397391033064 y final 0 7.212397391033064 z final 0 54.210846311688364      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2123974 7.2123974 54.210846) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2123974 7.2123974 54.210846) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2123974 7.2123974 54.210846) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2123974 7.2123974 54.210846) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2123974 7.2123974 54.210846) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2123974 7.2123974 54.210846) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.23297     -1999.9131      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.212864163227584 y final 0 7.212864163227584 z final 0 54.214354731195485      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2128642 7.2123974 54.210846) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2128642 7.2128642 54.210846) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2128642 7.2128642 54.214355) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2128642 7.2128642 54.214355) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2128642 7.2128642 54.214355) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2128642 7.2128642 54.214355) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.23392     -2164.783       0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.213395756542864 y final 0 7.213395756542864 z final 0 54.21835036842296      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2133958 7.2128642 54.214355) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2133958 7.2133958 54.214355) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2133958 7.2133958 54.21835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2133958 7.2133958 54.21835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2133958 7.2133958 54.21835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2133958 7.2133958 54.21835) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.23422     -2351.0391      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.213395756542864 y final 0 7.213395756542864 z final 0 54.21835036842296      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2133958 7.2133958 54.21835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2133958 7.2133958 54.21835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2133958 7.2133958 54.21835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2133958 7.2133958 54.21835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2133958 7.2133958 54.21835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2133958 7.2133958 54.21835) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.23422     -2351.0391      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.213278362437853 y final 0 7.213278362437853 z final 0 54.21746799416658      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2132784 7.2133958 54.21835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2132784 7.2132784 54.21835) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2132784 7.2132784 54.217468) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2132784 7.2132784 54.217468) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2132784 7.2132784 54.217468) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2132784 7.2132784 54.217468) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.23524     -2395.8164      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.212952905164306 y final 0 7.212952905164305 z final 0 54.21502174595248      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2129529 7.2132784 54.217468) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2129529 7.2129529 54.217468) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2129529 7.2129529 54.215022) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2129529 7.2129529 54.215022) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2129529 7.2129529 54.215022) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2129529 7.2129529 54.215022) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.23644     -2516.2886      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.212952905164305 y final 0 7.212952905164304 z final 0 54.21502174595248      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2129529 7.2129529 54.215022) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2129529 7.2129529 54.215022) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2129529 7.2129529 54.215022) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2129529 7.2129529 54.215022) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2129529 7.2129529 54.215022) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2129529 7.2129529 54.215022) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.23644     -2516.2886      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 160 atoms

320.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.25e-06   |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.211891293889371 y final 0 7.21189129388937 z final 0 54.20704231240939      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2118913 7.2129529 54.215022) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2118913 7.2118913 54.215022) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2118913 7.2118913 54.207042) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2118913 7.2118913 54.207042) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2118913 7.2118913 54.207042) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2118913 7.2118913 54.207042) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.23889     -2364.8181      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.207646411052208 y final 0 7.207646411052204 z final 0 54.17513632073701      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2076464 7.2118913 54.207042) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2076464 7.2076464 54.207042) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2076464 7.2076464 54.175136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2076464 7.2076464 54.175136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2076464 7.2076464 54.175136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2076464 7.2076464 54.175136) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.23718     -1727.0917      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.210164156664795 y final 0 7.210164156664792 z final 0 54.19406055813764      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2101642 7.2076464 54.175136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2101642 7.2101642 54.175136) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2101642 7.2101642 54.194061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2101642 7.2101642 54.194061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2101642 7.2101642 54.194061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2101642 7.2101642 54.194061) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.24048     -2111.5848      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.210164156664795 y final 0 7.210164156664792 z final 0 54.19406055813764      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2101642 7.2101642 54.194061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2101642 7.2101642 54.194061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2101642 7.2101642 54.194061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2101642 7.2101642 54.194061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2101642 7.2101642 54.194061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2101642 7.2101642 54.194061) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.24048     -2111.5848      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.208374461536592 y final 0 7.208374461536589 z final 0 54.18060859725981      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2083745 7.2101642 54.194061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2083745 7.2083745 54.194061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2083745 7.2083745 54.180609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2083745 7.2083745 54.180609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2083745 7.2083745 54.180609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2083745 7.2083745 54.180609) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.24537     -1909.3732      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.201220122273158 y final 0 7.20122012227315 z final 0 54.126834135698246      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2012201 7.2083745 54.180609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2012201 7.2012201 54.180609) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2012201 7.2012201 54.126834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2012201 7.2012201 54.126834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2012201 7.2012201 54.126834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2012201 7.2012201 54.126834) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.25203     -1053.1826      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.201220122273158 y final 0 7.20122012227315 z final 0 54.12683413569825      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2012201 7.2012201 54.126834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2012201 7.2012201 54.126834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2012201 7.2012201 54.126834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2012201 7.2012201 54.126834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2012201 7.2012201 54.126834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2012201 7.2012201 54.126834) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.25203     -1053.1826      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 160 atoms

257.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.167e-06  |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.19900603839121 y final 0 7.199006038391199 z final 0 54.11019232375451      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.199006 7.2012201 54.126834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.199006 7.199006 54.126834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.199006 7.199006 54.110192) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.199006 7.199006 54.110192) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.199006 7.199006 54.110192) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.199006 7.199006 54.110192) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.26343     -1203.4858      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.196354815136089 y final 0 7.196354815136077 z final 0 54.09026482272697      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.1963548 7.199006 54.110192) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1963548 7.1963548 54.110192) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1963548 7.1963548 54.090265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1963548 7.1963548 54.090265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1963548 7.1963548 54.090265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1963548 7.1963548 54.090265) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.26816     -1358.5495      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.196354815136088 y final 0 7.196354815136077 z final 0 54.09026482272698      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.1963548 7.1963548 54.090265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1963548 7.1963548 54.090265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1963548 7.1963548 54.090265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1963548 7.1963548 54.090265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1963548 7.1963548 54.090265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1963548 7.1963548 54.090265) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.26816     -1358.5495      0              0            
Loop time of 2e-06 on 1 procs for 0 steps with 160 atoms

200.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 2e-06      |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12944 ave       12944 max       12944 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12944
Ave neighs/atom = 80.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.191236998482811 y final 0 7.191236998482797 z final 0 54.051797561842776      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.191237 7.1963548 54.090265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.191237 7.191237 54.090265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.191237 7.191237 54.051798) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.191237 7.191237 54.051798) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.191237 7.191237 54.051798) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.191237 7.191237 54.051798) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.2867      -1451.5864      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12816 ave       12816 max       12816 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12816
Ave neighs/atom = 80.1
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.176486938574027 y final 0 7.176486938574003 z final 0 53.94093106524693      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.1764869 7.191237 54.051798) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1764869 7.1764869 54.051798) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1764869 7.1764869 53.940931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1764869 7.1764869 53.940931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1764869 7.1764869 53.940931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1764869 7.1764869 53.940931) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.29783     -1519.849       0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 160 atoms

165.4% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.209e-06  |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12432 ave       12432 max       12432 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12432
Ave neighs/atom = 77.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.176486938574026 y final 0 7.176486938574002 z final 0 53.94093106524693      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.1764869 7.1764869 53.940931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1764869 7.1764869 53.940931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1764869 7.1764869 53.940931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1764869 7.1764869 53.940931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1764869 7.1764869 53.940931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1764869 7.1764869 53.940931) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.29783     -1519.849       0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12432 ave       12432 max       12432 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12432
Ave neighs/atom = 77.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.1722290946308895 y final 0 7.172229094630864 z final 0 53.90892765346775      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.1722291 7.1764869 53.940931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1722291 7.1722291 53.940931) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1722291 7.1722291 53.908928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1722291 7.1722291 53.908928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1722291 7.1722291 53.908928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1722291 7.1722291 53.908928) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.3114      -410.2936       0              0            
Loop time of 7.09e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12432 ave       12432 max       12432 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12432
Ave neighs/atom = 77.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.1722290946308895 y final 0 7.172229094630864 z final 0 53.90892765346775      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.1722291 7.1722291 53.908928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1722291 7.1722291 53.908928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1722291 7.1722291 53.908928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1722291 7.1722291 53.908928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1722291 7.1722291 53.908928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1722291 7.1722291 53.908928) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.3114      -410.2936       0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12432 ave       12432 max       12432 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12432
Ave neighs/atom = 77.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 7.170145766880595 y final 0 7.170145766880569 z final 0 53.893268649344066      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.1701458 7.1722291 53.908928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1701458 7.1701458 53.908928) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1701458 7.1701458 53.893269) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1701458 7.1701458 53.893269) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1701458 7.1701458 53.893269) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.1701458 7.1701458 53.893269) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.408 | 3.408 | 3.408 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-556.31227     -8.9159042      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1530 ave        1530 max        1530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12432 ave       12432 max       12432 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12432
Ave neighs/atom = 77.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.224429950764275 y final 0 8.85461607657634 z final 0 38.78347721765165      xy final 5.1122149753821375 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.22443 7.1701458 53.893269) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.22443 8.8546161 53.893269) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.22443 8.8546161 38.783477) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.22443 8.8546161 38.783477) with tilt (5.112215 0 0)
  triclinic box = (0 0 0) to (10.22443 8.8546161 38.783477) with tilt (5.112215 0 0)
  triclinic box = (0 0 0) to (10.22443 8.8546161 38.783477) with tilt (5.112215 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.39588     -9067.3971      0              0            
Loop time of 1.5e-06 on 1 procs for 0 steps with 160 atoms

200.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.5e-06    |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.224429950764275 y final 0 8.85461607657634 z final 0 38.783477217651644      xy final 5.1122149753821375 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.22443 8.8546161 38.783477) with tilt (5.112215 0 0)
  triclinic box = (0 0 0) to (10.22443 8.8546161 38.783477) with tilt (5.112215 0 0)
  triclinic box = (0 0 0) to (10.22443 8.8546161 38.783477) with tilt (5.112215 0 0)
  triclinic box = (0 0 0) to (10.22443 8.8546161 38.783477) with tilt (5.112215 0 0)
  triclinic box = (0 0 0) to (10.22443 8.8546161 38.783477) with tilt (5.112215 0 0)
  triclinic box = (0 0 0) to (10.22443 8.8546161 38.783477) with tilt (5.112215 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.39588     -9067.3971      0              0            
Loop time of 1.917e-06 on 1 procs for 0 steps with 160 atoms

260.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.917e-06  |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 8.371055233382307 y final 0 7.2495464885917515 z final 0 31.75322550939068      xy final 4.185527616691154 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.3710552 8.8546161 38.783477) with tilt (5.112215 0 0)
  triclinic box = (0 0 0) to (8.3710552 7.2495465 38.783477) with tilt (5.112215 0 0)
  triclinic box = (0 0 0) to (8.3710552 7.2495465 31.753226) with tilt (5.112215 0 0)
  triclinic box = (0 0 0) to (8.3710552 7.2495465 31.753226) with tilt (4.1855276 0 0)
  triclinic box = (0 0 0) to (8.3710552 7.2495465 31.753226) with tilt (4.1855276 0 0)
  triclinic box = (0 0 0) to (8.3710552 7.2495465 31.753226) with tilt (4.1855276 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.41 | 3.41 | 3.41 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-111.09758      1515608.1      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1380 ave        1380 max        1380 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          21232 ave       21232 max       21232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 21232
Ave neighs/atom = 132.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.195371524538414 y final 0 8.82945074127075 z final 0 38.67325231348208      xy final 5.097685762269207 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.195372 7.2495465 31.753226) with tilt (4.1855276 0 0)
  triclinic box = (0 0 0) to (10.195372 8.8294507 31.753226) with tilt (4.1855276 0 0)
  triclinic box = (0 0 0) to (10.195372 8.8294507 38.673252) with tilt (4.1855276 0 0)
  triclinic box = (0 0 0) to (10.195372 8.8294507 38.673252) with tilt (5.0976858 0 0)
  triclinic box = (0 0 0) to (10.195372 8.8294507 38.673252) with tilt (5.0976858 0 0)
  triclinic box = (0 0 0) to (10.195372 8.8294507 38.673252) with tilt (5.0976858 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.47188     -2394.326       0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.195371524538414 y final 0 8.82945074127075 z final 0 38.67325231348208      xy final 5.097685762269207 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.195372 8.8294507 38.673252) with tilt (5.0976858 0 0)
  triclinic box = (0 0 0) to (10.195372 8.8294507 38.673252) with tilt (5.0976858 0 0)
  triclinic box = (0 0 0) to (10.195372 8.8294507 38.673252) with tilt (5.0976858 0 0)
  triclinic box = (0 0 0) to (10.195372 8.8294507 38.673252) with tilt (5.0976858 0 0)
  triclinic box = (0 0 0) to (10.195372 8.8294507 38.673252) with tilt (5.0976858 0 0)
  triclinic box = (0 0 0) to (10.195372 8.8294507 38.673252) with tilt (5.0976858 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.47188     -2394.326       0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.19405407509216 y final 0 8.82830979658209 z final 0 38.668254942398285      xy final 5.09702703754608 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.194054 8.8294507 38.673252) with tilt (5.0976858 0 0)
  triclinic box = (0 0 0) to (10.194054 8.8283098 38.673252) with tilt (5.0976858 0 0)
  triclinic box = (0 0 0) to (10.194054 8.8283098 38.668255) with tilt (5.0976858 0 0)
  triclinic box = (0 0 0) to (10.194054 8.8283098 38.668255) with tilt (5.097027 0 0)
  triclinic box = (0 0 0) to (10.194054 8.8283098 38.668255) with tilt (5.097027 0 0)
  triclinic box = (0 0 0) to (10.194054 8.8283098 38.668255) with tilt (5.097027 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.4837      -1671.5356      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.19405407509216 y final 0 8.82830979658209 z final 0 38.668254942398285      xy final 5.09702703754608 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.194054 8.8283098 38.668255) with tilt (5.097027 0 0)
  triclinic box = (0 0 0) to (10.194054 8.8283098 38.668255) with tilt (5.097027 0 0)
  triclinic box = (0 0 0) to (10.194054 8.8283098 38.668255) with tilt (5.097027 0 0)
  triclinic box = (0 0 0) to (10.194054 8.8283098 38.668255) with tilt (5.097027 0 0)
  triclinic box = (0 0 0) to (10.194054 8.8283098 38.668255) with tilt (5.097027 0 0)
  triclinic box = (0 0 0) to (10.194054 8.8283098 38.668255) with tilt (5.097027 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.4837      -1671.5356      0              0            
Loop time of 1.291e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.19358337835798 y final 0 8.827902161252812 z final 0 38.6664694877417      xy final 5.09679168917899 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.193583 8.8283098 38.668255) with tilt (5.097027 0 0)
  triclinic box = (0 0 0) to (10.193583 8.8279022 38.668255) with tilt (5.097027 0 0)
  triclinic box = (0 0 0) to (10.193583 8.8279022 38.666469) with tilt (5.097027 0 0)
  triclinic box = (0 0 0) to (10.193583 8.8279022 38.666469) with tilt (5.0967917 0 0)
  triclinic box = (0 0 0) to (10.193583 8.8279022 38.666469) with tilt (5.0967917 0 0)
  triclinic box = (0 0 0) to (10.193583 8.8279022 38.666469) with tilt (5.0967917 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.48653     -1555.3716      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.19358337835798 y final 0 8.827902161252812 z final 0 38.6664694877417      xy final 5.09679168917899 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.193583 8.8279022 38.666469) with tilt (5.0967917 0 0)
  triclinic box = (0 0 0) to (10.193583 8.8279022 38.666469) with tilt (5.0967917 0 0)
  triclinic box = (0 0 0) to (10.193583 8.8279022 38.666469) with tilt (5.0967917 0 0)
  triclinic box = (0 0 0) to (10.193583 8.8279022 38.666469) with tilt (5.0967917 0 0)
  triclinic box = (0 0 0) to (10.193583 8.8279022 38.666469) with tilt (5.0967917 0 0)
  triclinic box = (0 0 0) to (10.193583 8.8279022 38.666469) with tilt (5.0967917 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.48653     -1555.3716      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.19359638580648 y final 0 8.82791342603365 z final 0 38.66651882781101      xy final 5.09679819290324 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.193596 8.8279022 38.666469) with tilt (5.0967917 0 0)
  triclinic box = (0 0 0) to (10.193596 8.8279134 38.666469) with tilt (5.0967917 0 0)
  triclinic box = (0 0 0) to (10.193596 8.8279134 38.666519) with tilt (5.0967917 0 0)
  triclinic box = (0 0 0) to (10.193596 8.8279134 38.666519) with tilt (5.0967982 0 0)
  triclinic box = (0 0 0) to (10.193596 8.8279134 38.666519) with tilt (5.0967982 0 0)
  triclinic box = (0 0 0) to (10.193596 8.8279134 38.666519) with tilt (5.0967982 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.48913     -1599.8214      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.193610156902563 y final 0 8.827925352152697 z final 0 38.66657106456101      xy final 5.096805078451282 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.19361 8.8279134 38.666519) with tilt (5.0967982 0 0)
  triclinic box = (0 0 0) to (10.19361 8.8279254 38.666519) with tilt (5.0967982 0 0)
  triclinic box = (0 0 0) to (10.19361 8.8279254 38.666571) with tilt (5.0967982 0 0)
  triclinic box = (0 0 0) to (10.19361 8.8279254 38.666571) with tilt (5.0968051 0 0)
  triclinic box = (0 0 0) to (10.19361 8.8279254 38.666571) with tilt (5.0968051 0 0)
  triclinic box = (0 0 0) to (10.19361 8.8279254 38.666571) with tilt (5.0968051 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.49006     -1645.1979      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.193610156902563 y final 0 8.827925352152697 z final 0 38.66657106456101      xy final 5.096805078451282 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.19361 8.8279254 38.666571) with tilt (5.0968051 0 0)
  triclinic box = (0 0 0) to (10.19361 8.8279254 38.666571) with tilt (5.0968051 0 0)
  triclinic box = (0 0 0) to (10.19361 8.8279254 38.666571) with tilt (5.0968051 0 0)
  triclinic box = (0 0 0) to (10.19361 8.8279254 38.666571) with tilt (5.0968051 0 0)
  triclinic box = (0 0 0) to (10.19361 8.8279254 38.666571) with tilt (5.0968051 0 0)
  triclinic box = (0 0 0) to (10.19361 8.8279254 38.666571) with tilt (5.0968051 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.49006     -1645.1979      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.192994987372016 y final 0 8.82739259971161 z final 0 38.664237593297784      xy final 5.096497493686008 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.192995 8.8279254 38.666571) with tilt (5.0968051 0 0)
  triclinic box = (0 0 0) to (10.192995 8.8273926 38.666571) with tilt (5.0968051 0 0)
  triclinic box = (0 0 0) to (10.192995 8.8273926 38.664238) with tilt (5.0968051 0 0)
  triclinic box = (0 0 0) to (10.192995 8.8273926 38.664238) with tilt (5.0964975 0 0)
  triclinic box = (0 0 0) to (10.192995 8.8273926 38.664238) with tilt (5.0964975 0 0)
  triclinic box = (0 0 0) to (10.192995 8.8273926 38.664238) with tilt (5.0964975 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.49251     -1616.4614      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.192227634171484 y final 0 8.826728052346272 z final 0 38.661326856414256      xy final 5.096113817085742 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.192228 8.8273926 38.664238) with tilt (5.0964975 0 0)
  triclinic box = (0 0 0) to (10.192228 8.8267281 38.664238) with tilt (5.0964975 0 0)
  triclinic box = (0 0 0) to (10.192228 8.8267281 38.661327) with tilt (5.0964975 0 0)
  triclinic box = (0 0 0) to (10.192228 8.8267281 38.661327) with tilt (5.0961138 0 0)
  triclinic box = (0 0 0) to (10.192228 8.8267281 38.661327) with tilt (5.0961138 0 0)
  triclinic box = (0 0 0) to (10.192228 8.8267281 38.661327) with tilt (5.0961138 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.49359     -1577.9203      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.192227634171484 y final 0 8.826728052346272 z final 0 38.661326856414256      xy final 5.096113817085742 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.192228 8.8267281 38.661327) with tilt (5.0961138 0 0)
  triclinic box = (0 0 0) to (10.192228 8.8267281 38.661327) with tilt (5.0961138 0 0)
  triclinic box = (0 0 0) to (10.192228 8.8267281 38.661327) with tilt (5.0961138 0 0)
  triclinic box = (0 0 0) to (10.192228 8.8267281 38.661327) with tilt (5.0961138 0 0)
  triclinic box = (0 0 0) to (10.192228 8.8267281 38.661327) with tilt (5.0961138 0 0)
  triclinic box = (0 0 0) to (10.192228 8.8267281 38.661327) with tilt (5.0961138 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.49359     -1577.9203      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.190103089389062 y final 0 8.824888142593219 z final 0 38.65326799791908      xy final 5.095051544694531 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.190103 8.8267281 38.661327) with tilt (5.0961138 0 0)
  triclinic box = (0 0 0) to (10.190103 8.8248881 38.661327) with tilt (5.0961138 0 0)
  triclinic box = (0 0 0) to (10.190103 8.8248881 38.653268) with tilt (5.0961138 0 0)
  triclinic box = (0 0 0) to (10.190103 8.8248881 38.653268) with tilt (5.0950515 0 0)
  triclinic box = (0 0 0) to (10.190103 8.8248881 38.653268) with tilt (5.0950515 0 0)
  triclinic box = (0 0 0) to (10.190103 8.8248881 38.653268) with tilt (5.0950515 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.49757     -1353.954       0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.185592039741056 y final 0 8.820981459000313 z final 0 38.63615660959781      xy final 5.092796019870528 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.185592 8.8248881 38.653268) with tilt (5.0950515 0 0)
  triclinic box = (0 0 0) to (10.185592 8.8209815 38.653268) with tilt (5.0950515 0 0)
  triclinic box = (0 0 0) to (10.185592 8.8209815 38.636157) with tilt (5.0950515 0 0)
  triclinic box = (0 0 0) to (10.185592 8.8209815 38.636157) with tilt (5.092796 0 0)
  triclinic box = (0 0 0) to (10.185592 8.8209815 38.636157) with tilt (5.092796 0 0)
  triclinic box = (0 0 0) to (10.185592 8.8209815 38.636157) with tilt (5.092796 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.50099     -866.17765      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.185592039741056 y final 0 8.820981459000313 z final 0 38.63615660959781      xy final 5.092796019870528 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.185592 8.8209815 38.636157) with tilt (5.092796 0 0)
  triclinic box = (0 0 0) to (10.185592 8.8209815 38.636157) with tilt (5.092796 0 0)
  triclinic box = (0 0 0) to (10.185592 8.8209815 38.636157) with tilt (5.092796 0 0)
  triclinic box = (0 0 0) to (10.185592 8.8209815 38.636157) with tilt (5.092796 0 0)
  triclinic box = (0 0 0) to (10.185592 8.8209815 38.636157) with tilt (5.092796 0 0)
  triclinic box = (0 0 0) to (10.185592 8.8209815 38.636157) with tilt (5.092796 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.50099     -866.17765      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.182158241790155 y final 0 8.81800770274337 z final 0 38.6231314702756      xy final 5.091079120895078 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.182158 8.8209815 38.636157) with tilt (5.092796 0 0)
  triclinic box = (0 0 0) to (10.182158 8.8180077 38.636157) with tilt (5.092796 0 0)
  triclinic box = (0 0 0) to (10.182158 8.8180077 38.623131) with tilt (5.092796 0 0)
  triclinic box = (0 0 0) to (10.182158 8.8180077 38.623131) with tilt (5.0910791 0 0)
  triclinic box = (0 0 0) to (10.182158 8.8180077 38.623131) with tilt (5.0910791 0 0)
  triclinic box = (0 0 0) to (10.182158 8.8180077 38.623131) with tilt (5.0910791 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.50704     -562.99945      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.17545255519139 y final 0 8.812200407799024 z final 0 38.59769534868403      xy final 5.087726277595695 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.175453 8.8180077 38.623131) with tilt (5.0910791 0 0)
  triclinic box = (0 0 0) to (10.175453 8.8122004 38.623131) with tilt (5.0910791 0 0)
  triclinic box = (0 0 0) to (10.175453 8.8122004 38.597695) with tilt (5.0910791 0 0)
  triclinic box = (0 0 0) to (10.175453 8.8122004 38.597695) with tilt (5.0877263 0 0)
  triclinic box = (0 0 0) to (10.175453 8.8122004 38.597695) with tilt (5.0877263 0 0)
  triclinic box = (0 0 0) to (10.175453 8.8122004 38.597695) with tilt (5.0877263 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.51171      49.916873      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.17545255519139 y final 0 8.812200407799024 z final 0 38.59769534868403      xy final 5.087726277595695 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.175453 8.8122004 38.597695) with tilt (5.0877263 0 0)
  triclinic box = (0 0 0) to (10.175453 8.8122004 38.597695) with tilt (5.0877263 0 0)
  triclinic box = (0 0 0) to (10.175453 8.8122004 38.597695) with tilt (5.0877263 0 0)
  triclinic box = (0 0 0) to (10.175453 8.8122004 38.597695) with tilt (5.0877263 0 0)
  triclinic box = (0 0 0) to (10.175453 8.8122004 38.597695) with tilt (5.0877263 0 0)
  triclinic box = (0 0 0) to (10.175453 8.8122004 38.597695) with tilt (5.0877263 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.51171      49.916873      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.175709669719229 y final 0 8.812423075511814 z final 0 38.59867063977416      xy final 5.087854834859614 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.17571 8.8122004 38.597695) with tilt (5.0877263 0 0)
  triclinic box = (0 0 0) to (10.17571 8.8124231 38.597695) with tilt (5.0877263 0 0)
  triclinic box = (0 0 0) to (10.17571 8.8124231 38.598671) with tilt (5.0877263 0 0)
  triclinic box = (0 0 0) to (10.17571 8.8124231 38.598671) with tilt (5.0878548 0 0)
  triclinic box = (0 0 0) to (10.17571 8.8124231 38.598671) with tilt (5.0878548 0 0)
  triclinic box = (0 0 0) to (10.17571 8.8124231 38.598671) with tilt (5.0878548 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.39 | 3.39 | 3.39 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-549.51173     -0.53263243     0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          12112 ave       12112 max       12112 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 12112
Ave neighs/atom = 75.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.224429950764275 y final 0 12.522318145101105 z final 0 34.01980878791254      xy final -10.224429950764275 xz final 0.0 yz final -0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.22443 8.8124231 38.598671) with tilt (5.0878548 0 0)
  triclinic box = (0 0 0) to (10.22443 12.522318 38.598671) with tilt (5.0878548 0 0)
  triclinic box = (0 0 0) to (10.22443 12.522318 34.019809) with tilt (5.0878548 0 0)
  triclinic box = (0 0 0) to (10.22443 12.522318 34.019809) with tilt (-10.22443 0 0)
  triclinic box = (0 0 0) to (10.22443 12.522318 34.019809) with tilt (-10.22443 0 0)
  triclinic box = (0 0 0) to (10.22443 12.522318 34.019809) with tilt (-10.22443 0 -0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.392 | 3.392 | 3.392 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-538.67495     -13491.407      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.224429950764275 y final 0 12.522318145101103 z final 0 34.01980878791254      xy final -10.224429950764275 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.22443 12.522318 34.019809) with tilt (-10.22443 0 -0)
  triclinic box = (0 0 0) to (10.22443 12.522318 34.019809) with tilt (-10.22443 0 -0)
  triclinic box = (0 0 0) to (10.22443 12.522318 34.019809) with tilt (-10.22443 0 -0)
  triclinic box = (0 0 0) to (10.22443 12.522318 34.019809) with tilt (-10.22443 0 -0)
  triclinic box = (0 0 0) to (10.22443 12.522318 34.019809) with tilt (-10.22443 0 -0)
  triclinic box = (0 0 0) to (10.22443 12.522318 34.019809) with tilt (-10.22443 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.392 | 3.392 | 3.392 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-538.67495     -13491.407      0              0            
Loop time of 1.584e-06 on 1 procs for 0 steps with 160 atoms

63.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.584e-06  |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 8.371055233382307 y final 0 10.252406965220704 z final 0 27.853063668496578      xy final -8.371055233382307 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.3710552 12.522318 34.019809) with tilt (-10.22443 0 0)
  triclinic box = (0 0 0) to (8.3710552 10.252407 34.019809) with tilt (-10.22443 0 0)
  triclinic box = (0 0 0) to (8.3710552 10.252407 27.853064) with tilt (-10.22443 0 0)
  triclinic box = (0 0 0) to (8.3710552 10.252407 27.853064) with tilt (-8.3710552 0 0)
  triclinic box = (0 0 0) to (8.3710552 10.252407 27.853064) with tilt (-8.3710552 0 0)
  triclinic box = (0 0 0) to (8.3710552 10.252407 27.853064) with tilt (-8.3710552 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.404 | 3.404 | 3.404 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-115.79542      1196182        0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1185 ave        1185 max        1185 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19296 ave       19296 max       19296 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19296
Ave neighs/atom = 120.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.170384703468887 y final 0 12.456126505652987 z final 0 33.839983703508      xy final -10.170384703468887 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.170385 10.252407 27.853064) with tilt (-8.3710552 0 0)
  triclinic box = (0 0 0) to (10.170385 12.456127 27.853064) with tilt (-8.3710552 0 0)
  triclinic box = (0 0 0) to (10.170385 12.456127 33.839984) with tilt (-8.3710552 0 0)
  triclinic box = (0 0 0) to (10.170385 12.456127 33.839984) with tilt (-10.170385 0 0)
  triclinic box = (0 0 0) to (10.170385 12.456127 33.839984) with tilt (-10.170385 0 0)
  triclinic box = (0 0 0) to (10.170385 12.456127 33.839984) with tilt (-10.170385 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.392 | 3.392 | 3.392 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-538.94214     -3327.6929      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 160 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.292e-06  |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.170384703468887 y final 0 12.456126505652986 z final 0 33.839983703508      xy final -10.170384703468887 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.170385 12.456127 33.839984) with tilt (-10.170385 0 0)
  triclinic box = (0 0 0) to (10.170385 12.456127 33.839984) with tilt (-10.170385 0 0)
  triclinic box = (0 0 0) to (10.170385 12.456127 33.839984) with tilt (-10.170385 0 0)
  triclinic box = (0 0 0) to (10.170385 12.456127 33.839984) with tilt (-10.170385 0 0)
  triclinic box = (0 0 0) to (10.170385 12.456127 33.839984) with tilt (-10.170385 0 0)
  triclinic box = (0 0 0) to (10.170385 12.456127 33.839984) with tilt (-10.170385 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.392 | 3.392 | 3.392 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-538.94214     -3327.6929      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 160 atoms

533.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.5e-07    |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.163977930263012 y final 0 12.448279843026922 z final 0 33.818666407535424      xy final -10.163977930263012 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.163978 12.456127 33.839984) with tilt (-10.170385 0 0)
  triclinic box = (0 0 0) to (10.163978 12.44828 33.839984) with tilt (-10.170385 0 0)
  triclinic box = (0 0 0) to (10.163978 12.44828 33.818666) with tilt (-10.170385 0 0)
  triclinic box = (0 0 0) to (10.163978 12.44828 33.818666) with tilt (-10.163978 0 0)
  triclinic box = (0 0 0) to (10.163978 12.44828 33.818666) with tilt (-10.163978 0 0)
  triclinic box = (0 0 0) to (10.163978 12.44828 33.818666) with tilt (-10.163978 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.392 | 3.392 | 3.392 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.18012     -1972.4672      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.163977930263012 y final 0 12.44827984302692 z final 0 33.818666407535424      xy final -10.163977930263012 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.163978 12.44828 33.818666) with tilt (-10.163978 0 0)
  triclinic box = (0 0 0) to (10.163978 12.44828 33.818666) with tilt (-10.163978 0 0)
  triclinic box = (0 0 0) to (10.163978 12.44828 33.818666) with tilt (-10.163978 0 0)
  triclinic box = (0 0 0) to (10.163978 12.44828 33.818666) with tilt (-10.163978 0 0)
  triclinic box = (0 0 0) to (10.163978 12.44828 33.818666) with tilt (-10.163978 0 0)
  triclinic box = (0 0 0) to (10.163978 12.44828 33.818666) with tilt (-10.163978 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.392 | 3.392 | 3.392 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.18012     -1972.4672      0              0            
Loop time of 5.84e-07 on 1 procs for 0 steps with 160 atoms

342.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.84e-07   |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.162822385401018 y final 0 12.44686459538353 z final 0 33.81482155599454      xy final -10.162822385401018 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.162822 12.44828 33.818666) with tilt (-10.163978 0 0)
  triclinic box = (0 0 0) to (10.162822 12.446865 33.818666) with tilt (-10.163978 0 0)
  triclinic box = (0 0 0) to (10.162822 12.446865 33.814822) with tilt (-10.163978 0 0)
  triclinic box = (0 0 0) to (10.162822 12.446865 33.814822) with tilt (-10.162822 0 0)
  triclinic box = (0 0 0) to (10.162822 12.446865 33.814822) with tilt (-10.162822 0 0)
  triclinic box = (0 0 0) to (10.162822 12.446865 33.814822) with tilt (-10.162822 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.27739     -1740.2692      0              0            
Loop time of 1.333e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.162822385401018 y final 0 12.44686459538353 z final 0 33.81482155599454      xy final -10.162822385401018 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.162822 12.446865 33.814822) with tilt (-10.162822 0 0)
  triclinic box = (0 0 0) to (10.162822 12.446865 33.814822) with tilt (-10.162822 0 0)
  triclinic box = (0 0 0) to (10.162822 12.446865 33.814822) with tilt (-10.162822 0 0)
  triclinic box = (0 0 0) to (10.162822 12.446865 33.814822) with tilt (-10.162822 0 0)
  triclinic box = (0 0 0) to (10.162822 12.446865 33.814822) with tilt (-10.162822 0 0)
  triclinic box = (0 0 0) to (10.162822 12.446865 33.814822) with tilt (-10.162822 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.27739     -1740.2692      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.165335765999286 y final 0 12.449942845381113 z final 0 33.823184342749016      xy final -10.165335765999286 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.165336 12.446865 33.814822) with tilt (-10.162822 0 0)
  triclinic box = (0 0 0) to (10.165336 12.449943 33.814822) with tilt (-10.162822 0 0)
  triclinic box = (0 0 0) to (10.165336 12.449943 33.823184) with tilt (-10.162822 0 0)
  triclinic box = (0 0 0) to (10.165336 12.449943 33.823184) with tilt (-10.165336 0 0)
  triclinic box = (0 0 0) to (10.165336 12.449943 33.823184) with tilt (-10.165336 0 0)
  triclinic box = (0 0 0) to (10.165336 12.449943 33.823184) with tilt (-10.165336 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.30452     -2334.7188      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.165335765999286 y final 0 12.449942845381113 z final 0 33.823184342749016      xy final -10.165335765999286 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.165336 12.449943 33.823184) with tilt (-10.165336 0 0)
  triclinic box = (0 0 0) to (10.165336 12.449943 33.823184) with tilt (-10.165336 0 0)
  triclinic box = (0 0 0) to (10.165336 12.449943 33.823184) with tilt (-10.165336 0 0)
  triclinic box = (0 0 0) to (10.165336 12.449943 33.823184) with tilt (-10.165336 0 0)
  triclinic box = (0 0 0) to (10.165336 12.449943 33.823184) with tilt (-10.165336 0 0)
  triclinic box = (0 0 0) to (10.165336 12.449943 33.823184) with tilt (-10.165336 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.30452     -2334.7188      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.165336639486705 y final 0 12.449943915180349 z final 0 33.82318724910908      xy final -10.165336639486705 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.165337 12.449943 33.823184) with tilt (-10.165336 0 0)
  triclinic box = (0 0 0) to (10.165337 12.449944 33.823184) with tilt (-10.165336 0 0)
  triclinic box = (0 0 0) to (10.165337 12.449944 33.823187) with tilt (-10.165336 0 0)
  triclinic box = (0 0 0) to (10.165337 12.449944 33.823187) with tilt (-10.165337 0 0)
  triclinic box = (0 0 0) to (10.165337 12.449944 33.823187) with tilt (-10.165337 0 0)
  triclinic box = (0 0 0) to (10.165337 12.449944 33.823187) with tilt (-10.165337 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.31962     -2557.5774      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.165336639486705 y final 0 12.449943915180349 z final 0 33.82318724910908      xy final -10.165336639486705 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.165337 12.449944 33.823187) with tilt (-10.165337 0 0)
  triclinic box = (0 0 0) to (10.165337 12.449944 33.823187) with tilt (-10.165337 0 0)
  triclinic box = (0 0 0) to (10.165337 12.449944 33.823187) with tilt (-10.165337 0 0)
  triclinic box = (0 0 0) to (10.165337 12.449944 33.823187) with tilt (-10.165337 0 0)
  triclinic box = (0 0 0) to (10.165337 12.449944 33.823187) with tilt (-10.165337 0 0)
  triclinic box = (0 0 0) to (10.165337 12.449944 33.823187) with tilt (-10.165337 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.31962     -2557.5774      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.162815661357783 y final 0 12.44685636014606 z final 0 33.81479918304404      xy final -10.162815661357783 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.162816 12.449944 33.823187) with tilt (-10.165337 0 0)
  triclinic box = (0 0 0) to (10.162816 12.446856 33.823187) with tilt (-10.165337 0 0)
  triclinic box = (0 0 0) to (10.162816 12.446856 33.814799) with tilt (-10.165337 0 0)
  triclinic box = (0 0 0) to (10.162816 12.446856 33.814799) with tilt (-10.162816 0 0)
  triclinic box = (0 0 0) to (10.162816 12.446856 33.814799) with tilt (-10.162816 0 0)
  triclinic box = (0 0 0) to (10.162816 12.446856 33.814799) with tilt (-10.162816 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.32869     -2411.8289      0              0            
Loop time of 6.67e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.162815661357783 y final 0 12.44685636014606 z final 0 33.81479918304404      xy final -10.162815661357783 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.162816 12.446856 33.814799) with tilt (-10.162816 0 0)
  triclinic box = (0 0 0) to (10.162816 12.446856 33.814799) with tilt (-10.162816 0 0)
  triclinic box = (0 0 0) to (10.162816 12.446856 33.814799) with tilt (-10.162816 0 0)
  triclinic box = (0 0 0) to (10.162816 12.446856 33.814799) with tilt (-10.162816 0 0)
  triclinic box = (0 0 0) to (10.162816 12.446856 33.814799) with tilt (-10.162816 0 0)
  triclinic box = (0 0 0) to (10.162816 12.446856 33.814799) with tilt (-10.162816 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.32869     -2411.8289      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11232 ave       11232 max       11232 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11232
Ave neighs/atom = 70.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.15840867921022 y final 0 12.441458931362515 z final 0 33.80013580418441      xy final -10.15840867921022 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.158409 12.446856 33.814799) with tilt (-10.162816 0 0)
  triclinic box = (0 0 0) to (10.158409 12.441459 33.814799) with tilt (-10.162816 0 0)
  triclinic box = (0 0 0) to (10.158409 12.441459 33.800136) with tilt (-10.162816 0 0)
  triclinic box = (0 0 0) to (10.158409 12.441459 33.800136) with tilt (-10.158409 0 0)
  triclinic box = (0 0 0) to (10.158409 12.441459 33.800136) with tilt (-10.158409 0 0)
  triclinic box = (0 0 0) to (10.158409 12.441459 33.800136) with tilt (-10.158409 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.33719     -2124.9196      0              0            
Loop time of 5.83e-07 on 1 procs for 0 steps with 160 atoms

171.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.83e-07   |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11216 ave       11216 max       11216 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11216
Ave neighs/atom = 70.1
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.15840867921022 y final 0 12.441458931362515 z final 0 33.80013580418441      xy final -10.15840867921022 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.158409 12.441459 33.800136) with tilt (-10.158409 0 0)
  triclinic box = (0 0 0) to (10.158409 12.441459 33.800136) with tilt (-10.158409 0 0)
  triclinic box = (0 0 0) to (10.158409 12.441459 33.800136) with tilt (-10.158409 0 0)
  triclinic box = (0 0 0) to (10.158409 12.441459 33.800136) with tilt (-10.158409 0 0)
  triclinic box = (0 0 0) to (10.158409 12.441459 33.800136) with tilt (-10.158409 0 0)
  triclinic box = (0 0 0) to (10.158409 12.441459 33.800136) with tilt (-10.158409 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.33719     -2124.9196      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11216 ave       11216 max       11216 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11216
Ave neighs/atom = 70.1
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.153699591189561 y final 0 12.435691499960276 z final 0 33.78446722659135      xy final -10.153699591189561 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.1537 12.441459 33.800136) with tilt (-10.158409 0 0)
  triclinic box = (0 0 0) to (10.1537 12.435691 33.800136) with tilt (-10.158409 0 0)
  triclinic box = (0 0 0) to (10.1537 12.435691 33.784467) with tilt (-10.158409 0 0)
  triclinic box = (0 0 0) to (10.1537 12.435691 33.784467) with tilt (-10.1537 0 0)
  triclinic box = (0 0 0) to (10.1537 12.435691 33.784467) with tilt (-10.1537 0 0)
  triclinic box = (0 0 0) to (10.1537 12.435691 33.784467) with tilt (-10.1537 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.34398     -1816.2185      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11216 ave       11216 max       11216 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11216
Ave neighs/atom = 70.1
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.148550801135627 y final 0 12.429385545747847 z final 0 33.76733562571312      xy final -10.148550801135627 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.148551 12.435691 33.784467) with tilt (-10.1537 0 0)
  triclinic box = (0 0 0) to (10.148551 12.429386 33.784467) with tilt (-10.1537 0 0)
  triclinic box = (0 0 0) to (10.148551 12.429386 33.767336) with tilt (-10.1537 0 0)
  triclinic box = (0 0 0) to (10.148551 12.429386 33.767336) with tilt (-10.148551 0 0)
  triclinic box = (0 0 0) to (10.148551 12.429386 33.767336) with tilt (-10.148551 0 0)
  triclinic box = (0 0 0) to (10.148551 12.429386 33.767336) with tilt (-10.148551 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.34606     -1466.4179      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11200 ave       11200 max       11200 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11200
Ave neighs/atom = 70
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.148550801135627 y final 0 12.429385545747847 z final 0 33.76733562571312      xy final -10.148550801135627 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.148551 12.429386 33.767336) with tilt (-10.148551 0 0)
  triclinic box = (0 0 0) to (10.148551 12.429386 33.767336) with tilt (-10.148551 0 0)
  triclinic box = (0 0 0) to (10.148551 12.429386 33.767336) with tilt (-10.148551 0 0)
  triclinic box = (0 0 0) to (10.148551 12.429386 33.767336) with tilt (-10.148551 0 0)
  triclinic box = (0 0 0) to (10.148551 12.429386 33.767336) with tilt (-10.148551 0 0)
  triclinic box = (0 0 0) to (10.148551 12.429386 33.767336) with tilt (-10.148551 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.34606     -1466.4179      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11200 ave       11200 max       11200 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11200
Ave neighs/atom = 70
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.147012060557493 y final 0 12.427500981116374 z final 0 33.76221576470431      xy final -10.147012060557493 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.147012 12.429386 33.767336) with tilt (-10.148551 0 0)
  triclinic box = (0 0 0) to (10.147012 12.427501 33.767336) with tilt (-10.148551 0 0)
  triclinic box = (0 0 0) to (10.147012 12.427501 33.762216) with tilt (-10.148551 0 0)
  triclinic box = (0 0 0) to (10.147012 12.427501 33.762216) with tilt (-10.147012 0 0)
  triclinic box = (0 0 0) to (10.147012 12.427501 33.762216) with tilt (-10.147012 0 0)
  triclinic box = (0 0 0) to (10.147012 12.427501 33.762216) with tilt (-10.147012 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.35269     -1431.2535      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11200 ave       11200 max       11200 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11200
Ave neighs/atom = 70
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.14300857907402 y final 0 12.422597737701778 z final 0 33.748894956091064      xy final -10.14300857907402 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.143009 12.427501 33.762216) with tilt (-10.147012 0 0)
  triclinic box = (0 0 0) to (10.143009 12.422598 33.762216) with tilt (-10.147012 0 0)
  triclinic box = (0 0 0) to (10.143009 12.422598 33.748895) with tilt (-10.147012 0 0)
  triclinic box = (0 0 0) to (10.143009 12.422598 33.748895) with tilt (-10.143009 0 0)
  triclinic box = (0 0 0) to (10.143009 12.422598 33.748895) with tilt (-10.143009 0 0)
  triclinic box = (0 0 0) to (10.143009 12.422598 33.748895) with tilt (-10.143009 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.35994     -1323.4624      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11200 ave       11200 max       11200 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11200
Ave neighs/atom = 70
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.14300857907402 y final 0 12.422597737701778 z final 0 33.748894956091064      xy final -10.14300857907402 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.143009 12.422598 33.748895) with tilt (-10.143009 0 0)
  triclinic box = (0 0 0) to (10.143009 12.422598 33.748895) with tilt (-10.143009 0 0)
  triclinic box = (0 0 0) to (10.143009 12.422598 33.748895) with tilt (-10.143009 0 0)
  triclinic box = (0 0 0) to (10.143009 12.422598 33.748895) with tilt (-10.143009 0 0)
  triclinic box = (0 0 0) to (10.143009 12.422598 33.748895) with tilt (-10.143009 0 0)
  triclinic box = (0 0 0) to (10.143009 12.422598 33.748895) with tilt (-10.143009 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.35994     -1323.4624      0              0            
Loop time of 1.75e-06 on 1 procs for 0 steps with 160 atoms

285.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.75e-06   |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11200 ave       11200 max       11200 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11200
Ave neighs/atom = 70
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.138199535024128 y final 0 12.416707885665375 z final 0 33.73289379418592      xy final -10.138199535024128 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.1382 12.422598 33.748895) with tilt (-10.143009 0 0)
  triclinic box = (0 0 0) to (10.1382 12.416708 33.748895) with tilt (-10.143009 0 0)
  triclinic box = (0 0 0) to (10.1382 12.416708 33.732894) with tilt (-10.143009 0 0)
  triclinic box = (0 0 0) to (10.1382 12.416708 33.732894) with tilt (-10.1382 0 0)
  triclinic box = (0 0 0) to (10.1382 12.416708 33.732894) with tilt (-10.1382 0 0)
  triclinic box = (0 0 0) to (10.1382 12.416708 33.732894) with tilt (-10.1382 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.36914     -1123.9264      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11184 ave       11184 max       11184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11184
Ave neighs/atom = 69.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.138199535024128 y final 0 12.416707885665375 z final 0 33.73289379418592      xy final -10.138199535024128 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.1382 12.416708 33.732894) with tilt (-10.1382 0 0)
  triclinic box = (0 0 0) to (10.1382 12.416708 33.732894) with tilt (-10.1382 0 0)
  triclinic box = (0 0 0) to (10.1382 12.416708 33.732894) with tilt (-10.1382 0 0)
  triclinic box = (0 0 0) to (10.1382 12.416708 33.732894) with tilt (-10.1382 0 0)
  triclinic box = (0 0 0) to (10.1382 12.416708 33.732894) with tilt (-10.1382 0 0)
  triclinic box = (0 0 0) to (10.1382 12.416708 33.732894) with tilt (-10.1382 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.36914     -1123.9264      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11184 ave       11184 max       11184 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11184
Ave neighs/atom = 69.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.133672257518294 y final 0 12.411163125758733 z final 0 33.71783015583934      xy final -10.133672257518294 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.133672 12.416708 33.732894) with tilt (-10.1382 0 0)
  triclinic box = (0 0 0) to (10.133672 12.411163 33.732894) with tilt (-10.1382 0 0)
  triclinic box = (0 0 0) to (10.133672 12.411163 33.71783) with tilt (-10.1382 0 0)
  triclinic box = (0 0 0) to (10.133672 12.411163 33.71783) with tilt (-10.133672 0 0)
  triclinic box = (0 0 0) to (10.133672 12.411163 33.71783) with tilt (-10.133672 0 0)
  triclinic box = (0 0 0) to (10.133672 12.411163 33.71783) with tilt (-10.133672 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37217     -869.79146      0              0            
Loop time of 5.42e-07 on 1 procs for 0 steps with 160 atoms

184.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.42e-07   |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11168 ave       11168 max       11168 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11168
Ave neighs/atom = 69.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.133672257518294 y final 0 12.411163125758732 z final 0 33.71783015583934      xy final -10.133672257518294 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.133672 12.411163 33.71783) with tilt (-10.133672 0 0)
  triclinic box = (0 0 0) to (10.133672 12.411163 33.71783) with tilt (-10.133672 0 0)
  triclinic box = (0 0 0) to (10.133672 12.411163 33.71783) with tilt (-10.133672 0 0)
  triclinic box = (0 0 0) to (10.133672 12.411163 33.71783) with tilt (-10.133672 0 0)
  triclinic box = (0 0 0) to (10.133672 12.411163 33.71783) with tilt (-10.133672 0 0)
  triclinic box = (0 0 0) to (10.133672 12.411163 33.71783) with tilt (-10.133672 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37217     -869.79146      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11168 ave       11168 max       11168 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11168
Ave neighs/atom = 69.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.131054669184502 y final 0 12.407957247871503 z final 0 33.709120638044226      xy final -10.131054669184502 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.131055 12.411163 33.71783) with tilt (-10.133672 0 0)
  triclinic box = (0 0 0) to (10.131055 12.407957 33.71783) with tilt (-10.133672 0 0)
  triclinic box = (0 0 0) to (10.131055 12.407957 33.709121) with tilt (-10.133672 0 0)
  triclinic box = (0 0 0) to (10.131055 12.407957 33.709121) with tilt (-10.131055 0 0)
  triclinic box = (0 0 0) to (10.131055 12.407957 33.709121) with tilt (-10.131055 0 0)
  triclinic box = (0 0 0) to (10.131055 12.407957 33.709121) with tilt (-10.131055 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37367     -771.21755      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.131054669184502 y final 0 12.407957247871503 z final 0 33.709120638044226      xy final -10.131054669184502 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.131055 12.407957 33.709121) with tilt (-10.131055 0 0)
  triclinic box = (0 0 0) to (10.131055 12.407957 33.709121) with tilt (-10.131055 0 0)
  triclinic box = (0 0 0) to (10.131055 12.407957 33.709121) with tilt (-10.131055 0 0)
  triclinic box = (0 0 0) to (10.131055 12.407957 33.709121) with tilt (-10.131055 0 0)
  triclinic box = (0 0 0) to (10.131055 12.407957 33.709121) with tilt (-10.131055 0 0)
  triclinic box = (0 0 0) to (10.131055 12.407957 33.709121) with tilt (-10.131055 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37367     -771.21755      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.130178109074393 y final 0 12.406883685372174 z final 0 33.70620405418766      xy final -10.130178109074393 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.130178 12.407957 33.709121) with tilt (-10.131055 0 0)
  triclinic box = (0 0 0) to (10.130178 12.406884 33.709121) with tilt (-10.131055 0 0)
  triclinic box = (0 0 0) to (10.130178 12.406884 33.706204) with tilt (-10.131055 0 0)
  triclinic box = (0 0 0) to (10.130178 12.406884 33.706204) with tilt (-10.130178 0 0)
  triclinic box = (0 0 0) to (10.130178 12.406884 33.706204) with tilt (-10.130178 0 0)
  triclinic box = (0 0 0) to (10.130178 12.406884 33.706204) with tilt (-10.130178 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.3747      -783.11355      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.126672626986537 y final 0 12.402590364163308 z final 0 33.69454024203261      xy final -10.126672626986537 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.126673 12.406884 33.706204) with tilt (-10.130178 0 0)
  triclinic box = (0 0 0) to (10.126673 12.40259 33.706204) with tilt (-10.130178 0 0)
  triclinic box = (0 0 0) to (10.126673 12.40259 33.69454) with tilt (-10.130178 0 0)
  triclinic box = (0 0 0) to (10.126673 12.40259 33.69454) with tilt (-10.126673 0 0)
  triclinic box = (0 0 0) to (10.126673 12.40259 33.69454) with tilt (-10.126673 0 0)
  triclinic box = (0 0 0) to (10.126673 12.40259 33.69454) with tilt (-10.126673 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37151     -820.00065      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.12936279525895 y final 0 12.405885133958137 z final 0 33.7034912555048      xy final -10.12936279525895 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.129363 12.40259 33.69454) with tilt (-10.126673 0 0)
  triclinic box = (0 0 0) to (10.129363 12.405885 33.69454) with tilt (-10.126673 0 0)
  triclinic box = (0 0 0) to (10.129363 12.405885 33.703491) with tilt (-10.126673 0 0)
  triclinic box = (0 0 0) to (10.129363 12.405885 33.703491) with tilt (-10.129363 0 0)
  triclinic box = (0 0 0) to (10.129363 12.405885 33.703491) with tilt (-10.129363 0 0)
  triclinic box = (0 0 0) to (10.129363 12.405885 33.703491) with tilt (-10.129363 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37503     -793.42965      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.12936279525895 y final 0 12.405885133958137 z final 0 33.7034912555048      xy final -10.12936279525895 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.129363 12.405885 33.703491) with tilt (-10.129363 0 0)
  triclinic box = (0 0 0) to (10.129363 12.405885 33.703491) with tilt (-10.129363 0 0)
  triclinic box = (0 0 0) to (10.129363 12.405885 33.703491) with tilt (-10.129363 0 0)
  triclinic box = (0 0 0) to (10.129363 12.405885 33.703491) with tilt (-10.129363 0 0)
  triclinic box = (0 0 0) to (10.129363 12.405885 33.703491) with tilt (-10.129363 0 0)
  triclinic box = (0 0 0) to (10.129363 12.405885 33.703491) with tilt (-10.129363 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37503     -793.42965      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.129702890913476 y final 0 12.406301664366799 z final 0 33.7046228578721      xy final -10.129702890913476 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.129703 12.405885 33.703491) with tilt (-10.129363 0 0)
  triclinic box = (0 0 0) to (10.129703 12.406302 33.703491) with tilt (-10.129363 0 0)
  triclinic box = (0 0 0) to (10.129703 12.406302 33.704623) with tilt (-10.129363 0 0)
  triclinic box = (0 0 0) to (10.129703 12.406302 33.704623) with tilt (-10.129703 0 0)
  triclinic box = (0 0 0) to (10.129703 12.406302 33.704623) with tilt (-10.129703 0 0)
  triclinic box = (0 0 0) to (10.129703 12.406302 33.704623) with tilt (-10.129703 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37585     -821.08487      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.130506539302027 y final 0 12.407285928609078 z final 0 33.707296842108185      xy final -10.130506539302027 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.130507 12.406302 33.704623) with tilt (-10.129703 0 0)
  triclinic box = (0 0 0) to (10.130507 12.407286 33.704623) with tilt (-10.129703 0 0)
  triclinic box = (0 0 0) to (10.130507 12.407286 33.707297) with tilt (-10.129703 0 0)
  triclinic box = (0 0 0) to (10.130507 12.407286 33.707297) with tilt (-10.130507 0 0)
  triclinic box = (0 0 0) to (10.130507 12.407286 33.707297) with tilt (-10.130507 0 0)
  triclinic box = (0 0 0) to (10.130507 12.407286 33.707297) with tilt (-10.130507 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37668     -885.21469      0              0            
Loop time of 1.5e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.130506539302027 y final 0 12.407285928609078 z final 0 33.707296842108185      xy final -10.130506539302027 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.130507 12.407286 33.707297) with tilt (-10.130507 0 0)
  triclinic box = (0 0 0) to (10.130507 12.407286 33.707297) with tilt (-10.130507 0 0)
  triclinic box = (0 0 0) to (10.130507 12.407286 33.707297) with tilt (-10.130507 0 0)
  triclinic box = (0 0 0) to (10.130507 12.407286 33.707297) with tilt (-10.130507 0 0)
  triclinic box = (0 0 0) to (10.130507 12.407286 33.707297) with tilt (-10.130507 0 0)
  triclinic box = (0 0 0) to (10.130507 12.407286 33.707297) with tilt (-10.130507 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37668     -885.21469      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.130728621302755 y final 0 12.407557922400494 z final 0 33.70803577691789      xy final -10.130728621302755 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.130729 12.407286 33.707297) with tilt (-10.130507 0 0)
  triclinic box = (0 0 0) to (10.130729 12.407558 33.707297) with tilt (-10.130507 0 0)
  triclinic box = (0 0 0) to (10.130729 12.407558 33.708036) with tilt (-10.130507 0 0)
  triclinic box = (0 0 0) to (10.130729 12.407558 33.708036) with tilt (-10.130729 0 0)
  triclinic box = (0 0 0) to (10.130729 12.407558 33.708036) with tilt (-10.130729 0 0)
  triclinic box = (0 0 0) to (10.130729 12.407558 33.708036) with tilt (-10.130729 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37762     -940.13973      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.131096132899142 y final 0 12.408008030343343 z final 0 33.70925860051751      xy final -10.131096132899142 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.131096 12.407558 33.708036) with tilt (-10.130729 0 0)
  triclinic box = (0 0 0) to (10.131096 12.408008 33.708036) with tilt (-10.130729 0 0)
  triclinic box = (0 0 0) to (10.131096 12.408008 33.709259) with tilt (-10.130729 0 0)
  triclinic box = (0 0 0) to (10.131096 12.408008 33.709259) with tilt (-10.131096 0 0)
  triclinic box = (0 0 0) to (10.131096 12.408008 33.709259) with tilt (-10.131096 0 0)
  triclinic box = (0 0 0) to (10.131096 12.408008 33.709259) with tilt (-10.131096 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37821     -1029.3668      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.131096132899142 y final 0 12.408008030343343 z final 0 33.70925860051751      xy final -10.131096132899142 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.131096 12.408008 33.709259) with tilt (-10.131096 0 0)
  triclinic box = (0 0 0) to (10.131096 12.408008 33.709259) with tilt (-10.131096 0 0)
  triclinic box = (0 0 0) to (10.131096 12.408008 33.709259) with tilt (-10.131096 0 0)
  triclinic box = (0 0 0) to (10.131096 12.408008 33.709259) with tilt (-10.131096 0 0)
  triclinic box = (0 0 0) to (10.131096 12.408008 33.709259) with tilt (-10.131096 0 0)
  triclinic box = (0 0 0) to (10.131096 12.408008 33.709259) with tilt (-10.131096 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.37821     -1029.3668      0              0            
Loop time of 2.208e-06 on 1 procs for 0 steps with 160 atoms

271.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 2.208e-06  |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.130345919006237 y final 0 12.407089209725553 z final 0 33.70676240920811      xy final -10.130345919006237 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.130346 12.408008 33.709259) with tilt (-10.131096 0 0)
  triclinic box = (0 0 0) to (10.130346 12.407089 33.709259) with tilt (-10.131096 0 0)
  triclinic box = (0 0 0) to (10.130346 12.407089 33.706762) with tilt (-10.131096 0 0)
  triclinic box = (0 0 0) to (10.130346 12.407089 33.706762) with tilt (-10.130346 0 0)
  triclinic box = (0 0 0) to (10.130346 12.407089 33.706762) with tilt (-10.130346 0 0)
  triclinic box = (0 0 0) to (10.130346 12.407089 33.706762) with tilt (-10.130346 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.38007     -1071.5077      0              0            
Loop time of 5.84e-07 on 1 procs for 0 steps with 160 atoms

171.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.84e-07   |            |       |100.00

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11136 ave       11136 max       11136 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11136
Ave neighs/atom = 69.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.128066702984327 y final 0 12.40429775159191 z final 0 33.699178759691705      xy final -10.128066702984327 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.128067 12.407089 33.706762) with tilt (-10.130346 0 0)
  triclinic box = (0 0 0) to (10.128067 12.404298 33.706762) with tilt (-10.130346 0 0)
  triclinic box = (0 0 0) to (10.128067 12.404298 33.699179) with tilt (-10.130346 0 0)
  triclinic box = (0 0 0) to (10.128067 12.404298 33.699179) with tilt (-10.128067 0 0)
  triclinic box = (0 0 0) to (10.128067 12.404298 33.699179) with tilt (-10.128067 0 0)
  triclinic box = (0 0 0) to (10.128067 12.404298 33.699179) with tilt (-10.128067 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.38246     -1193.3331      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11104 ave       11104 max       11104 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11104
Ave neighs/atom = 69.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.128066702984327 y final 0 12.40429775159191 z final 0 33.699178759691705      xy final -10.128066702984327 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.128067 12.404298 33.699179) with tilt (-10.128067 0 0)
  triclinic box = (0 0 0) to (10.128067 12.404298 33.699179) with tilt (-10.128067 0 0)
  triclinic box = (0 0 0) to (10.128067 12.404298 33.699179) with tilt (-10.128067 0 0)
  triclinic box = (0 0 0) to (10.128067 12.404298 33.699179) with tilt (-10.128067 0 0)
  triclinic box = (0 0 0) to (10.128067 12.404298 33.699179) with tilt (-10.128067 0 0)
  triclinic box = (0 0 0) to (10.128067 12.404298 33.699179) with tilt (-10.128067 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.38246     -1193.3331      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11104 ave       11104 max       11104 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11104
Ave neighs/atom = 69.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.127577110722301 y final 0 12.403698125979917 z final 0 33.69754973634093      xy final -10.127577110722301 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.127577 12.404298 33.699179) with tilt (-10.128067 0 0)
  triclinic box = (0 0 0) to (10.127577 12.403698 33.699179) with tilt (-10.128067 0 0)
  triclinic box = (0 0 0) to (10.127577 12.403698 33.69755) with tilt (-10.128067 0 0)
  triclinic box = (0 0 0) to (10.127577 12.403698 33.69755) with tilt (-10.127577 0 0)
  triclinic box = (0 0 0) to (10.127577 12.403698 33.69755) with tilt (-10.127577 0 0)
  triclinic box = (0 0 0) to (10.127577 12.403698 33.69755) with tilt (-10.127577 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.38373     -1219.2785      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11040 ave       11040 max       11040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11040
Ave neighs/atom = 69
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.127577110722301 y final 0 12.403698125979915 z final 0 33.69754973634093      xy final -10.127577110722301 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.127577 12.403698 33.69755) with tilt (-10.127577 0 0)
  triclinic box = (0 0 0) to (10.127577 12.403698 33.69755) with tilt (-10.127577 0 0)
  triclinic box = (0 0 0) to (10.127577 12.403698 33.69755) with tilt (-10.127577 0 0)
  triclinic box = (0 0 0) to (10.127577 12.403698 33.69755) with tilt (-10.127577 0 0)
  triclinic box = (0 0 0) to (10.127577 12.403698 33.69755) with tilt (-10.127577 0 0)
  triclinic box = (0 0 0) to (10.127577 12.403698 33.69755) with tilt (-10.127577 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.38373     -1219.2785      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11040 ave       11040 max       11040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11040
Ave neighs/atom = 69
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  160 settings made for x
  160 settings made for y
  160 settings made for z
change_box all     x final 0 10.12790058725481 y final 0 12.404094302204118 z final 0 33.69862604180113      xy final -10.12790058725481 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.127901 12.403698 33.69755) with tilt (-10.127577 0 0)
  triclinic box = (0 0 0) to (10.127901 12.404094 33.69755) with tilt (-10.127577 0 0)
  triclinic box = (0 0 0) to (10.127901 12.404094 33.698626) with tilt (-10.127577 0 0)
  triclinic box = (0 0 0) to (10.127901 12.404094 33.698626) with tilt (-10.127901 0 0)
  triclinic box = (0 0 0) to (10.127901 12.404094 33.698626) with tilt (-10.127901 0 0)
  triclinic box = (0 0 0) to (10.127901 12.404094 33.698626) with tilt (-10.127901 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.393 | 3.393 | 3.393 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-539.38432     -1284.7382      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 160 atoms

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

Nlocal:            160 ave         160 max         160 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            864 ave         864 max         864 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          11040 ave       11040 max       11040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 11040
Ave neighs/atom = 69
Neighbor list builds = 0
Dangerous builds = 0
Total wall time: 0:00:12
