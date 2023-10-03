LAMMPS (23 Jun 2022 - Update 4)
units metal
atom_style atomic
atom_modify map array sort 0 0
boundary p p p
box tilt large
region cell prism    0 2.845 0 2.845 0 2.845     0.0 0.0 0.0     units box
create_box 1 cell
Created triclinic box = (0 0 0) to (2.845 2.845 2.845) with tilt (0 0 0)
  1 by 1 by 1 MPI processor grid
echo none
Created 2 atoms
  using lattice units in triclinic box = (0 0 0) to (2.845 2.845 2.845) with tilt (0 0 0)
  create_atoms CPU = 0.000 seconds
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
pair_style eam/alloy
pair_coeff * * potentials/Cr.eam.alloy Cr
Reading eam/alloy potential file potentials/Cr.eam.alloy with DATE: 2018-03-30
mass 1 51.996099986624294331249984679744
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
  master list distance cutoff = 7.576483
  ghost atom cutoff = 7.576483
  binsize = 3.7882415, bins = 1 1 1
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair eam/alloy, perpetual
      attributes: half, newton on
      pair build: half/bin/newton/tri
      stencil: half/bin/3d/tri
      bin: standard
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.3688647      20198.343      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.9288018363564836 y final 0 2.9288018363564836 z final 0 2.9288018363564836      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9288018 2.845 2.845) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9288018 2.9288018 2.845) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9288018 2.9288018 2.9288018) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9288018 2.9288018 2.9288018) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9288018 2.9288018 2.9288018) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9288018 2.9288018 2.9288018) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.2923059     -125891.42      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.854758475660703 y final 0 2.854758475660703 z final 0 2.854758475660703      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8547585 2.9288018 2.9288018) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8547585 2.8547585 2.9288018) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8547585 2.8547585 2.8547585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8547585 2.8547585 2.8547585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8547585 2.8547585 2.8547585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8547585 2.8547585 2.8547585) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.3703462     -41.131606      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.569282628094633 y final 0 2.569282628094633 z final 0 2.569282628094633      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.5692826 2.8547585 2.8547585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.5692826 2.5692826 2.8547585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.5692826 2.5692826 2.5692826) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.5692826 2.5692826 2.5692826) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.5692826 2.5692826 2.5692826) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.5692826 2.5692826 2.5692826) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-6.7633524      1056190.7      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.59783021285124 y final 0 2.59783021285124 z final 0 2.59783021285124      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.5978302 2.5692826 2.5692826) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.5978302 2.5978302 2.5692826) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.5978302 2.5978302 2.5978302) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.5978302 2.5978302 2.5978302) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.5978302 2.5978302 2.5978302) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.5978302 2.5978302 2.5978302) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-7.1087269      883961.34      0              0            
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
change_box all     x final 0 2.626377797607847 y final 0 2.626377797607847 z final 0 2.626377797607847      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.6263778 2.5978302 2.5978302) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.6263778 2.6263778 2.5978302) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.6263778 2.6263778 2.6263778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.6263778 2.6263778 2.6263778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.6263778 2.6263778 2.6263778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.6263778 2.6263778 2.6263778) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-7.4027256      731955.11      0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.6549253823644543 y final 0 2.6549253823644543 z final 0 2.6549253823644543      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.6549254 2.6263778 2.6263778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.6549254 2.6549254 2.6263778) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.6549254 2.6549254 2.6549254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.6549254 2.6549254 2.6549254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.6549254 2.6549254 2.6549254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.6549254 2.6549254 2.6549254) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-7.6499821      598063.92      0              0            
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
change_box all     x final 0 2.683472967121061 y final 0 2.683472967121061 z final 0 2.683472967121061      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.683473 2.6549254 2.6549254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.683473 2.683473 2.6549254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.683473 2.683473 2.683473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.683473 2.683473 2.683473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.683473 2.683473 2.683473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.683473 2.683473 2.683473) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-7.8547455      480160.12      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.7120205518776683 y final 0 2.7120205518776683 z final 0 2.7120205518776683      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.7120206 2.683473 2.683473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7120206 2.7120206 2.683473) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7120206 2.7120206 2.7120206) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7120206 2.7120206 2.7120206) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7120206 2.7120206 2.7120206) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7120206 2.7120206 2.7120206) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.0208234      376148.67      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.740568136634275 y final 0 2.740568136634275 z final 0 2.740568136634275      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.7405681 2.7120206 2.7120206) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7405681 2.7405681 2.7120206) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7405681 2.7405681 2.7405681) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7405681 2.7405681 2.7405681) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7405681 2.7405681 2.7405681) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7405681 2.7405681 2.7405681) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.1515433      284015.35      0              0            
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
change_box all     x final 0 2.7691157213908824 y final 0 2.7691157213908824 z final 0 2.7691157213908824      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.7691157 2.7405681 2.7405681) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7691157 2.7691157 2.7405681) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7691157 2.7691157 2.7691157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7691157 2.7691157 2.7691157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7691157 2.7691157 2.7691157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7691157 2.7691157 2.7691157) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.249735       201877.78      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.7976633061474896 y final 0 2.7976633061474896 z final 0 2.7976633061474896      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.7976633 2.7691157 2.7691157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7976633 2.7976633 2.7691157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7976633 2.7976633 2.7976633) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7976633 2.7976633 2.7976633) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7976633 2.7976633 2.7976633) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.7976633 2.7976633 2.7976633) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.3177373      128043.99      0              0            
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
change_box all     x final 0 2.8262108909040964 y final 0 2.8262108909040964 z final 0 2.8262108909040964      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8262109 2.7976633 2.7976633) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8262109 2.8262109 2.7976633) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8262109 2.8262109 2.8262109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8262109 2.8262109 2.8262109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8262109 2.8262109 2.8262109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8262109 2.8262109 2.8262109) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.357437       61078.37       0              0            
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
change_box all     x final 0 2.854758475660703 y final 0 2.854758475660703 z final 0 2.854758475660703      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8547585 2.8262109 2.8262109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8547585 2.8547585 2.8262109) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8547585 2.8547585 2.8547585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8547585 2.8547585 2.8547585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8547585 2.8547585 2.8547585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8547585 2.8547585 2.8547585) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.3703462     -41.131606      0              0            
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
change_box all     x final 0 2.883306060417311 y final 0 2.883306060417311 z final 0 2.883306060417311      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8833061 2.8547585 2.8547585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8833061 2.8833061 2.8547585) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8833061 2.8833061 2.8833061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8833061 2.8833061 2.8833061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8833061 2.8833061 2.8833061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8833061 2.8833061 2.8833061) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.3581485     -54006.568      0              0            
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
change_box all     x final 0 2.911853645173917 y final 0 2.911853645173917 z final 0 2.911853645173917      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9118536 2.8833061 2.8833061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9118536 2.9118536 2.8833061) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9118536 2.9118536 2.9118536) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9118536 2.9118536 2.9118536) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9118536 2.9118536 2.9118536) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9118536 2.9118536 2.9118536) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.3230727     -101043.96      0              0            
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
change_box all     x final 0 2.940401229930525 y final 0 2.940401229930525 z final 0 2.940401229930525      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9404012 2.9118536 2.9118536) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9404012 2.9404012 2.9118536) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9404012 2.9404012 2.9404012) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9404012 2.9404012 2.9404012) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9404012 2.9404012 2.9404012) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9404012 2.9404012 2.9404012) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.2672714     -141596.84      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.9689488146871312 y final 0 2.9689488146871312 z final 0 2.9689488146871312      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9689488 2.9404012 2.9404012) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9689488 2.9689488 2.9404012) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9689488 2.9689488 2.9689488) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9689488 2.9689488 2.9689488) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9689488 2.9689488 2.9689488) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9689488 2.9689488 2.9689488) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.1929551     -175709.03      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.997496399443739 y final 0 2.997496399443739 z final 0 2.997496399443739      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9974964 2.9689488 2.9689488) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9974964 2.9974964 2.9689488) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9974964 2.9974964 2.9974964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9974964 2.9974964 2.9974964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9974964 2.9974964 2.9974964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9974964 2.9974964 2.9974964) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-8.1025424     -203178.68      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.0260439842003457 y final 0 3.0260439842003457 z final 0 3.0260439842003457      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.026044 2.9974964 2.9974964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.026044 3.026044 2.9974964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.026044 3.026044 3.026044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.026044 3.026044 3.026044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.026044 3.026044 3.026044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.026044 3.026044 3.026044) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-7.9985637     -225661.66      0              0            
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
change_box all     x final 0 3.0545915689569534 y final 0 3.0545915689569534 z final 0 3.0545915689569534      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0545916 3.026044 3.026044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0545916 3.0545916 3.026044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0545916 3.0545916 3.0545916) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0545916 3.0545916 3.0545916) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0545916 3.0545916 3.0545916) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0545916 3.0545916 3.0545916) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-7.8809441     -250756.86      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.0831391537135597 y final 0 3.0831391537135597 z final 0 3.0831391537135597      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0831392 3.0545916 3.0545916) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0831392 3.0831392 3.0545916) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0831392 3.0831392 3.0831392) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0831392 3.0831392 3.0831392) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0831392 3.0831392 3.0831392) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0831392 3.0831392 3.0831392) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-7.7480563     -277113.91      0              0            
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
change_box all     x final 0 3.1116867384701674 y final 0 3.1116867384701674 z final 0 3.1116867384701674      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1116867 3.0831392 3.0831392) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1116867 3.1116867 3.0831392) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1116867 3.1116867 3.1116867) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1116867 3.1116867 3.1116867) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1116867 3.1116867 3.1116867) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1116867 3.1116867 3.1116867) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-7.5992972     -302666.41      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.1402343232267738 y final 0 3.1402343232267738 z final 0 3.1402343232267738      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1402343 3.1116867 3.1116867) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1402343 3.1402343 3.1116867) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1402343 3.1402343 3.1402343) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1402343 3.1402343 3.1402343) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1402343 3.1402343 3.1402343) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1402343 3.1402343 3.1402343) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-7.4349975     -325867.88      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 8.56427542698211 y final 0 8.56427542698211 z final 0 8.56427542698211      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5642754 3.1402343 3.1402343) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642754 3.1402343) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642754 8.5642754) with tilt (0 0 0)
echo none
Created 52 atoms
  using lattice units in triclinic box = (0 0 0) to (8.5642754 8.5642754 8.5642754) with tilt (0 0 0)
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
-225.99935     -41.131606      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4536 ave        4536 max        4536 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4536
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 8.392989918442469 y final 0 8.56427542698211 z final 0 8.56427542698211      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.3929899 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.3929899 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.3929899 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.3929899 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.3929899 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.3929899 8.5642754 8.5642754) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-225.76791      58122.114      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4536 ave        4536 max        4536 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4536
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 8.478632672712289 y final 0 8.56427542698211 z final 0 8.56427542698211      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.4786327 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.4786327 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.4786327 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.4786327 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.4786327 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.4786327 8.5642754 8.5642754) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-225.94069      29768.133      0              0            
Loop time of 1.875e-06 on 1 procs for 0 steps with 54 atoms

160.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.875e-06  |            |       |100.00

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4536 ave        4536 max        4536 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4536
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 8.56427542698211 y final 0 8.56427542698211 z final 0 8.56427542698211      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5642754 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642754 8.5642754) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-225.99935     -41.131606      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4536 ave        4536 max        4536 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4536
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 8.649918181251932 y final 0 8.56427542698211 z final 0 8.56427542698211      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.6499182 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.6499182 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.6499182 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.6499182 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.6499182 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.6499182 8.5642754 8.5642754) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-225.93887     -30988.622      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4536 ave        4536 max        4536 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4536
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 8.735560935521752 y final 0 8.56427542698211 z final 0 8.56427542698211      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.7355609 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.7355609 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.7355609 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.7355609 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.7355609 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.7355609 8.5642754 8.5642754) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-225.75527     -62753.77       0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4536 ave        4536 max        4536 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4536
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 8.56427542698211 y final 0 8.564292555515836 z final 0 8.564258298482642      xy final 0.0 xz final 0.0 yz final 0.017128516596965283 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5642754 8.5642754 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642926 8.5642754) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642926 8.5642583) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642926 8.5642583) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642926 8.5642583) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.5642754 8.5642926 8.5642583) with tilt (0 0 0.017128517)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-225.99839     -30.978842      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4536 ave        4536 max        4536 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4536
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 8.56427542698211 y final 0 8.564456345389575 z final 0 8.564094512396426      xy final 0.0 xz final 0.0 yz final 0.05566661433057678 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5642754 8.5642926 8.5642583) with tilt (0 0 0.017128517)
  triclinic box = (0 0 0) to (8.5642754 8.5644563 8.5642583) with tilt (0 0 0.017128517)
  triclinic box = (0 0 0) to (8.5642754 8.5644563 8.5640945) with tilt (0 0 0.017128517)
  triclinic box = (0 0 0) to (8.5642754 8.5644563 8.5640945) with tilt (0 0 0.017128517)
  triclinic box = (0 0 0) to (8.5642754 8.5644563 8.5640945) with tilt (0 0 0.017128517)
  triclinic box = (0 0 0) to (8.5642754 8.5644563 8.5640945) with tilt (0 0 0.055666614)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-225.98918      66.16751       0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4536 ave        4536 max        4536 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4536
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 8.56427542698211 y final 0 8.564793549972697 z final 0 8.56375733533512      xy final 0.0 xz final 0.0 yz final 0.09420133068868636 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5642754 8.5644563 8.5640945) with tilt (0 0 0.055666614)
  triclinic box = (0 0 0) to (8.5642754 8.5647935 8.5640945) with tilt (0 0 0.055666614)
  triclinic box = (0 0 0) to (8.5642754 8.5647935 8.5637573) with tilt (0 0 0.055666614)
  triclinic box = (0 0 0) to (8.5642754 8.5647935 8.5637573) with tilt (0 0 0.055666614)
  triclinic box = (0 0 0) to (8.5642754 8.5647935 8.5637573) with tilt (0 0 0.055666614)
  triclinic box = (0 0 0) to (8.5642754 8.5647935 8.5637573) with tilt (0 0 0.094201331)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-225.97023      266.14585      0              0            
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
Neighs:           4536 ave        4536 max        4536 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4536
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 8.56427542698211 y final 0 8.565304148783884 z final 0 8.56324682873328      xy final 0.0 xz final 0.0 yz final 0.13273032584536584 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5642754 8.5647935 8.5637573) with tilt (0 0 0.094201331)
  triclinic box = (0 0 0) to (8.5642754 8.5653041 8.5637573) with tilt (0 0 0.094201331)
  triclinic box = (0 0 0) to (8.5642754 8.5653041 8.5632468) with tilt (0 0 0.094201331)
  triclinic box = (0 0 0) to (8.5642754 8.5653041 8.5632468) with tilt (0 0 0.094201331)
  triclinic box = (0 0 0) to (8.5642754 8.5653041 8.5632468) with tilt (0 0 0.094201331)
  triclinic box = (0 0 0) to (8.5642754 8.5653041 8.5632468) with tilt (0 0 0.13273033)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-225.94154      568.93478      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4536 ave        4536 max        4536 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4536
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 8.56427542698211 y final 0 8.565988110816246 z final 0 8.562563085582013      xy final 0.0 xz final 0.0 yz final 0.17125126171164032 units box
Changing box ...
  triclinic box = (0 0 0) to (8.5642754 8.5653041 8.5632468) with tilt (0 0 0.13273033)
  triclinic box = (0 0 0) to (8.5642754 8.5659881 8.5632468) with tilt (0 0 0.13273033)
  triclinic box = (0 0 0) to (8.5642754 8.5659881 8.5625631) with tilt (0 0 0.13273033)
  triclinic box = (0 0 0) to (8.5642754 8.5659881 8.5625631) with tilt (0 0 0.13273033)
  triclinic box = (0 0 0) to (8.5642754 8.5659881 8.5625631) with tilt (0 0 0.13273033)
  triclinic box = (0 0 0) to (8.5642754 8.5659881 8.5625631) with tilt (0 0 0.17125126)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-225.9031       974.49773      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1187 ave        1187 max        1187 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4536 ave        4536 max        4536 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4536
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 11.419033902642813 y final 0 11.419033902642813 z final 0 11.419033902642813      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.419034 8.5659881 8.5625631) with tilt (0 0 0.17125126)
  triclinic box = (0 0 0) to (11.419034 11.419034 8.5625631) with tilt (0 0 0.17125126)
  triclinic box = (0 0 0) to (11.419034 11.419034 11.419034) with tilt (0 0 0.17125126)
  triclinic box = (0 0 0) to (11.419034 11.419034 11.419034) with tilt (0 0 0.17125126)
  triclinic box = (0 0 0) to (11.419034 11.419034 11.419034) with tilt (0 0 0.17125126)
  triclinic box = (0 0 0) to (11.419034 11.419034 11.419034) with tilt (0 0 0)
echo none
Created 74 atoms
  using lattice units in triclinic box = (0 0 0) to (11.419034 11.419034 11.419034) with tilt (0 0 0)
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
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.257 | 3.257 | 3.257 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-535.70158     -43.066918      0              0            
Loop time of 1.5e-06 on 1 procs for 0 steps with 128 atoms

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

Nlocal:            128 ave         128 max         128 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1601 ave        1601 max        1601 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          10752 ave       10752 max       10752 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 10752
Ave neighs/atom = 84
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  128 settings made for x
  128 settings made for y
  128 settings made for z
change_box all     x final 0 11.419033902642813 y final 0 14.273792378303515 z final 0 17.12855085396422      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.419034 11.419034 11.419034) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 11.419034) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
echo none
Created 111 atoms
  using lattice units in triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
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
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.09408     -210.69394      0              0            
Loop time of 1.458e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.007398770526667 y final 0 13.759248463158332 z final 0 16.511098155790002      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.007399 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.007399 13.759248 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.007399 13.759248 16.511098) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.007399 13.759248 16.511098) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.007399 13.759248 16.511098) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.007399 13.759248 16.511098) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-977.3244       247543.06      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 239 atoms

72.7% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.417262757019593 y final 0 14.27157844627449 z final 0 17.12589413552939      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417263 13.759248 16.511098) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417263 14.271578 16.511098) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417263 14.271578 17.125894) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417263 14.271578 17.125894) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417263 14.271578 17.125894) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417263 14.271578 17.125894) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.09446      673.42548      0              0            
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
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.417262757019593 y final 0 14.27157844627449 z final 0 17.12589413552939      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417263 14.271578 17.125894) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417263 14.271578 17.125894) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417263 14.271578 17.125894) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417263 14.271578 17.125894) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417263 14.271578 17.125894) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417263 14.271578 17.125894) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.09446      673.42548      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.420305518094343 y final 0 14.27538189761793 z final 0 17.130458277141518      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.420306 14.271578 17.125894) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.420306 14.275382 17.125894) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.420306 14.275382 17.130458) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.420306 14.275382 17.130458) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.420306 14.275382 17.130458) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.420306 14.275382 17.130458) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-997.78241     -1714.6526      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 239 atoms

461.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.084e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.418241861457213 y final 0 14.272802326821516 z final 0 17.12736279218582      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.418242 14.275382 17.130458) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418242 14.272802 17.130458) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418242 14.272802 17.127363) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418242 14.272802 17.127363) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418242 14.272802 17.127363) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418242 14.272802 17.127363) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.18548     -63.690177      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.418241861457213 y final 0 14.272802326821516 z final 0 17.12736279218582      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.418242 14.272802 17.127363) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418242 14.272802 17.127363) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418242 14.272802 17.127363) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418242 14.272802 17.127363) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418242 14.272802 17.127363) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418242 14.272802 17.127363) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.18548     -63.690177      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.417963869654397 y final 0 14.272454837067995 z final 0 17.126945804481597      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417964 14.272802 17.127363) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417964 14.272455 17.127363) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417964 14.272455 17.126946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417964 14.272455 17.126946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417964 14.272455 17.126946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417964 14.272455 17.126946) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.19684     -88.074501      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.417963869654397 y final 0 14.272454837067995 z final 0 17.126945804481597      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417964 14.272455 17.126946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417964 14.272455 17.126946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417964 14.272455 17.126946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417964 14.272455 17.126946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417964 14.272455 17.126946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417964 14.272455 17.126946) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.19684     -88.074501      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.417818861718464 y final 0 14.27227357714808 z final 0 17.1267282925777      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417819 14.272455 17.126946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417819 14.272274 17.126946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417819 14.272274 17.126728) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417819 14.272274 17.126728) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417819 14.272274 17.126728) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417819 14.272274 17.126728) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.19892     -14.389573      0              0            
Loop time of 1.291e-06 on 1 procs for 0 steps with 239 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.291e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.417818861718464 y final 0 14.27227357714808 z final 0 17.1267282925777      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417819 14.272274 17.126728) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417819 14.272274 17.126728) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417819 14.272274 17.126728) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417819 14.272274 17.126728) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417819 14.272274 17.126728) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417819 14.272274 17.126728) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.19892     -14.389573      0              0            
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
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.417758751206668 y final 0 14.272198439008335 z final 0 17.126638126810004      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417759 14.272274 17.126728) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417759 14.272198 17.126728) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417759 14.272198 17.126638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417759 14.272198 17.126638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417759 14.272198 17.126638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417759 14.272198 17.126638) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.1992       3.7895624      0              0            
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
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.417758751206668 y final 0 14.272198439008335 z final 0 17.126638126810004      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417759 14.272198 17.126638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417759 14.272198 17.126638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417759 14.272198 17.126638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417759 14.272198 17.126638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417759 14.272198 17.126638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417759 14.272198 17.126638) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.1992       3.7895624      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.417748196273308 y final 0 14.272185245341637 z final 0 17.126622294409966      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417748 14.272198 17.126638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417748 14.272185 17.126638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417748 14.272185 17.126622) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417748 14.272185 17.126622) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417748 14.272185 17.126622) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417748 14.272185 17.126622) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.1993       2.5000782      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.417748196273308 y final 0 14.272185245341637 z final 0 17.126622294409966      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417748 14.272185 17.126622) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417748 14.272185 17.126622) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417748 14.272185 17.126622) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417748 14.272185 17.126622) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417748 14.272185 17.126622) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417748 14.272185 17.126622) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.1993       2.5000782      0              0            
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
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.417748632239292 y final 0 14.272185790299117 z final 0 17.126622948358946      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417749 14.272185 17.126622) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417749 14.272186 17.126622) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417749 14.272186 17.126623) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417749 14.272186 17.126623) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417749 14.272186 17.126623) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417749 14.272186 17.126623) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-998.19933     -4.5569026      0              0            
Loop time of 6.25e-07 on 1 procs for 0 steps with 239 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2063 ave        2063 max        2063 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19992 ave       19992 max       19992 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19992
Ave neighs/atom = 83.648536
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 11.419033902642813 y final 0 14.273792378303515 z final 0 17.12855085396422      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.419034 14.272186 17.126623) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.126623) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
echo none
1 atoms in group delatoms
Deleted 1 atoms, new total = 238
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.06559     -125.42671      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 238 atoms

545.3% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.172168023390157 y final 0 13.965210029237694 z final 0 16.758252035085235      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.172168 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.172168 13.96521 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.172168 13.96521 16.758252) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.172168 13.96521 16.758252) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.172168 13.96521 16.758252) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.172168 13.96521 16.758252) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-984.97187      136339.94      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.415625062816435 y final 0 14.269531328520543 z final 0 17.123437594224654      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.415625 13.96521 16.758252) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415625 14.269531 16.758252) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415625 14.269531 17.123438) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415625 14.269531 17.123438) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415625 14.269531 17.123438) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415625 14.269531 17.123438) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.06697      1561.0393      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.415625062816435 y final 0 14.269531328520543 z final 0 17.123437594224654      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.415625 14.269531 17.123438) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415625 14.269531 17.123438) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415625 14.269531 17.123438) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415625 14.269531 17.123438) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415625 14.269531 17.123438) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415625 14.269531 17.123438) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.06697      1561.0393      0              0            
Loop time of 1.417e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.428600053953385 y final 0 14.285750067442386 z final 0 17.14290008093087      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.4286 14.269531 17.123438) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.4286 14.28575 17.123438) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.4286 14.28575 17.1429) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.4286 14.28575 17.1429) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.4286 14.28575 17.1429) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.4286 14.28575 17.1429) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-990.74059     -5985.8486      0              0            
Loop time of 1.458e-06 on 1 procs for 0 steps with 238 atoms

68.6% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19788 ave       19788 max       19788 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19788
Ave neighs/atom = 83.142857
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.418543316965312 y final 0 14.273179146206788 z final 0 17.127814975448146      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.418543 14.28575 17.1429) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418543 14.273179 17.1429) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418543 14.273179 17.127815) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418543 14.273179 17.127815) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418543 14.273179 17.127815) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418543 14.273179 17.127815) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.19012     -131.12727      0              0            
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
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.418543316965312 y final 0 14.273179146206788 z final 0 17.127814975448146      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.418543 14.273179 17.127815) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418543 14.273179 17.127815) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418543 14.273179 17.127815) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418543 14.273179 17.127815) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418543 14.273179 17.127815) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418543 14.273179 17.127815) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.19012     -131.12727      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 238 atoms

640.0% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.418078210581553 y final 0 14.272597763227115 z final 0 17.12711731587254      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.418078 14.273179 17.127815) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418078 14.272598 17.127815) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418078 14.272598 17.127117) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418078 14.272598 17.127117) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418078 14.272598 17.127117) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418078 14.272598 17.127117) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.2074      -108.39319      0              0            
Loop time of 1.334e-06 on 1 procs for 0 steps with 238 atoms

75.0% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.418078210581552 y final 0 14.272597763227115 z final 0 17.12711731587254      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.418078 14.272598 17.127117) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418078 14.272598 17.127117) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418078 14.272598 17.127117) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418078 14.272598 17.127117) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418078 14.272598 17.127117) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418078 14.272598 17.127117) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.2074      -108.39319      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417823649450456 y final 0 14.272279561813242 z final 0 17.12673547417589      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417824 14.272598 17.127117) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417824 14.27228 17.127117) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417824 14.27228 17.126735) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417824 14.27228 17.126735) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417824 14.27228 17.126735) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417824 14.27228 17.126735) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.21106     -23.705141      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417823649450456 y final 0 14.272279561813242 z final 0 17.12673547417589      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417824 14.27228 17.126735) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417824 14.27228 17.126735) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417824 14.27228 17.126735) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417824 14.27228 17.126735) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417824 14.27228 17.126735) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417824 14.27228 17.126735) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.21106     -23.705141      0              0            
Loop time of 1.334e-06 on 1 procs for 0 steps with 238 atoms

299.9% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417709615756635 y final 0 14.272137019695968 z final 0 17.126564423635163      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41771 14.27228 17.126735) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41771 14.272137 17.126735) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41771 14.272137 17.126564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41771 14.272137 17.126564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41771 14.272137 17.126564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41771 14.272137 17.126564) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.2117       3.7675936      0              0            
Loop time of 1.583e-06 on 1 procs for 0 steps with 238 atoms

189.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.583e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417709615756635 y final 0 14.272137019695968 z final 0 17.126564423635163      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41771 14.272137 17.126564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41771 14.272137 17.126564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41771 14.272137 17.126564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41771 14.272137 17.126564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41771 14.272137 17.126564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41771 14.272137 17.126564) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.2117       3.7675936      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417669284086827 y final 0 14.272086605108711 z final 0 17.126503926130454      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417669 14.272137 17.126564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417669 14.272087 17.126564) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417669 14.272087 17.126504) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417669 14.272087 17.126504) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417669 14.272087 17.126504) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417669 14.272087 17.126504) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.21198      0.11519747     0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417669284086827 y final 0 14.272086605108711 z final 0 17.126503926130454      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417669 14.272087 17.126504) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417669 14.272087 17.126504) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417669 14.272087 17.126504) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417669 14.272087 17.126504) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417669 14.272087 17.126504) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417669 14.272087 17.126504) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.21198      0.11519747     0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417657673044731 y final 0 14.272072091306095 z final 0 17.126486509567314      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417658 14.272087 17.126504) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417658 14.272072 17.126504) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417658 14.272072 17.126487) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417658 14.272072 17.126487) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417658 14.272072 17.126487) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417658 14.272072 17.126487) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.21209     -14.563144      0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 238 atoms

330.9% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.419033902642813 y final 0 14.273792378303515 z final 0 17.12855085396422      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.419034 14.272072 17.126487) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.126487) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.04006     -787.67227      0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 238 atoms

330.9% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 10.61052888630125 y final 0 13.263161107876561 z final 0 15.915793329451876      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.610529 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.610529 13.263161 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.610529 13.263161 15.915793) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.610529 13.263161 15.915793) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.610529 13.263161 15.915793) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.610529 13.263161 15.915793) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-905.67079      595333.73      0              0            
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
Neighs:          21241 ave       21241 max       21241 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 21241
Ave neighs/atom = 89.247899
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416820769912036 y final 0 14.271025962390043 z final 0 17.125231154868054      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416821 13.263161 15.915793) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416821 14.271026 15.915793) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416821 14.271026 17.125231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416821 14.271026 17.125231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416821 14.271026 17.125231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416821 14.271026 17.125231) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.04065      305.84438      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416820769912036 y final 0 14.271025962390043 z final 0 17.125231154868054      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416821 14.271026 17.125231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416821 14.271026 17.125231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416821 14.271026 17.125231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416821 14.271026 17.125231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416821 14.271026 17.125231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416821 14.271026 17.125231) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.04065      305.84438      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417315052954851 y final 0 14.271643816193563 z final 0 17.12597257943228      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417315 14.271026 17.125231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417315 14.271644 17.125231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417315 14.271644 17.125973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417315 14.271644 17.125973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417315 14.271644 17.125973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417315 14.271644 17.125973) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.01636     -1148.4103      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.41705830102283 y final 0 14.271322876278537 z final 0 17.125587451534248      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417058 14.271644 17.125973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417058 14.271323 17.125973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417058 14.271323 17.125587) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417058 14.271323 17.125587) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417058 14.271323 17.125587) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417058 14.271323 17.125587) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.21085     -515.32957      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.41705830102283 y final 0 14.271322876278537 z final 0 17.125587451534248      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417058 14.271323 17.125587) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417058 14.271323 17.125587) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417058 14.271323 17.125587) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417058 14.271323 17.125587) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417058 14.271323 17.125587) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417058 14.271323 17.125587) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.21085     -515.32957      0              0            
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
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416410291531955 y final 0 14.270512864414945 z final 0 17.124615437297937      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41641 14.271323 17.125587) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41641 14.270513 17.125587) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41641 14.270513 17.124615) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41641 14.270513 17.124615) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41641 14.270513 17.124615) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41641 14.270513 17.124615) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.23621     -582.2077       0              0            
Loop time of 1.458e-06 on 1 procs for 0 steps with 238 atoms

68.6% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416410291531955 y final 0 14.270512864414945 z final 0 17.124615437297937      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41641 14.270513 17.124615) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41641 14.270513 17.124615) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41641 14.270513 17.124615) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41641 14.270513 17.124615) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41641 14.270513 17.124615) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41641 14.270513 17.124615) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.23621     -582.2077       0              0            
Loop time of 7.09e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416093041141936 y final 0 14.270116301427421 z final 0 17.12413956171291      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416093 14.270513 17.124615) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416093 14.270116 17.124615) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416093 14.270116 17.12414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416093 14.270116 17.12414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416093 14.270116 17.12414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416093 14.270116 17.12414) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.24122     -464.13973      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416093041141934 y final 0 14.270116301427421 z final 0 17.12413956171291      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416093 14.270116 17.12414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416093 14.270116 17.12414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416093 14.270116 17.12414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416093 14.270116 17.12414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416093 14.270116 17.12414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416093 14.270116 17.12414) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.24122     -464.13973      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 238 atoms

494.4% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.415894084513349 y final 0 14.269867605641688 z final 0 17.123841126770028      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.415894 14.270116 17.12414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415894 14.269868 17.12414) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415894 14.269868 17.123841) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415894 14.269868 17.123841) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415894 14.269868 17.123841) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415894 14.269868 17.123841) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.24246     -396.49384      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.415894084513349 y final 0 14.269867605641688 z final 0 17.123841126770028      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.415894 14.269868 17.123841) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415894 14.269868 17.123841) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415894 14.269868 17.123841) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415894 14.269868 17.123841) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415894 14.269868 17.123841) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415894 14.269868 17.123841) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.24246     -396.49384      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.415836951931173 y final 0 14.269796189913968 z final 0 17.123755427896764      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.415837 14.269868 17.123841) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415837 14.269796 17.123841) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415837 14.269796 17.123755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415837 14.269796 17.123755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415837 14.269796 17.123755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415837 14.269796 17.123755) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.24294     -401.71382      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.415836951931173 y final 0 14.269796189913968 z final 0 17.123755427896764      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.415837 14.269796 17.123755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415837 14.269796 17.123755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415837 14.269796 17.123755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415837 14.269796 17.123755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415837 14.269796 17.123755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415837 14.269796 17.123755) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.24294     -401.71382      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.415851371900477 y final 0 14.2698142148756 z final 0 17.12377705785072      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.415851 14.269796 17.123755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415851 14.269814 17.123755) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415851 14.269814 17.123777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415851 14.269814 17.123777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415851 14.269814 17.123777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.415851 14.269814 17.123777) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-992.24307     -430.17808      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.419033902642813 y final 0 14.273792378303515 z final 0 17.12855085396422      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.419034 14.269814 17.123777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.123777) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.74645     -305.68288      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 10.82664494747497 y final 0 13.533306184343711 z final 0 16.23996742121246      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.826645 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.826645 13.533306 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.826645 13.533306 16.239967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.826645 13.533306 16.239967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.826645 13.533306 16.239967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.826645 13.533306 16.239967) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-947.64251      386394.8       0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416609827500121 y final 0 14.27076228437515 z final 0 17.124914741250183      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41661 13.533306 16.239967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41661 14.270762 16.239967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41661 14.270762 17.124915) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41661 14.270762 17.124915) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41661 14.270762 17.124915) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41661 14.270762 17.124915) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.74715      891.08381      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.41660982750012 y final 0 14.27076228437515 z final 0 17.124914741250183      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41661 14.270762 17.124915) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41661 14.270762 17.124915) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41661 14.270762 17.124915) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41661 14.270762 17.124915) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41661 14.270762 17.124915) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41661 14.270762 17.124915) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.74715      891.08381      0              0            
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
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.42042962991416 y final 0 14.275537037393894 z final 0 17.130644444872676      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.42043 14.270762 17.124915) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.42043 14.275537 17.124915) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.42043 14.275537 17.130644) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.42043 14.275537 17.130644) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.42043 14.275537 17.130644) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.42043 14.275537 17.130644) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.21096     -1750.6096      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417861587232002 y final 0 14.272326984040394 z final 0 17.126792380848475      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417862 14.275537 17.130644) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417862 14.272327 17.130644) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417862 14.272327 17.126792) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417862 14.272327 17.126792) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417862 14.272327 17.126792) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417862 14.272327 17.126792) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.91383     -40.885606      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417861587232002 y final 0 14.272326984040394 z final 0 17.126792380848475      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417862 14.272327 17.126792) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417862 14.272327 17.126792) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417862 14.272327 17.126792) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417862 14.272327 17.126792) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417862 14.272327 17.126792) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417862 14.272327 17.126792) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.91383     -40.885606      0              0            
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
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.41744355244223 y final 0 14.271804440553238 z final 0 17.126165328663888      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417444 14.272327 17.126792) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417444 14.271804 17.126792) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417444 14.271804 17.126165) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417444 14.271804 17.126165) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417444 14.271804 17.126165) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417444 14.271804 17.126165) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.93495     -124.74794      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.41744355244223 y final 0 14.271804440553238 z final 0 17.126165328663888      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417444 14.271804 17.126165) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417444 14.271804 17.126165) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417444 14.271804 17.126165) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417444 14.271804 17.126165) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417444 14.271804 17.126165) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417444 14.271804 17.126165) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.93495     -124.74794      0              0            
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
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417189125301826 y final 0 14.271486406627732 z final 0 17.12578368795328      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417189 14.271804 17.126165) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417189 14.271486 17.126165) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417189 14.271486 17.125784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417189 14.271486 17.125784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417189 14.271486 17.125784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417189 14.271486 17.125784) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.93877      5.9336161      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417189125301826 y final 0 14.271486406627732 z final 0 17.12578368795328      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417189 14.271486 17.125784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417189 14.271486 17.125784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417189 14.271486 17.125784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417189 14.271486 17.125784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417189 14.271486 17.125784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417189 14.271486 17.125784) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.93877      5.9336161      0              0            
Loop time of 1.333e-06 on 1 procs for 0 steps with 238 atoms

375.1% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417059597192575 y final 0 14.271324496491165 z final 0 17.1255893957894      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41706 14.271486 17.125784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41706 14.271324 17.125784) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41706 14.271324 17.125589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41706 14.271324 17.125589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41706 14.271324 17.125589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41706 14.271324 17.125589) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.93935      49.231615      0              0            
Loop time of 1.583e-06 on 1 procs for 0 steps with 238 atoms

63.2% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.583e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417059597192575 y final 0 14.271324496491165 z final 0 17.1255893957894      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41706 14.271324 17.125589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41706 14.271324 17.125589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41706 14.271324 17.125589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41706 14.271324 17.125589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41706 14.271324 17.125589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41706 14.271324 17.125589) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.93935      49.231615      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 238 atoms

428.4% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.41702566567481 y final 0 14.271282082093961 z final 0 17.125538498512757      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417026 14.271324 17.125589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417026 14.271282 17.125589) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417026 14.271282 17.125538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417026 14.271282 17.125538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417026 14.271282 17.125538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417026 14.271282 17.125538) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.93958      49.592644      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.41702566567481 y final 0 14.271282082093961 z final 0 17.125538498512757      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417026 14.271282 17.125538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417026 14.271282 17.125538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417026 14.271282 17.125538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417026 14.271282 17.125538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417026 14.271282 17.125538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417026 14.271282 17.125538) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.93958      49.592644      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417020075494472 y final 0 14.27127509436854 z final 0 17.125530113242256      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41702 14.271282 17.125538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41702 14.271275 17.125538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41702 14.271275 17.12553) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41702 14.271275 17.12553) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41702 14.271275 17.12553) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41702 14.271275 17.12553) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.93964      36.698795      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.419033902642813 y final 0 14.273792378303515 z final 0 17.12855085396422      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.419034 14.271275 17.12553) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.12553) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
WARNING: Too many warnings: 101 vs 100. All future warnings will be suppressed (src/src/thermo.cpp:465)
-991.74601     -350.03779      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 10.743173660394051 y final 0 13.428967075492563 z final 0 16.114760490591078      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.743174 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.743174 13.428967 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.743174 13.428967 16.11476) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.743174 13.428967 16.11476) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.743174 13.428967 16.11476) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.743174 13.428967 16.11476) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-933.30754      461264.25      0              0            
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
Neighs:          19913 ave       19913 max       19913 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19913
Ave neighs/atom = 83.668067
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416556267202113 y final 0 14.270695334002639 z final 0 17.12483440080317      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416556 13.428967 16.11476) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416556 14.270695 16.11476) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416556 14.270695 17.124834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416556 14.270695 17.124834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416556 14.270695 17.124834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416556 14.270695 17.124834) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.74674      872.6694       0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416556267202113 y final 0 14.270695334002639 z final 0 17.12483440080317      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416556 14.270695 17.124834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416556 14.270695 17.124834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416556 14.270695 17.124834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416556 14.270695 17.124834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416556 14.270695 17.124834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416556 14.270695 17.124834) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.74674      872.6694       0              0            
Loop time of 1.584e-06 on 1 procs for 0 steps with 238 atoms

441.9% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.418662077711568 y final 0 14.273327597139458 z final 0 17.127993116567353      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.418662 14.270695 17.124834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418662 14.273328 17.124834) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418662 14.273328 17.127993) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418662 14.273328 17.127993) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418662 14.273328 17.127993) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.418662 14.273328 17.127993) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.54598     -1898.7831      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19817 ave       19817 max       19817 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19817
Ave neighs/atom = 83.264706
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417431988826442 y final 0 14.27178998603305 z final 0 17.126147983239665      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417432 14.273328 17.127993) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417432 14.27179 17.127993) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417432 14.27179 17.126148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417432 14.27179 17.126148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417432 14.27179 17.126148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417432 14.27179 17.126148) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.95915     -194.41548      0              0            
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
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417431988826442 y final 0 14.27178998603305 z final 0 17.126147983239665      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417432 14.27179 17.126148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417432 14.27179 17.126148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417432 14.27179 17.126148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417432 14.27179 17.126148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417432 14.27179 17.126148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417432 14.27179 17.126148) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.95915     -194.41548      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416814503211535 y final 0 14.271018129014418 z final 0 17.125221754817307      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416815 14.27179 17.126148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416815 14.271018 17.126148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416815 14.271018 17.125222) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416815 14.271018 17.125222) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416815 14.271018 17.125222) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416815 14.271018 17.125222) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.98554     -244.58408      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416814503211535 y final 0 14.271018129014418 z final 0 17.125221754817307      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416815 14.271018 17.125222) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416815 14.271018 17.125222) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416815 14.271018 17.125222) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416815 14.271018 17.125222) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416815 14.271018 17.125222) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416815 14.271018 17.125222) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.98554     -244.58408      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416450871524525 y final 0 14.270563589405656 z final 0 17.12467630728679      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416451 14.271018 17.125222) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416451 14.270564 17.125222) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416451 14.270564 17.124676) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416451 14.270564 17.124676) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416451 14.270564 17.124676) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416451 14.270564 17.124676) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.99127     -96.743503      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416450871524525 y final 0 14.270563589405656 z final 0 17.12467630728679      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416451 14.270564 17.124676) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416451 14.270564 17.124676) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416451 14.270564 17.124676) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416451 14.270564 17.124676) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416451 14.270564 17.124676) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416451 14.270564 17.124676) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.99127     -96.743503      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.41627283273964 y final 0 14.270341040924551 z final 0 17.124409249109465      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416273 14.270564 17.124676) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416273 14.270341 17.124676) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416273 14.270341 17.124409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416273 14.270341 17.124409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416273 14.270341 17.124409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416273 14.270341 17.124409) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.99237     -58.715657      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.41627283273964 y final 0 14.270341040924551 z final 0 17.124409249109465      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416273 14.270341 17.124409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416273 14.270341 17.124409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416273 14.270341 17.124409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416273 14.270341 17.124409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416273 14.270341 17.124409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416273 14.270341 17.124409) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.99237     -58.715657      0              0            
Loop time of 1.667e-06 on 1 procs for 0 steps with 238 atoms

359.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.667e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416227299945737 y final 0 14.270284124932175 z final 0 17.12434094991861      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416227 14.270341 17.124409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416227 14.270284 17.124409) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416227 14.270284 17.124341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416227 14.270284 17.124341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416227 14.270284 17.124341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416227 14.270284 17.124341) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.99283     -66.961964      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416227299945737 y final 0 14.270284124932175 z final 0 17.12434094991861      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416227 14.270284 17.124341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416227 14.270284 17.124341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416227 14.270284 17.124341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416227 14.270284 17.124341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416227 14.270284 17.124341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416227 14.270284 17.124341) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.99283     -66.961964      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416240164582641 y final 0 14.270300205728308 z final 0 17.12436024687397      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41624 14.270284 17.124341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41624 14.2703 17.124341) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41624 14.2703 17.12436) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41624 14.2703 17.12436) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41624 14.2703 17.12436) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41624 14.2703 17.12436) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.99298     -94.522133      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.419033902642813 y final 0 14.273792378303515 z final 0 17.12855085396422      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.419034 14.2703 17.12436) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.12436) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419034 14.273792 17.128551) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.74679     -350.24624      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 10.74295647621836 y final 0 13.428695595272949 z final 0 16.11443471432754      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.742956 14.273792 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.742956 13.428696 17.128551) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.742956 13.428696 16.114435) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.742956 13.428696 16.114435) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.742956 13.428696 16.114435) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.742956 13.428696 16.114435) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-933.17781      461789.01      0              0            
Loop time of 1.791e-06 on 1 procs for 0 steps with 238 atoms

223.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.791e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19912 ave       19912 max       19912 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19912
Ave neighs/atom = 83.663866
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416796620908295 y final 0 14.270995776135369 z final 0 17.125194931362444      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416797 13.428696 16.114435) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416797 14.270996 16.114435) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416797 14.270996 17.125195) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416797 14.270996 17.125195) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416797 14.270996 17.125195) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416797 14.270996 17.125195) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.74739      754.10995      0              0            
Loop time of 6.25e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416796620908293 y final 0 14.270995776135369 z final 0 17.125194931362444      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416797 14.270996 17.125195) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416797 14.270996 17.125195) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416797 14.270996 17.125195) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416797 14.270996 17.125195) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416797 14.270996 17.125195) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416797 14.270996 17.125195) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.74739      754.10995      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.419017301107 y final 0 14.273771626384855 z final 0 17.128525951660503      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.419017 14.270996 17.125195) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419017 14.273772 17.125195) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419017 14.273772 17.128526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419017 14.273772 17.128526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419017 14.273772 17.128526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.419017 14.273772 17.128526) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.64513     -1317.5948      0              0            
Loop time of 1.708e-06 on 1 procs for 0 steps with 238 atoms

58.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.708e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417776662720756 y final 0 14.272220828401432 z final 0 17.126664994081136      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417777 14.273772 17.128526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417777 14.272221 17.128526) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417777 14.272221 17.126665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417777 14.272221 17.126665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417777 14.272221 17.126665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417777 14.272221 17.126665) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.92464     -103.17273      0              0            
Loop time of 1.291e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417776662720756 y final 0 14.272220828401432 z final 0 17.126664994081136      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.417777 14.272221 17.126665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417777 14.272221 17.126665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417777 14.272221 17.126665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417777 14.272221 17.126665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417777 14.272221 17.126665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.417777 14.272221 17.126665) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.92464     -103.17273      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417239840363601 y final 0 14.27154980045506 z final 0 17.125859760545403      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41724 14.272221 17.126665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41724 14.27155 17.126665) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41724 14.27155 17.12586) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41724 14.27155 17.12586) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41724 14.27155 17.12586) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41724 14.27155 17.12586) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.9448      -182.81189      0              0            
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
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.417239840363601 y final 0 14.27154980045506 z final 0 17.125859760545403      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41724 14.27155 17.12586) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41724 14.27155 17.12586) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41724 14.27155 17.12586) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41724 14.27155 17.12586) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41724 14.27155 17.12586) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41724 14.27155 17.12586) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.9448      -182.81189      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416930450103271 y final 0 14.271163062629643 z final 0 17.12539567515491      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41693 14.27155 17.12586) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41693 14.271163 17.12586) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41693 14.271163 17.125396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41693 14.271163 17.125396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41693 14.271163 17.125396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41693 14.271163 17.125396) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.94896     -33.230803      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416930450103271 y final 0 14.271163062629643 z final 0 17.12539567515491      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.41693 14.271163 17.125396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41693 14.271163 17.125396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41693 14.271163 17.125396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41693 14.271163 17.125396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41693 14.271163 17.125396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.41693 14.271163 17.125396) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.94896     -33.230803      0              0            
Loop time of 1.458e-06 on 1 procs for 0 steps with 238 atoms

137.2% CPU use with 1 MPI tasks x no OpenMP threads

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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416797829273458 y final 0 14.270997286592376 z final 0 17.12519674391019      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416798 14.271163 17.125396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416798 14.270997 17.125396) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416798 14.270997 17.125197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416798 14.270997 17.125197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416798 14.270997 17.125197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416798 14.270997 17.125197) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.94953      12.134539      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416797829273458 y final 0 14.270997286592376 z final 0 17.12519674391019      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416798 14.270997 17.125197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416798 14.270997 17.125197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416798 14.270997 17.125197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416798 14.270997 17.125197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416798 14.270997 17.125197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416798 14.270997 17.125197) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.94953      12.134539      0              0            
Loop time of 1.583e-06 on 1 procs for 0 steps with 238 atoms

189.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.583e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416767458396777 y final 0 14.270959322996525 z final 0 17.12515118759517      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416767 14.270997 17.125197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416767 14.270959 17.125197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416767 14.270959 17.125151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416767 14.270959 17.125151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416767 14.270959 17.125151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416767 14.270959 17.125151) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.94978      11.687659      0              0            
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
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416767458396777 y final 0 14.270959322996525 z final 0 17.12515118759517      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416767 14.270959 17.125151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416767 14.270959 17.125151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416767 14.270959 17.125151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416767 14.270959 17.125151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416767 14.270959 17.125151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416767 14.270959 17.125151) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.94978      11.687659      0              0            
Loop time of 1.708e-06 on 1 procs for 0 steps with 238 atoms

117.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.708e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.416765270649373 y final 0 14.270956588312274 z final 0 17.125147905974064      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.416765 14.270959 17.125151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416765 14.270957 17.125151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416765 14.270957 17.125148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416765 14.270957 17.125148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416765 14.270957 17.125148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.416765 14.270957 17.125148) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.264 | 3.264 | 3.264 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-991.94986     -5.0694366      0              0            
Loop time of 1.791e-06 on 1 procs for 0 steps with 238 atoms

55.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.791e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2056 ave        2056 max        2056 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          19825 ave       19825 max       19825 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 19825
Ave neighs/atom = 83.298319
Neighbor list builds = 0
Dangerous builds = 0
change_box all boundary p p f
Changing box ...
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 8.07447630715782 y final 0 5.709516951321406 z final 0 38.167571691105096      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0744763 14.270957 17.125148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 5.709517 17.125148) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 5.709517 38.167572) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 5.709517 38.167572) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 5.709517 38.167572) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 5.709517 38.167572) with tilt (0 0 0)
echo none
158 atoms in group delatoms
Deleted 158 atoms, new total = 80
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.03665      7915.8678      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.07447630715782 y final 0 5.709516951321406 z final 0 38.16757169110509      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0744763 5.709517 38.167572) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 5.709517 38.167572) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 5.709517 38.167572) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 5.709517 38.167572) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 5.709517 38.167572) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 5.709517 38.167572) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.03665      7915.8678      0              0            
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
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 9.862187632261504 y final 0 6.9736197521062095 z final 0 46.61797733579177      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.8621876 5.709517 38.167572) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.8621876 6.9736198 38.167572) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.8621876 6.9736198 46.617977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.8621876 6.9736198 46.617977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.8621876 6.9736198 46.617977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.8621876 6.9736198 46.617977) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-173.54775     -253971.94      0              0            
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
Nghost:            635 ave         635 max         635 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           2304 ave        2304 max        2304 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 2304
Ave neighs/atom = 28.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.080945412185773 y final 0 5.714091299354881 z final 0 38.198150767758385      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0809454 6.9736198 46.617977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0809454 5.7140913 46.617977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0809454 5.7140913 38.198151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0809454 5.7140913 38.198151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0809454 5.7140913 38.198151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0809454 5.7140913 38.198151) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.03898      5584.9328      0              0            
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
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.080945412185772 y final 0 5.714091299354881 z final 0 38.198150767758385      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0809454 5.7140913 38.198151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0809454 5.7140913 38.198151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0809454 5.7140913 38.198151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0809454 5.7140913 38.198151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0809454 5.7140913 38.198151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0809454 5.7140913 38.198151) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.03898      5584.9328      0              0            
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
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.080048432398383 y final 0 5.713457038864631 z final 0 38.193910797382806      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0800484 5.7140913 38.198151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0800484 5.713457 38.198151) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0800484 5.713457 38.193911) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0800484 5.713457 38.193911) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0800484 5.713457 38.193911) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0800484 5.713457 38.193911) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.03924      5882.6298      0              0            
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
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.080048432398383 y final 0 5.713457038864631 z final 0 38.193910797382806      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0800484 5.713457 38.193911) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0800484 5.713457 38.193911) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0800484 5.713457 38.193911) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0800484 5.713457 38.193911) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0800484 5.713457 38.193911) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0800484 5.713457 38.193911) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.03924      5882.6298      0              0            
Loop time of 6.67e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.08005598433814 y final 0 5.7134623788924435 z final 0 38.19394649495514      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.080056 5.713457 38.193911) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.080056 5.7134624 38.193911) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.080056 5.7134624 38.193946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.080056 5.7134624 38.193946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.080056 5.7134624 38.193946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.080056 5.7134624 38.193946) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.03941      5905.3336      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.08005598433814 y final 0 5.7134623788924435 z final 0 38.19394649495514      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.080056 5.7134624 38.193946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.080056 5.7134624 38.193946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.080056 5.7134624 38.193946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.080056 5.7134624 38.193946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.080056 5.7134624 38.193946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.080056 5.7134624 38.193946) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.03941      5905.3336      0              0            
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
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.080370028771691 y final 0 5.713684441841002 z final 0 38.195430964408736      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.08037 5.7134624 38.193946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.08037 5.7136844 38.193946) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.08037 5.7136844 38.195431) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.08037 5.7136844 38.195431) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.08037 5.7136844 38.195431) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.08037 5.7136844 38.195431) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.03955      5841.8178      0              0            
Loop time of 6.25e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.25e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.080820883640504 y final 0 5.71400324437607 z final 0 38.197562128693974      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0808209 5.7136844 38.195431) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0808209 5.7140032 38.195431) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0808209 5.7140032 38.197562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0808209 5.7140032 38.197562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0808209 5.7140032 38.197562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0808209 5.7140032 38.197562) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.03963      5750.6005      0              0            
Loop time of 5.41e-07 on 1 procs for 0 steps with 80 atoms

184.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.41e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.080820883640504 y final 0 5.71400324437607 z final 0 38.197562128693974      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0808209 5.7140032 38.197562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0808209 5.7140032 38.197562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0808209 5.7140032 38.197562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0808209 5.7140032 38.197562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0808209 5.7140032 38.197562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0808209 5.7140032 38.197562) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.03963      5750.6005      0              0            
Loop time of 1.291e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.081600008271627 y final 0 5.714554168686127 z final 0 38.20124500472005      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0816 5.7140032 38.197562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0816 5.7145542 38.197562) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0816 5.7145542 38.201245) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0816 5.7145542 38.201245) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0816 5.7145542 38.201245) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0816 5.7145542 38.201245) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.03999      5602.3283      0              0            
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
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.082592931339498 y final 0 5.715256271320616 z final 0 38.20593849330496      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0825929 5.7145542 38.201245) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0825929 5.7152563 38.201245) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0825929 5.7152563 38.205938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0825929 5.7152563 38.205938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0825929 5.7152563 38.205938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0825929 5.7152563 38.205938) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.04015      5413.5514      0              0            
Loop time of 5.83e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.082592931339496 y final 0 5.715256271320616 z final 0 38.205938493304956      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0825929 5.7152563 38.205938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0825929 5.7152563 38.205938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0825929 5.7152563 38.205938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0825929 5.7152563 38.205938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0825929 5.7152563 38.205938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0825929 5.7152563 38.205938) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.04015      5413.5514      0              0            
Loop time of 1.458e-06 on 1 procs for 0 steps with 80 atoms

68.6% CPU use with 1 MPI tasks x no OpenMP threads

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
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.083658794679563 y final 0 5.716009950516195 z final 0 38.21097676624024      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0836588 5.7152563 38.205938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0836588 5.71601 38.205938) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0836588 5.71601 38.210977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0836588 5.71601 38.210977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0836588 5.71601 38.210977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0836588 5.71601 38.210977) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.04064      5231.8043      0              0            
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
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.083658794679563 y final 0 5.716009950516194 z final 0 38.21097676624024      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0836588 5.71601 38.210977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0836588 5.71601 38.210977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0836588 5.71601 38.210977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0836588 5.71601 38.210977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0836588 5.71601 38.210977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0836588 5.71601 38.210977) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.04064      5231.8043      0              0            
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
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.084577671024581 y final 0 5.716659694210829 z final 0 38.215320240348      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0845777 5.71601 38.210977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0845777 5.7166597 38.210977) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0845777 5.7166597 38.21532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0845777 5.7166597 38.21532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0845777 5.7166597 38.21532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0845777 5.7166597 38.21532) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.265 | 3.265 | 3.265 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-325.04077      5056.4726      0              0            
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
Nghost:            745 ave         745 max         745 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5776 ave        5776 max        5776 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5776
Ave neighs/atom = 72.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.709516951321406 y final 0 5.709516951321406 z final 0 47.12020551877668      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.709517 5.7166597 38.21532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.709517 5.709517 38.21532) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.709517 5.709517 47.120206) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.709517 5.709517 47.120206) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.709517 5.709517 47.120206) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.709517 5.709517 47.120206) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.31281     -6627.0352      0              0            
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
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 4.674557113266878 y final 0 4.674557113266878 z final 0 38.578761349577306      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (4.6745571 5.709517 47.120206) with tilt (0 0 0)
  triclinic box = (0 0 0) to (4.6745571 4.6745571 47.120206) with tilt (0 0 0)
  triclinic box = (0 0 0) to (4.6745571 4.6745571 38.578761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (4.6745571 4.6745571 38.578761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (4.6745571 4.6745571 38.578761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (4.6745571 4.6745571 38.578761) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-63.663235      1996208.1      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           9936 ave        9936 max        9936 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 9936
Ave neighs/atom = 124.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.70342200114331 y final 0 5.70342200114331 z final 0 47.069904362397395      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.703422 4.6745571 38.578761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.703422 5.703422 38.578761) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.703422 5.703422 47.069904) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.703422 5.703422 47.069904) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.703422 5.703422 47.069904) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.703422 5.703422 47.069904) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.31894     -2924.4702      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.70342200114331 y final 0 5.70342200114331 z final 0 47.069904362397395      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.703422 5.703422 47.069904) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.703422 5.703422 47.069904) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.703422 5.703422 47.069904) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.703422 5.703422 47.069904) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.703422 5.703422 47.069904) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.703422 5.703422 47.069904) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.31894     -2924.4702      0              0            
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
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.700117478616846 y final 0 5.700117478616846 z final 0 47.04263239142056      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.7001175 5.703422 47.069904) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001175 5.7001175 47.069904) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001175 5.7001175 47.042632) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001175 5.7001175 47.042632) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001175 5.7001175 47.042632) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001175 5.7001175 47.042632) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.34917     -2731.0714      0              0            
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
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.700117478616846 y final 0 5.700117478616846 z final 0 47.04263239142056      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.7001175 5.7001175 47.042632) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001175 5.7001175 47.042632) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001175 5.7001175 47.042632) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001175 5.7001175 47.042632) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001175 5.7001175 47.042632) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001175 5.7001175 47.042632) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.34917     -2731.0714      0              0            
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
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.7001846731405745 y final 0 5.7001846731405745 z final 0 47.043186942671525      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.7001847 5.7001175 47.042632) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001847 5.7001847 47.042632) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001847 5.7001847 47.043187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001847 5.7001847 47.043187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001847 5.7001847 47.043187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001847 5.7001847 47.043187) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.35553     -2770.9156      0              0            
Loop time of 6.67e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.7001846731405745 y final 0 5.7001846731405745 z final 0 47.043186942671525      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.7001847 5.7001847 47.043187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001847 5.7001847 47.043187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001847 5.7001847 47.043187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001847 5.7001847 47.043187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001847 5.7001847 47.043187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.7001847 5.7001847 47.043187) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.35553     -2770.9156      0              0            
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
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.699803829754832 y final 0 5.699803829754832 z final 0 47.04004387141705      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6998038 5.7001847 47.043187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6998038 5.6998038 47.043187) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6998038 5.6998038 47.040044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6998038 5.6998038 47.040044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6998038 5.6998038 47.040044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6998038 5.6998038 47.040044) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.35678     -2692.1586      0              0            
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
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.6993683977748155 y final 0 5.6993683977748155 z final 0 47.03645028468059      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6993684 5.6998038 47.040044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6993684 5.6993684 47.040044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6993684 5.6993684 47.03645) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6993684 5.6993684 47.03645) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6993684 5.6993684 47.03645) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6993684 5.6993684 47.03645) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.35727     -2605.0181      0              0            
Loop time of 6.67e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.698582871280086 y final 0 5.698582871280086 z final 0 47.02996739476405      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6985829 5.6993684 47.03645) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6985829 5.6985829 47.03645) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6985829 5.6985829 47.029967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6985829 5.6985829 47.029967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6985829 5.6985829 47.029967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6985829 5.6985829 47.029967) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.35869     -2494.5019      0              0            
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
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.6974746452604705 y final 0 5.6974746452604705 z final 0 47.02082129041743      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6974746 5.6985829 47.029967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6974746 5.6974746 47.029967) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6974746 5.6974746 47.020821) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6974746 5.6974746 47.020821) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6974746 5.6974746 47.020821) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6974746 5.6974746 47.020821) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.35942     -2336.2088      0              0            
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
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.6974746452604705 y final 0 5.6974746452604705 z final 0 47.02082129041743      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6974746 5.6974746 47.020821) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6974746 5.6974746 47.020821) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6974746 5.6974746 47.020821) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6974746 5.6974746 47.020821) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6974746 5.6974746 47.020821) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6974746 5.6974746 47.020821) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.35942     -2336.2088      0              0            
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
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.696876167948934 y final 0 5.696876167948934 z final 0 47.01588210306439      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6968762 5.6974746 47.020821) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6968762 5.6968762 47.020821) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6968762 5.6968762 47.015882) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6968762 5.6968762 47.015882) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6968762 5.6968762 47.015882) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6968762 5.6968762 47.015882) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.36089     -2242.0451      0              0            
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
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.695859647939593 y final 0 5.695859647939593 z final 0 47.00749284138736      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6958596 5.6968762 47.015882) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6958596 5.6958596 47.015882) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6958596 5.6958596 47.007493) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6958596 5.6958596 47.007493) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6958596 5.6958596 47.007493) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6958596 5.6958596 47.007493) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.36186     -2077.3043      0              0            
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
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.695859647939593 y final 0 5.695859647939593 z final 0 47.00749284138736      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6958596 5.6958596 47.007493) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6958596 5.6958596 47.007493) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6958596 5.6958596 47.007493) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6958596 5.6958596 47.007493) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6958596 5.6958596 47.007493) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6958596 5.6958596 47.007493) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.36186     -2077.3043      0              0            
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
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.695127729490904 y final 0 5.695127729490904 z final 0 47.00145237456341      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6951277 5.6958596 47.007493) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6951277 5.6951277 47.007493) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6951277 5.6951277 47.001452) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6951277 5.6951277 47.001452) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6951277 5.6951277 47.001452) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6951277 5.6951277 47.001452) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.36386     -1938.8143      0              0            
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
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.693775289036553 y final 0 5.693775289036553 z final 0 46.99029078721649      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6937753 5.6951277 47.001452) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6937753 5.6937753 47.001452) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6937753 5.6937753 46.990291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6937753 5.6937753 46.990291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6937753 5.6937753 46.990291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6937753 5.6937753 46.990291) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.36531     -1675.4228      0              0            
Loop time of 5.84e-07 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 5.84e-07   |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.693775289036553 y final 0 5.693775289036553 z final 0 46.99029078721649      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6937753 5.6937753 46.990291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6937753 5.6937753 46.990291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6937753 5.6937753 46.990291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6937753 5.6937753 46.990291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6937753 5.6937753 46.990291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6937753 5.6937753 46.990291) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.36531     -1675.4228      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.692412580159533 y final 0 5.692412580159533 z final 0 46.979044455364786      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6924126 5.6937753 46.990291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6924126 5.6924126 46.990291) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6924126 5.6924126 46.979044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6924126 5.6924126 46.979044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6924126 5.6924126 46.979044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6924126 5.6924126 46.979044) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.36855     -1444.7971      0              0            
Loop time of 6.25e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.688454683101228 y final 0 5.688454683101228 z final 0 46.94638023448593      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6884547 5.6924126 46.979044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6884547 5.6884547 46.979044) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6884547 5.6884547 46.94638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6884547 5.6884547 46.94638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6884547 5.6884547 46.94638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6884547 5.6884547 46.94638) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.37255     -753.70088      0              0            
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
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.688454683101228 y final 0 5.688454683101228 z final 0 46.94638023448593      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6884547 5.6884547 46.94638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6884547 5.6884547 46.94638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6884547 5.6884547 46.94638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6884547 5.6884547 46.94638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6884547 5.6884547 46.94638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6884547 5.6884547 46.94638) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.37255     -753.70088      0              0            
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
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.686023933695445 y final 0 5.686023933695445 z final 0 46.92631944606167      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6860239 5.6884547 46.94638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6860239 5.6860239 46.94638) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6860239 5.6860239 46.926319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6860239 5.6860239 46.926319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6860239 5.6860239 46.926319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6860239 5.6860239 46.926319) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.37743     -375.71535      0              0            
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
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.686023933695445 y final 0 5.686023933695445 z final 0 46.92631944606167      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6860239 5.6860239 46.926319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6860239 5.6860239 46.926319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6860239 5.6860239 46.926319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6860239 5.6860239 46.926319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6860239 5.6860239 46.926319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6860239 5.6860239 46.926319) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.37743     -375.71535      0              0            
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
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.683673555378457 y final 0 5.683673555378457 z final 0 46.906921954069325      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6836736 5.6860239 46.926319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6836736 5.6836736 46.926319) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6836736 5.6836736 46.906922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6836736 5.6836736 46.906922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6836736 5.6836736 46.906922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6836736 5.6836736 46.906922) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.37884      17.799893      0              0            
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
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.683673555378457 y final 0 5.683673555378457 z final 0 46.906921954069325      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6836736 5.6836736 46.906922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6836736 5.6836736 46.906922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6836736 5.6836736 46.906922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6836736 5.6836736 46.906922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6836736 5.6836736 46.906922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6836736 5.6836736 46.906922) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.37884      17.799893      0              0            
Loop time of 5.83e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.683734226899617 y final 0 5.683734226899617 z final 0 46.907422671480376      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.6837342 5.6836736 46.906922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6837342 5.6837342 46.906922) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6837342 5.6837342 46.907423) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6837342 5.6837342 46.907423) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6837342 5.6837342 46.907423) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.6837342 5.6837342 46.907423) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.273 | 3.273 | 3.273 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-327.37892      12.054984      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1050 ave        1050 max        1050 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6040 ave        6040 max        6040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6040
Ave neighs/atom = 75.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.07447630715782 y final 0 6.9927016042542345 z final 0 35.65785795674369      xy final 4.037238153578909 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0744763 5.6837342 46.907423) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 6.9927016 46.907423) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 6.9927016 35.657858) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.0744763 6.9927016 35.657858) with tilt (4.0372382 0 0)
  triclinic box = (0 0 0) to (8.0744763 6.9927016 35.657858) with tilt (4.0372382 0 0)
  triclinic box = (0 0 0) to (8.0744763 6.9927016 35.657858) with tilt (4.0372382 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.72574     -1163.3577      0              0            
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
change_box all     x final 0 6.610822067669644 y final 0 5.725139850500681 z final 0 29.194184898072468      xy final 3.305411033834821 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.6108221 6.9927016 35.657858) with tilt (4.0372382 0 0)
  triclinic box = (0 0 0) to (6.6108221 5.7251399 35.657858) with tilt (4.0372382 0 0)
  triclinic box = (0 0 0) to (6.6108221 5.7251399 29.194185) with tilt (4.0372382 0 0)
  triclinic box = (0 0 0) to (6.6108221 5.7251399 29.194185) with tilt (3.305411 0 0)
  triclinic box = (0 0 0) to (6.6108221 5.7251399 29.194185) with tilt (3.305411 0 0)
  triclinic box = (0 0 0) to (6.6108221 5.7251399 29.194185) with tilt (3.305411 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-62.298995      1506156.1      0              0            
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
Nghost:            900 ave         900 max         900 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           8904 ave        8904 max        8904 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 8904
Ave neighs/atom = 111.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 8.065696362314759 y final 0 6.985097948976317 z final 0 35.61908466493225      xy final 4.032848181157378 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0656964 5.7251399 29.194185) with tilt (3.305411 0 0)
  triclinic box = (0 0 0) to (8.0656964 6.9850979 29.194185) with tilt (3.305411 0 0)
  triclinic box = (0 0 0) to (8.0656964 6.9850979 35.619085) with tilt (3.305411 0 0)
  triclinic box = (0 0 0) to (8.0656964 6.9850979 35.619085) with tilt (4.0328482 0 0)
  triclinic box = (0 0 0) to (8.0656964 6.9850979 35.619085) with tilt (4.0328482 0 0)
  triclinic box = (0 0 0) to (8.0656964 6.9850979 35.619085) with tilt (4.0328482 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.73207      1583.9328      0              0            
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
change_box all     x final 0 8.065696362314759 y final 0 6.985097948976317 z final 0 35.61908466493225      xy final 4.032848181157378 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0656964 6.9850979 35.619085) with tilt (4.0328482 0 0)
  triclinic box = (0 0 0) to (8.0656964 6.9850979 35.619085) with tilt (4.0328482 0 0)
  triclinic box = (0 0 0) to (8.0656964 6.9850979 35.619085) with tilt (4.0328482 0 0)
  triclinic box = (0 0 0) to (8.0656964 6.9850979 35.619085) with tilt (4.0328482 0 0)
  triclinic box = (0 0 0) to (8.0656964 6.9850979 35.619085) with tilt (4.0328482 0 0)
  triclinic box = (0 0 0) to (8.0656964 6.9850979 35.619085) with tilt (4.0328482 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.73207      1583.9328      0              0            
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
change_box all     x final 0 8.074395019980384 y final 0 6.992631207493553 z final 0 35.65749898279699      xy final 4.037197509990191 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.074395 6.9850979 35.619085) with tilt (4.0328482 0 0)
  triclinic box = (0 0 0) to (8.074395 6.9926312 35.619085) with tilt (4.0328482 0 0)
  triclinic box = (0 0 0) to (8.074395 6.9926312 35.657499) with tilt (4.0328482 0 0)
  triclinic box = (0 0 0) to (8.074395 6.9926312 35.657499) with tilt (4.0371975 0 0)
  triclinic box = (0 0 0) to (8.074395 6.9926312 35.657499) with tilt (4.0371975 0 0)
  triclinic box = (0 0 0) to (8.074395 6.9926312 35.657499) with tilt (4.0371975 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.85851      1302.3552      0              0            
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
change_box all     x final 0 8.074395019980384 y final 0 6.992631207493553 z final 0 35.65749898279699      xy final 4.037197509990191 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.074395 6.9926312 35.657499) with tilt (4.0371975 0 0)
  triclinic box = (0 0 0) to (8.074395 6.9926312 35.657499) with tilt (4.0371975 0 0)
  triclinic box = (0 0 0) to (8.074395 6.9926312 35.657499) with tilt (4.0371975 0 0)
  triclinic box = (0 0 0) to (8.074395 6.9926312 35.657499) with tilt (4.0371975 0 0)
  triclinic box = (0 0 0) to (8.074395 6.9926312 35.657499) with tilt (4.0371975 0 0)
  triclinic box = (0 0 0) to (8.074395 6.9926312 35.657499) with tilt (4.0371975 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.85851      1302.3552      0              0            
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
change_box all     x final 0 8.074641542856826 y final 0 6.992844702567166 z final 0 35.658587657452394      xy final 4.037320771428412 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0746415 6.9926312 35.657499) with tilt (4.0371975 0 0)
  triclinic box = (0 0 0) to (8.0746415 6.9928447 35.657499) with tilt (4.0371975 0 0)
  triclinic box = (0 0 0) to (8.0746415 6.9928447 35.658588) with tilt (4.0371975 0 0)
  triclinic box = (0 0 0) to (8.0746415 6.9928447 35.658588) with tilt (4.0373208 0 0)
  triclinic box = (0 0 0) to (8.0746415 6.9928447 35.658588) with tilt (4.0373208 0 0)
  triclinic box = (0 0 0) to (8.0746415 6.9928447 35.658588) with tilt (4.0373208 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.87933      1308.9588      0              0            
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
change_box all     x final 0 8.074641542856824 y final 0 6.992844702567166 z final 0 35.658587657452394      xy final 4.037320771428411 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0746415 6.9928447 35.658588) with tilt (4.0373208 0 0)
  triclinic box = (0 0 0) to (8.0746415 6.9928447 35.658588) with tilt (4.0373208 0 0)
  triclinic box = (0 0 0) to (8.0746415 6.9928447 35.658588) with tilt (4.0373208 0 0)
  triclinic box = (0 0 0) to (8.0746415 6.9928447 35.658588) with tilt (4.0373208 0 0)
  triclinic box = (0 0 0) to (8.0746415 6.9928447 35.658588) with tilt (4.0373208 0 0)
  triclinic box = (0 0 0) to (8.0746415 6.9928447 35.658588) with tilt (4.0373208 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.87933      1308.9588      0              0            
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
change_box all     x final 0 8.075182653245646 y final 0 6.993313317910138 z final 0 35.66097726597306      xy final 4.037591326622822 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0751827 6.9928447 35.658588) with tilt (4.0373208 0 0)
  triclinic box = (0 0 0) to (8.0751827 6.9933133 35.658588) with tilt (4.0373208 0 0)
  triclinic box = (0 0 0) to (8.0751827 6.9933133 35.660977) with tilt (4.0373208 0 0)
  triclinic box = (0 0 0) to (8.0751827 6.9933133 35.660977) with tilt (4.0375913 0 0)
  triclinic box = (0 0 0) to (8.0751827 6.9933133 35.660977) with tilt (4.0375913 0 0)
  triclinic box = (0 0 0) to (8.0751827 6.9933133 35.660977) with tilt (4.0375913 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88174      1250.4125      0              0            
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
change_box all     x final 0 8.075182653245646 y final 0 6.993313317910138 z final 0 35.66097726597306      xy final 4.037591326622822 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0751827 6.9933133 35.660977) with tilt (4.0375913 0 0)
  triclinic box = (0 0 0) to (8.0751827 6.9933133 35.660977) with tilt (4.0375913 0 0)
  triclinic box = (0 0 0) to (8.0751827 6.9933133 35.660977) with tilt (4.0375913 0 0)
  triclinic box = (0 0 0) to (8.0751827 6.9933133 35.660977) with tilt (4.0375913 0 0)
  triclinic box = (0 0 0) to (8.0751827 6.9933133 35.660977) with tilt (4.0375913 0 0)
  triclinic box = (0 0 0) to (8.0751827 6.9933133 35.660977) with tilt (4.0375913 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88174      1250.4125      0              0            
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
change_box all     x final 0 8.075927918304936 y final 0 6.993958736384036 z final 0 35.66426844605873      xy final 4.037963959152467 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0759279 6.9933133 35.660977) with tilt (4.0375913 0 0)
  triclinic box = (0 0 0) to (8.0759279 6.9939587 35.660977) with tilt (4.0375913 0 0)
  triclinic box = (0 0 0) to (8.0759279 6.9939587 35.664268) with tilt (4.0375913 0 0)
  triclinic box = (0 0 0) to (8.0759279 6.9939587 35.664268) with tilt (4.037964 0 0)
  triclinic box = (0 0 0) to (8.0759279 6.9939587 35.664268) with tilt (4.037964 0 0)
  triclinic box = (0 0 0) to (8.0759279 6.9939587 35.664268) with tilt (4.037964 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88254      1160.4954      0              0            
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
change_box all     x final 0 8.075927918304936 y final 0 6.993958736384036 z final 0 35.66426844605873      xy final 4.037963959152467 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0759279 6.9939587 35.664268) with tilt (4.037964 0 0)
  triclinic box = (0 0 0) to (8.0759279 6.9939587 35.664268) with tilt (4.037964 0 0)
  triclinic box = (0 0 0) to (8.0759279 6.9939587 35.664268) with tilt (4.037964 0 0)
  triclinic box = (0 0 0) to (8.0759279 6.9939587 35.664268) with tilt (4.037964 0 0)
  triclinic box = (0 0 0) to (8.0759279 6.9939587 35.664268) with tilt (4.037964 0 0)
  triclinic box = (0 0 0) to (8.0759279 6.9939587 35.664268) with tilt (4.037964 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88254      1160.4954      0              0            
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
change_box all     x final 0 8.07748106491379 y final 0 6.995303800803105 z final 0 35.67112732817786      xy final 4.038740532456894 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0774811 6.9939587 35.664268) with tilt (4.037964 0 0)
  triclinic box = (0 0 0) to (8.0774811 6.9953038 35.664268) with tilt (4.037964 0 0)
  triclinic box = (0 0 0) to (8.0774811 6.9953038 35.671127) with tilt (4.037964 0 0)
  triclinic box = (0 0 0) to (8.0774811 6.9953038 35.671127) with tilt (4.0387405 0 0)
  triclinic box = (0 0 0) to (8.0774811 6.9953038 35.671127) with tilt (4.0387405 0 0)
  triclinic box = (0 0 0) to (8.0774811 6.9953038 35.671127) with tilt (4.0387405 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88371      960.8291       0              0            
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
change_box all     x final 0 8.080316745478482 y final 0 6.997759572209147 z final 0 35.683650034411684      xy final 4.04015837273924 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0803167 6.9953038 35.671127) with tilt (4.0387405 0 0)
  triclinic box = (0 0 0) to (8.0803167 6.9977596 35.671127) with tilt (4.0387405 0 0)
  triclinic box = (0 0 0) to (8.0803167 6.9977596 35.68365) with tilt (4.0387405 0 0)
  triclinic box = (0 0 0) to (8.0803167 6.9977596 35.68365) with tilt (4.0401584 0 0)
  triclinic box = (0 0 0) to (8.0803167 6.9977596 35.68365) with tilt (4.0401584 0 0)
  triclinic box = (0 0 0) to (8.0803167 6.9977596 35.68365) with tilt (4.0401584 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88455      597.09328      0              0            
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
change_box all     x final 0 8.080316745478482 y final 0 6.997759572209147 z final 0 35.683650034411684      xy final 4.04015837273924 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0803167 6.9977596 35.68365) with tilt (4.0401584 0 0)
  triclinic box = (0 0 0) to (8.0803167 6.9977596 35.68365) with tilt (4.0401584 0 0)
  triclinic box = (0 0 0) to (8.0803167 6.9977596 35.68365) with tilt (4.0401584 0 0)
  triclinic box = (0 0 0) to (8.0803167 6.9977596 35.68365) with tilt (4.0401584 0 0)
  triclinic box = (0 0 0) to (8.0803167 6.9977596 35.68365) with tilt (4.0401584 0 0)
  triclinic box = (0 0 0) to (8.0803167 6.9977596 35.68365) with tilt (4.0401584 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88455      597.09328      0              0            
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
change_box all     x final 0 8.081872882101981 y final 0 6.999107226056859 z final 0 35.69052212079491      xy final 4.04093644105099 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0818729 6.9977596 35.68365) with tilt (4.0401584 0 0)
  triclinic box = (0 0 0) to (8.0818729 6.9991072 35.68365) with tilt (4.0401584 0 0)
  triclinic box = (0 0 0) to (8.0818729 6.9991072 35.690522) with tilt (4.0401584 0 0)
  triclinic box = (0 0 0) to (8.0818729 6.9991072 35.690522) with tilt (4.0409364 0 0)
  triclinic box = (0 0 0) to (8.0818729 6.9991072 35.690522) with tilt (4.0409364 0 0)
  triclinic box = (0 0 0) to (8.0818729 6.9991072 35.690522) with tilt (4.0409364 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88623      394.22712      0              0            
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
change_box all     x final 0 8.084458010199704 y final 0 7.001346012661524 z final 0 35.70193835721719      xy final 4.042229005099851 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.084458 6.9991072 35.690522) with tilt (4.0409364 0 0)
  triclinic box = (0 0 0) to (8.084458 7.001346 35.690522) with tilt (4.0409364 0 0)
  triclinic box = (0 0 0) to (8.084458 7.001346 35.701938) with tilt (4.0409364 0 0)
  triclinic box = (0 0 0) to (8.084458 7.001346 35.701938) with tilt (4.042229 0 0)
  triclinic box = (0 0 0) to (8.084458 7.001346 35.701938) with tilt (4.042229 0 0)
  triclinic box = (0 0 0) to (8.084458 7.001346 35.701938) with tilt (4.042229 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88729      58.05748       0              0            
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
change_box all     x final 0 8.084458010199704 y final 0 7.001346012661524 z final 0 35.70193835721719      xy final 4.042229005099851 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.084458 7.001346 35.701938) with tilt (4.042229 0 0)
  triclinic box = (0 0 0) to (8.084458 7.001346 35.701938) with tilt (4.042229 0 0)
  triclinic box = (0 0 0) to (8.084458 7.001346 35.701938) with tilt (4.042229 0 0)
  triclinic box = (0 0 0) to (8.084458 7.001346 35.701938) with tilt (4.042229 0 0)
  triclinic box = (0 0 0) to (8.084458 7.001346 35.701938) with tilt (4.042229 0 0)
  triclinic box = (0 0 0) to (8.084458 7.001346 35.701938) with tilt (4.042229 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88729      58.05748       0              0            
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
change_box all     x final 0 8.0851699271696 y final 0 7.00196255084284 z final 0 35.70508226813152      xy final 4.042584963584799 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0851699 7.001346 35.701938) with tilt (4.042229 0 0)
  triclinic box = (0 0 0) to (8.0851699 7.0019626 35.701938) with tilt (4.042229 0 0)
  triclinic box = (0 0 0) to (8.0851699 7.0019626 35.705082) with tilt (4.042229 0 0)
  triclinic box = (0 0 0) to (8.0851699 7.0019626 35.705082) with tilt (4.042585 0 0)
  triclinic box = (0 0 0) to (8.0851699 7.0019626 35.705082) with tilt (4.042585 0 0)
  triclinic box = (0 0 0) to (8.0851699 7.0019626 35.705082) with tilt (4.042585 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88778     -20.978079      0              0            
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
change_box all     x final 0 8.0851699271696 y final 0 7.00196255084284 z final 0 35.70508226813152      xy final 4.042584963584799 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0851699 7.0019626 35.705082) with tilt (4.042585 0 0)
  triclinic box = (0 0 0) to (8.0851699 7.0019626 35.705082) with tilt (4.042585 0 0)
  triclinic box = (0 0 0) to (8.0851699 7.0019626 35.705082) with tilt (4.042585 0 0)
  triclinic box = (0 0 0) to (8.0851699 7.0019626 35.705082) with tilt (4.042585 0 0)
  triclinic box = (0 0 0) to (8.0851699 7.0019626 35.705082) with tilt (4.042585 0 0)
  triclinic box = (0 0 0) to (8.0851699 7.0019626 35.705082) with tilt (4.042585 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88778     -20.978079      0              0            
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
change_box all     x final 0 8.084995356017382 y final 0 7.001811367790251 z final 0 35.70431134093917      xy final 4.04249767800869 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0849954 7.0019626 35.705082) with tilt (4.042585 0 0)
  triclinic box = (0 0 0) to (8.0849954 7.0018114 35.705082) with tilt (4.042585 0 0)
  triclinic box = (0 0 0) to (8.0849954 7.0018114 35.704311) with tilt (4.042585 0 0)
  triclinic box = (0 0 0) to (8.0849954 7.0018114 35.704311) with tilt (4.0424977 0 0)
  triclinic box = (0 0 0) to (8.0849954 7.0018114 35.704311) with tilt (4.0424977 0 0)
  triclinic box = (0 0 0) to (8.0849954 7.0018114 35.704311) with tilt (4.0424977 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.274 | 3.274 | 3.274 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-319.88783     -0.19697114     0              0            
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
change_box all     x final 0 8.07447630715782 y final 0 9.889173446364437 z final 0 30.489052406381354      xy final -8.074476307157818 xz final 0.0 yz final -0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0744763 7.0018114 35.704311) with tilt (4.0424977 0 0)
  triclinic box = (0 0 0) to (8.0744763 9.8891734 35.704311) with tilt (4.0424977 0 0)
  triclinic box = (0 0 0) to (8.0744763 9.8891734 30.489052) with tilt (4.0424977 0 0)
  triclinic box = (0 0 0) to (8.0744763 9.8891734 30.489052) with tilt (-8.0744763 0 0)
  triclinic box = (0 0 0) to (8.0744763 9.8891734 30.489052) with tilt (-8.0744763 0 0)
  triclinic box = (0 0 0) to (8.0744763 9.8891734 30.489052) with tilt (-8.0744763 0 -0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.1371      -2305.8771      0              0            
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
change_box all     x final 0 7.011564571268151 y final 0 8.587377749091633 z final 0 26.475520087242618      xy final -7.011564571268149 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.0115646 9.8891734 30.489052) with tilt (-8.0744763 0 -0)
  triclinic box = (0 0 0) to (7.0115646 8.5873777 30.489052) with tilt (-8.0744763 0 -0)
  triclinic box = (0 0 0) to (7.0115646 8.5873777 26.47552) with tilt (-8.0744763 0 -0)
  triclinic box = (0 0 0) to (7.0115646 8.5873777 26.47552) with tilt (-7.0115646 0 -0)
  triclinic box = (0 0 0) to (7.0115646 8.5873777 26.47552) with tilt (-7.0115646 0 -0)
  triclinic box = (0 0 0) to (7.0115646 8.5873777 26.47552) with tilt (-7.0115646 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-204.59483      600069.9       0              0            
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
change_box all     x final 0 8.068984973506383 y final 0 9.88244796363774 z final 0 30.468317246216085      xy final -8.068984973506382 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.068985 8.5873777 26.47552) with tilt (-7.0115646 0 0)
  triclinic box = (0 0 0) to (8.068985 9.882448 26.47552) with tilt (-7.0115646 0 0)
  triclinic box = (0 0 0) to (8.068985 9.882448 30.468317) with tilt (-7.0115646 0 0)
  triclinic box = (0 0 0) to (8.068985 9.882448 30.468317) with tilt (-8.068985 0 0)
  triclinic box = (0 0 0) to (8.068985 9.882448 30.468317) with tilt (-8.068985 0 0)
  triclinic box = (0 0 0) to (8.068985 9.882448 30.468317) with tilt (-8.068985 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.13925     -917.98377      0              0            
Loop time of 6.67e-07 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.068984973506383 y final 0 9.88244796363774 z final 0 30.468317246216085      xy final -8.068984973506382 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.068985 9.882448 30.468317) with tilt (-8.068985 0 0)
  triclinic box = (0 0 0) to (8.068985 9.882448 30.468317) with tilt (-8.068985 0 0)
  triclinic box = (0 0 0) to (8.068985 9.882448 30.468317) with tilt (-8.068985 0 0)
  triclinic box = (0 0 0) to (8.068985 9.882448 30.468317) with tilt (-8.068985 0 0)
  triclinic box = (0 0 0) to (8.068985 9.882448 30.468317) with tilt (-8.068985 0 0)
  triclinic box = (0 0 0) to (8.068985 9.882448 30.468317) with tilt (-8.068985 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.13925     -917.98377      0              0            
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
change_box all     x final 0 8.076173055173133 y final 0 9.891251529794234 z final 0 30.495459291136644      xy final -8.07617305517313 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0761731 9.882448 30.468317) with tilt (-8.068985 0 0)
  triclinic box = (0 0 0) to (8.0761731 9.8912515 30.468317) with tilt (-8.068985 0 0)
  triclinic box = (0 0 0) to (8.0761731 9.8912515 30.495459) with tilt (-8.068985 0 0)
  triclinic box = (0 0 0) to (8.0761731 9.8912515 30.495459) with tilt (-8.0761731 0 0)
  triclinic box = (0 0 0) to (8.0761731 9.8912515 30.495459) with tilt (-8.0761731 0 0)
  triclinic box = (0 0 0) to (8.0761731 9.8912515 30.495459) with tilt (-8.0761731 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.39554     -2108.1847      0              0            
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
change_box all     x final 0 8.076173055173133 y final 0 9.891251529794234 z final 0 30.495459291136644      xy final -8.07617305517313 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0761731 9.8912515 30.495459) with tilt (-8.0761731 0 0)
  triclinic box = (0 0 0) to (8.0761731 9.8912515 30.495459) with tilt (-8.0761731 0 0)
  triclinic box = (0 0 0) to (8.0761731 9.8912515 30.495459) with tilt (-8.0761731 0 0)
  triclinic box = (0 0 0) to (8.0761731 9.8912515 30.495459) with tilt (-8.0761731 0 0)
  triclinic box = (0 0 0) to (8.0761731 9.8912515 30.495459) with tilt (-8.0761731 0 0)
  triclinic box = (0 0 0) to (8.0761731 9.8912515 30.495459) with tilt (-8.0761731 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.39554     -2108.1847      0              0            
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
change_box all     x final 0 8.074661622742864 y final 0 9.889400410676808 z final 0 30.489752154125878      xy final -8.074661622742862 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0746616 9.8912515 30.495459) with tilt (-8.0761731 0 0)
  triclinic box = (0 0 0) to (8.0746616 9.8894004 30.495459) with tilt (-8.0761731 0 0)
  triclinic box = (0 0 0) to (8.0746616 9.8894004 30.489752) with tilt (-8.0761731 0 0)
  triclinic box = (0 0 0) to (8.0746616 9.8894004 30.489752) with tilt (-8.0746616 0 0)
  triclinic box = (0 0 0) to (8.0746616 9.8894004 30.489752) with tilt (-8.0746616 0 0)
  triclinic box = (0 0 0) to (8.0746616 9.8894004 30.489752) with tilt (-8.0746616 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.43349     -2432.6884      0              0            
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
change_box all     x final 0 8.074034416629482 y final 0 9.888632243206139 z final 0 30.487383837057      xy final -8.07403441662948 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0740344 9.8894004 30.489752) with tilt (-8.0746616 0 0)
  triclinic box = (0 0 0) to (8.0740344 9.8886322 30.489752) with tilt (-8.0746616 0 0)
  triclinic box = (0 0 0) to (8.0740344 9.8886322 30.487384) with tilt (-8.0746616 0 0)
  triclinic box = (0 0 0) to (8.0740344 9.8886322 30.487384) with tilt (-8.0740344 0 0)
  triclinic box = (0 0 0) to (8.0740344 9.8886322 30.487384) with tilt (-8.0740344 0 0)
  triclinic box = (0 0 0) to (8.0740344 9.8886322 30.487384) with tilt (-8.0740344 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.44636     -2196.2025      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.074034416629482 y final 0 9.888632243206139 z final 0 30.487383837057      xy final -8.07403441662948 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0740344 9.8886322 30.487384) with tilt (-8.0740344 0 0)
  triclinic box = (0 0 0) to (8.0740344 9.8886322 30.487384) with tilt (-8.0740344 0 0)
  triclinic box = (0 0 0) to (8.0740344 9.8886322 30.487384) with tilt (-8.0740344 0 0)
  triclinic box = (0 0 0) to (8.0740344 9.8886322 30.487384) with tilt (-8.0740344 0 0)
  triclinic box = (0 0 0) to (8.0740344 9.8886322 30.487384) with tilt (-8.0740344 0 0)
  triclinic box = (0 0 0) to (8.0740344 9.8886322 30.487384) with tilt (-8.0740344 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.44636     -2196.2025      0              0            
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
change_box all     x final 0 8.07427063665729 y final 0 9.888921552473718 z final 0 30.488275798904844      xy final -8.074270636657289 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0742706 9.8886322 30.487384) with tilt (-8.0740344 0 0)
  triclinic box = (0 0 0) to (8.0742706 9.8889216 30.487384) with tilt (-8.0740344 0 0)
  triclinic box = (0 0 0) to (8.0742706 9.8889216 30.488276) with tilt (-8.0740344 0 0)
  triclinic box = (0 0 0) to (8.0742706 9.8889216 30.488276) with tilt (-8.0742706 0 0)
  triclinic box = (0 0 0) to (8.0742706 9.8889216 30.488276) with tilt (-8.0742706 0 0)
  triclinic box = (0 0 0) to (8.0742706 9.8889216 30.488276) with tilt (-8.0742706 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45007     -2159.6786      0              0            
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
change_box all     x final 0 8.07427063665729 y final 0 9.888921552473718 z final 0 30.488275798904844      xy final -8.074270636657289 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0742706 9.8889216 30.488276) with tilt (-8.0742706 0 0)
  triclinic box = (0 0 0) to (8.0742706 9.8889216 30.488276) with tilt (-8.0742706 0 0)
  triclinic box = (0 0 0) to (8.0742706 9.8889216 30.488276) with tilt (-8.0742706 0 0)
  triclinic box = (0 0 0) to (8.0742706 9.8889216 30.488276) with tilt (-8.0742706 0 0)
  triclinic box = (0 0 0) to (8.0742706 9.8889216 30.488276) with tilt (-8.0742706 0 0)
  triclinic box = (0 0 0) to (8.0742706 9.8889216 30.488276) with tilt (-8.0742706 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45007     -2159.6786      0              0            
Loop time of 1.959e-06 on 1 procs for 0 steps with 80 atoms

306.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.959e-06  |            |       |100.00

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
change_box all     x final 0 8.075262817056839 y final 0 9.890136720329561 z final 0 30.492022251188274      xy final -8.075262817056837 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0752628 9.8889216 30.488276) with tilt (-8.0742706 0 0)
  triclinic box = (0 0 0) to (8.0752628 9.8901367 30.488276) with tilt (-8.0742706 0 0)
  triclinic box = (0 0 0) to (8.0752628 9.8901367 30.492022) with tilt (-8.0742706 0 0)
  triclinic box = (0 0 0) to (8.0752628 9.8901367 30.492022) with tilt (-8.0752628 0 0)
  triclinic box = (0 0 0) to (8.0752628 9.8901367 30.492022) with tilt (-8.0752628 0 0)
  triclinic box = (0 0 0) to (8.0752628 9.8901367 30.492022) with tilt (-8.0752628 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45212     -2337.7132      0              0            
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
change_box all     x final 0 8.076057352256775 y final 0 9.891109823240829 z final 0 30.49502239936237      xy final -8.076057352256774 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0760574 9.8901367 30.492022) with tilt (-8.0752628 0 0)
  triclinic box = (0 0 0) to (8.0760574 9.8911098 30.492022) with tilt (-8.0752628 0 0)
  triclinic box = (0 0 0) to (8.0760574 9.8911098 30.495022) with tilt (-8.0752628 0 0)
  triclinic box = (0 0 0) to (8.0760574 9.8911098 30.495022) with tilt (-8.0760574 0 0)
  triclinic box = (0 0 0) to (8.0760574 9.8911098 30.495022) with tilt (-8.0760574 0 0)
  triclinic box = (0 0 0) to (8.0760574 9.8911098 30.495022) with tilt (-8.0760574 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45315     -2470.3748      0              0            
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
change_box all     x final 0 8.077463461814588 y final 0 9.892831948710375 z final 0 30.500331839425904      xy final -8.077463461814586 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0774635 9.8911098 30.495022) with tilt (-8.0760574 0 0)
  triclinic box = (0 0 0) to (8.0774635 9.8928319 30.495022) with tilt (-8.0760574 0 0)
  triclinic box = (0 0 0) to (8.0774635 9.8928319 30.500332) with tilt (-8.0760574 0 0)
  triclinic box = (0 0 0) to (8.0774635 9.8928319 30.500332) with tilt (-8.0774635 0 0)
  triclinic box = (0 0 0) to (8.0774635 9.8928319 30.500332) with tilt (-8.0774635 0 0)
  triclinic box = (0 0 0) to (8.0774635 9.8928319 30.500332) with tilt (-8.0774635 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45385     -2705.0027      0              0            
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
change_box all     x final 0 8.077463461814588 y final 0 9.892831948710375 z final 0 30.500331839425904      xy final -8.077463461814586 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0774635 9.8928319 30.500332) with tilt (-8.0774635 0 0)
  triclinic box = (0 0 0) to (8.0774635 9.8928319 30.500332) with tilt (-8.0774635 0 0)
  triclinic box = (0 0 0) to (8.0774635 9.8928319 30.500332) with tilt (-8.0774635 0 0)
  triclinic box = (0 0 0) to (8.0774635 9.8928319 30.500332) with tilt (-8.0774635 0 0)
  triclinic box = (0 0 0) to (8.0774635 9.8928319 30.500332) with tilt (-8.0774635 0 0)
  triclinic box = (0 0 0) to (8.0774635 9.8928319 30.500332) with tilt (-8.0774635 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45385     -2705.0027      0              0            
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
change_box all     x final 0 8.077968529076522 y final 0 9.893450527249138 z final 0 30.50223896276519      xy final -8.07796852907652 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0779685 9.8928319 30.500332) with tilt (-8.0774635 0 0)
  triclinic box = (0 0 0) to (8.0779685 9.8934505 30.500332) with tilt (-8.0774635 0 0)
  triclinic box = (0 0 0) to (8.0779685 9.8934505 30.502239) with tilt (-8.0774635 0 0)
  triclinic box = (0 0 0) to (8.0779685 9.8934505 30.502239) with tilt (-8.0779685 0 0)
  triclinic box = (0 0 0) to (8.0779685 9.8934505 30.502239) with tilt (-8.0779685 0 0)
  triclinic box = (0 0 0) to (8.0779685 9.8934505 30.502239) with tilt (-8.0779685 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45518     -2758.4568      0              0            
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
change_box all     x final 0 8.079260180197645 y final 0 9.895032470335362 z final 0 30.507116210183426      xy final -8.079260180197643 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0792602 9.8934505 30.502239) with tilt (-8.0779685 0 0)
  triclinic box = (0 0 0) to (8.0792602 9.8950325 30.502239) with tilt (-8.0779685 0 0)
  triclinic box = (0 0 0) to (8.0792602 9.8950325 30.507116) with tilt (-8.0779685 0 0)
  triclinic box = (0 0 0) to (8.0792602 9.8950325 30.507116) with tilt (-8.0792602 0 0)
  triclinic box = (0 0 0) to (8.0792602 9.8950325 30.507116) with tilt (-8.0792602 0 0)
  triclinic box = (0 0 0) to (8.0792602 9.8950325 30.507116) with tilt (-8.0792602 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.4566      -2895.2554      0              0            
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
change_box all     x final 0 8.079260180197645 y final 0 9.895032470335362 z final 0 30.507116210183426      xy final -8.079260180197643 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0792602 9.8950325 30.507116) with tilt (-8.0792602 0 0)
  triclinic box = (0 0 0) to (8.0792602 9.8950325 30.507116) with tilt (-8.0792602 0 0)
  triclinic box = (0 0 0) to (8.0792602 9.8950325 30.507116) with tilt (-8.0792602 0 0)
  triclinic box = (0 0 0) to (8.0792602 9.8950325 30.507116) with tilt (-8.0792602 0 0)
  triclinic box = (0 0 0) to (8.0792602 9.8950325 30.507116) with tilt (-8.0792602 0 0)
  triclinic box = (0 0 0) to (8.0792602 9.8950325 30.507116) with tilt (-8.0792602 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.4566      -2895.2554      0              0            
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
change_box all     x final 0 8.079197620708007 y final 0 9.89495585092127 z final 0 30.506879986868682      xy final -8.079197620708005 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0791976 9.8950325 30.507116) with tilt (-8.0792602 0 0)
  triclinic box = (0 0 0) to (8.0791976 9.8949559 30.507116) with tilt (-8.0792602 0 0)
  triclinic box = (0 0 0) to (8.0791976 9.8949559 30.50688) with tilt (-8.0792602 0 0)
  triclinic box = (0 0 0) to (8.0791976 9.8949559 30.50688) with tilt (-8.0791976 0 0)
  triclinic box = (0 0 0) to (8.0791976 9.8949559 30.50688) with tilt (-8.0791976 0 0)
  triclinic box = (0 0 0) to (8.0791976 9.8949559 30.50688) with tilt (-8.0791976 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45674     -2882.0938      0              0            
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
change_box all     x final 0 8.079197620708007 y final 0 9.89495585092127 z final 0 30.506879986868682      xy final -8.079197620708005 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0791976 9.8949559 30.50688) with tilt (-8.0791976 0 0)
  triclinic box = (0 0 0) to (8.0791976 9.8949559 30.50688) with tilt (-8.0791976 0 0)
  triclinic box = (0 0 0) to (8.0791976 9.8949559 30.50688) with tilt (-8.0791976 0 0)
  triclinic box = (0 0 0) to (8.0791976 9.8949559 30.50688) with tilt (-8.0791976 0 0)
  triclinic box = (0 0 0) to (8.0791976 9.8949559 30.50688) with tilt (-8.0791976 0 0)
  triclinic box = (0 0 0) to (8.0791976 9.8949559 30.50688) with tilt (-8.0791976 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45674     -2882.0938      0              0            
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
change_box all     x final 0 8.078981976537014 y final 0 9.894691741828801 z final 0 30.506065719022644      xy final -8.078981976537012 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.078982 9.8949559 30.50688) with tilt (-8.0791976 0 0)
  triclinic box = (0 0 0) to (8.078982 9.8946917 30.50688) with tilt (-8.0791976 0 0)
  triclinic box = (0 0 0) to (8.078982 9.8946917 30.506066) with tilt (-8.0791976 0 0)
  triclinic box = (0 0 0) to (8.078982 9.8946917 30.506066) with tilt (-8.078982 0 0)
  triclinic box = (0 0 0) to (8.078982 9.8946917 30.506066) with tilt (-8.078982 0 0)
  triclinic box = (0 0 0) to (8.078982 9.8946917 30.506066) with tilt (-8.078982 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45683     -2860.1758      0              0            
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
change_box all     x final 0 8.078981976537014 y final 0 9.894691741828801 z final 0 30.506065719022644      xy final -8.078981976537012 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.078982 9.8946917 30.506066) with tilt (-8.078982 0 0)
  triclinic box = (0 0 0) to (8.078982 9.8946917 30.506066) with tilt (-8.078982 0 0)
  triclinic box = (0 0 0) to (8.078982 9.8946917 30.506066) with tilt (-8.078982 0 0)
  triclinic box = (0 0 0) to (8.078982 9.8946917 30.506066) with tilt (-8.078982 0 0)
  triclinic box = (0 0 0) to (8.078982 9.8946917 30.506066) with tilt (-8.078982 0 0)
  triclinic box = (0 0 0) to (8.078982 9.8946917 30.506066) with tilt (-8.078982 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45683     -2860.1758      0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.078668288649107 y final 0 9.89430755419687 z final 0 30.504881240167318      xy final -8.078668288649105 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0786683 9.8946917 30.506066) with tilt (-8.078982 0 0)
  triclinic box = (0 0 0) to (8.0786683 9.8943076 30.506066) with tilt (-8.078982 0 0)
  triclinic box = (0 0 0) to (8.0786683 9.8943076 30.504881) with tilt (-8.078982 0 0)
  triclinic box = (0 0 0) to (8.0786683 9.8943076 30.504881) with tilt (-8.0786683 0 0)
  triclinic box = (0 0 0) to (8.0786683 9.8943076 30.504881) with tilt (-8.0786683 0 0)
  triclinic box = (0 0 0) to (8.0786683 9.8943076 30.504881) with tilt (-8.0786683 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.4569      -2839.9855      0              0            
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
change_box all     x final 0 8.078108268582383 y final 0 9.893621672492273 z final 0 30.502766616195007      xy final -8.078108268582382 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0781083 9.8943076 30.504881) with tilt (-8.0786683 0 0)
  triclinic box = (0 0 0) to (8.0781083 9.8936217 30.504881) with tilt (-8.0786683 0 0)
  triclinic box = (0 0 0) to (8.0781083 9.8936217 30.502767) with tilt (-8.0786683 0 0)
  triclinic box = (0 0 0) to (8.0781083 9.8936217 30.502767) with tilt (-8.0781083 0 0)
  triclinic box = (0 0 0) to (8.0781083 9.8936217 30.502767) with tilt (-8.0781083 0 0)
  triclinic box = (0 0 0) to (8.0781083 9.8936217 30.502767) with tilt (-8.0781083 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45694     -2804.0222      0              0            
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
change_box all     x final 0 8.078108268582383 y final 0 9.893621672492273 z final 0 30.502766616195007      xy final -8.078108268582382 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0781083 9.8936217 30.502767) with tilt (-8.0781083 0 0)
  triclinic box = (0 0 0) to (8.0781083 9.8936217 30.502767) with tilt (-8.0781083 0 0)
  triclinic box = (0 0 0) to (8.0781083 9.8936217 30.502767) with tilt (-8.0781083 0 0)
  triclinic box = (0 0 0) to (8.0781083 9.8936217 30.502767) with tilt (-8.0781083 0 0)
  triclinic box = (0 0 0) to (8.0781083 9.8936217 30.502767) with tilt (-8.0781083 0 0)
  triclinic box = (0 0 0) to (8.0781083 9.8936217 30.502767) with tilt (-8.0781083 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45694     -2804.0222      0              0            
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
change_box all     x final 0 8.077777120579539 y final 0 9.893216100674119 z final 0 30.501516208313568      xy final -8.077777120579537 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0777771 9.8936217 30.502767) with tilt (-8.0781083 0 0)
  triclinic box = (0 0 0) to (8.0777771 9.8932161 30.502767) with tilt (-8.0781083 0 0)
  triclinic box = (0 0 0) to (8.0777771 9.8932161 30.501516) with tilt (-8.0781083 0 0)
  triclinic box = (0 0 0) to (8.0777771 9.8932161 30.501516) with tilt (-8.0777771 0 0)
  triclinic box = (0 0 0) to (8.0777771 9.8932161 30.501516) with tilt (-8.0777771 0 0)
  triclinic box = (0 0 0) to (8.0777771 9.8932161 30.501516) with tilt (-8.0777771 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45702     -2786.7994      0              0            
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
change_box all     x final 0 8.07716842113279 y final 0 9.892470599148494 z final 0 30.499217772027958      xy final -8.077168421132788 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.0771684 9.8932161 30.501516) with tilt (-8.0777771 0 0)
  triclinic box = (0 0 0) to (8.0771684 9.8924706 30.501516) with tilt (-8.0777771 0 0)
  triclinic box = (0 0 0) to (8.0771684 9.8924706 30.499218) with tilt (-8.0777771 0 0)
  triclinic box = (0 0 0) to (8.0771684 9.8924706 30.499218) with tilt (-8.0771684 0 0)
  triclinic box = (0 0 0) to (8.0771684 9.8924706 30.499218) with tilt (-8.0771684 0 0)
  triclinic box = (0 0 0) to (8.0771684 9.8924706 30.499218) with tilt (-8.0771684 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-315.45707     -2755.168       0              0            
Loop time of 6.67e-07 on 1 procs for 0 steps with 80 atoms

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
Total wall time: 0:00:03
