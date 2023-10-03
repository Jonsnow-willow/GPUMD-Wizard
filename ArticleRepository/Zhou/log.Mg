LAMMPS (23 Jun 2022 - Update 4)
units metal
atom_style atomic
atom_modify map array sort 0 0
boundary p p p
box tilt large
region cell prism    0 3.195 0 2.7669511650912813 0 5.186     -1.5975 0.0 0.0     units box
create_box 1 cell
Created triclinic box = (0 0 0) to (3.195 2.7669512 5.186) with tilt (-1.5975 0 0)
  1 by 1 by 1 MPI processor grid
echo none
Created 2 atoms
  using lattice units in triclinic box = (0 0 0) to (3.195 2.7669512 5.186) with tilt (-1.5975 0 0)
  create_atoms CPU = 0.000 seconds
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
pair_style eam/alloy
pair_coeff * * potentials/Mg.eam.alloy Mg
Reading eam/alloy potential file potentials/Mg.eam.alloy with DATE: 2018-03-30
mass 1 24.304999993747674835731231723912
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
  master list distance cutoff = 9.147124
  ghost atom cutoff = 9.147124
  binsize = 4.573562, bins = 2 1 2
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair eam/alloy, perpetual
      attributes: half, newton on
      pair build: half/bin/newton/tri
      stencil: half/bin/3d/tri
      bin: standard
Per MPI rank memory allocation (min/avg/max) = 3.249 | 3.249 | 3.249 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0996994      2088.1338      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 2 atoms

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

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            134 ave         134 max         134 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 134
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.219642279529805 y final 0 2.78829200517125 z final 0 5.2259983917501005      xy final -1.6098211397649025 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2196423 2.7669512 5.186) with tilt (-1.5975 0 0)
  triclinic box = (0 0 0) to (3.2196423 2.788292 5.186) with tilt (-1.5975 0 0)
  triclinic box = (0 0 0) to (3.2196423 2.788292 5.2259984) with tilt (-1.5975 0 0)
  triclinic box = (0 0 0) to (3.2196423 2.788292 5.2259984) with tilt (-1.6098211 0 0)
  triclinic box = (0 0 0) to (3.2196423 2.788292 5.2259984) with tilt (-1.6098211 0 0)
  triclinic box = (0 0 0) to (3.2196423 2.788292 5.2259984) with tilt (-1.6098211 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.249 | 3.249 | 3.249 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0987249     -5719.4725      0              0            
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
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            134 ave         134 max         134 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 134
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.202728473465918 y final 0 2.77364421944524 z final 0 5.198544558182864      xy final -1.601364236732959 xz final 0.0 yz final 0.0 units box
Changing box ...
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (3.2027285 2.788292 5.2259984) with tilt (-1.6098211 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (3.2027285 2.7736442 5.2259984) with tilt (-1.6098211 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (3.2027285 2.7736442 5.1985446) with tilt (-1.6098211 0 0)
  triclinic box = (0 0 0) to (3.2027285 2.7736442 5.1985446) with tilt (-1.6013642 0 0)
  triclinic box = (0 0 0) to (3.2027285 2.7736442 5.1985446) with tilt (-1.6013642 0 0)
  triclinic box = (0 0 0) to (3.2027285 2.7736442 5.1985446) with tilt (-1.6013642 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.249 | 3.249 | 3.249 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0999416     -435.03273      0              0            
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
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            134 ave         134 max         134 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 134
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 2.8824556261193264 y final 0 2.496279797500716 z final 0 4.678690102364578      xy final -1.4412278130596632 xz final 0.0 yz final 0.0 units box
Changing box ...
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (2.8824556 2.7736442 5.1985446) with tilt (-1.6013642 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (2.8824556 2.4962798 5.1985446) with tilt (-1.6013642 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (2.8824556 2.4962798 4.6786901) with tilt (-1.6013642 0 0)
  triclinic box = (0 0 0) to (2.8824556 2.4962798 4.6786901) with tilt (-1.4412278 0 0)
  triclinic box = (0 0 0) to (2.8824556 2.4962798 4.6786901) with tilt (-1.4412278 0 0)
  triclinic box = (0 0 0) to (2.8824556 2.4962798 4.6786901) with tilt (-1.4412278 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2.3753066      257650.07      0              0            
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
Nghost:            574 ave         574 max         574 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            194 ave         194 max         194 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 194
Ave neighs/atom = 97
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 2.9144829108539856 y final 0 2.5240162396951686 z final 0 4.730675547946407      xy final -1.4572414554269928 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9144829 2.4962798 4.6786901) with tilt (-1.4412278 0 0)
  triclinic box = (0 0 0) to (2.9144829 2.5240162 4.6786901) with tilt (-1.4412278 0 0)
  triclinic box = (0 0 0) to (2.9144829 2.5240162 4.7306755) with tilt (-1.4412278 0 0)
  triclinic box = (0 0 0) to (2.9144829 2.5240162 4.7306755) with tilt (-1.4572415 0 0)
  triclinic box = (0 0 0) to (2.9144829 2.5240162 4.7306755) with tilt (-1.4572415 0 0)
  triclinic box = (0 0 0) to (2.9144829 2.5240162 4.7306755) with tilt (-1.4572415 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2.5429953      210011.61      0              0            
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
Nghost:            574 ave         574 max         574 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            182 ave         182 max         182 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 182
Ave neighs/atom = 91
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 2.946510195588645 y final 0 2.5517526818896212 z final 0 4.782660993528236      xy final -1.4732550977943224 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9465102 2.5240162 4.7306755) with tilt (-1.4572415 0 0)
  triclinic box = (0 0 0) to (2.9465102 2.5517527 4.7306755) with tilt (-1.4572415 0 0)
  triclinic box = (0 0 0) to (2.9465102 2.5517527 4.782661) with tilt (-1.4572415 0 0)
  triclinic box = (0 0 0) to (2.9465102 2.5517527 4.782661) with tilt (-1.4732551 0 0)
  triclinic box = (0 0 0) to (2.9465102 2.5517527 4.782661) with tilt (-1.4732551 0 0)
  triclinic box = (0 0 0) to (2.9465102 2.5517527 4.782661) with tilt (-1.4732551 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2.6821013      169233.07      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 2 atoms

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

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            574 ave         574 max         574 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            182 ave         182 max         182 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 182
Ave neighs/atom = 91
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 2.978537480323304 y final 0 2.5794891240840734 z final 0 4.834646439110064      xy final -1.489268740161652 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9785375 2.5517527 4.782661) with tilt (-1.4732551 0 0)
  triclinic box = (0 0 0) to (2.9785375 2.5794891 4.782661) with tilt (-1.4732551 0 0)
  triclinic box = (0 0 0) to (2.9785375 2.5794891 4.8346464) with tilt (-1.4732551 0 0)
  triclinic box = (0 0 0) to (2.9785375 2.5794891 4.8346464) with tilt (-1.4892687 0 0)
  triclinic box = (0 0 0) to (2.9785375 2.5794891 4.8346464) with tilt (-1.4892687 0 0)
  triclinic box = (0 0 0) to (2.9785375 2.5794891 4.8346464) with tilt (-1.4892687 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2.796002       134291.43      0              0            
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
Nghost:            574 ave         574 max         574 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            158 ave         158 max         158 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 158
Ave neighs/atom = 79
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.010564765057963 y final 0 2.607225566278526 z final 0 4.886631884691893      xy final -1.5052823825289816 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0105648 2.5794891 4.8346464) with tilt (-1.4892687 0 0)
  triclinic box = (0 0 0) to (3.0105648 2.6072256 4.8346464) with tilt (-1.4892687 0 0)
  triclinic box = (0 0 0) to (3.0105648 2.6072256 4.8866319) with tilt (-1.4892687 0 0)
  triclinic box = (0 0 0) to (3.0105648 2.6072256 4.8866319) with tilt (-1.5052824 0 0)
  triclinic box = (0 0 0) to (3.0105648 2.6072256 4.8866319) with tilt (-1.5052824 0 0)
  triclinic box = (0 0 0) to (3.0105648 2.6072256 4.8866319) with tilt (-1.5052824 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2.8876318      104394.32      0              0            
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
Nghost:            574 ave         574 max         574 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            158 ave         158 max         158 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 158
Ave neighs/atom = 79
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.0425920497926224 y final 0 2.6349620084729786 z final 0 4.938617330273722      xy final -1.5212960248963112 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.042592 2.6072256 4.8866319) with tilt (-1.5052824 0 0)
  triclinic box = (0 0 0) to (3.042592 2.634962 4.8866319) with tilt (-1.5052824 0 0)
  triclinic box = (0 0 0) to (3.042592 2.634962 4.9386173) with tilt (-1.5052824 0 0)
  triclinic box = (0 0 0) to (3.042592 2.634962 4.9386173) with tilt (-1.521296 0 0)
  triclinic box = (0 0 0) to (3.042592 2.634962 4.9386173) with tilt (-1.521296 0 0)
  triclinic box = (0 0 0) to (3.042592 2.634962 4.9386173) with tilt (-1.521296 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2.9595956      78882.115      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 2 atoms

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

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            574 ave         574 max         574 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            158 ave         158 max         158 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 158
Ave neighs/atom = 79
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.0746193345272816 y final 0 2.6626984506674307 z final 0 4.9906027758555505      xy final -1.5373096672636408 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0746193 2.634962 4.9386173) with tilt (-1.521296 0 0)
  triclinic box = (0 0 0) to (3.0746193 2.6626985 4.9386173) with tilt (-1.521296 0 0)
  triclinic box = (0 0 0) to (3.0746193 2.6626985 4.9906028) with tilt (-1.521296 0 0)
  triclinic box = (0 0 0) to (3.0746193 2.6626985 4.9906028) with tilt (-1.5373097 0 0)
  triclinic box = (0 0 0) to (3.0746193 2.6626985 4.9906028) with tilt (-1.5373097 0 0)
  triclinic box = (0 0 0) to (3.0746193 2.6626985 4.9906028) with tilt (-1.5373097 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0143099      57156.789      0              0            
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
Nghost:            574 ave         574 max         574 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            146 ave         146 max         146 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 146
Ave neighs/atom = 73
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.106646619261941 y final 0 2.6904348928618833 z final 0 5.042588221437379      xy final -1.5533233096309704 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1066466 2.6626985 4.9906028) with tilt (-1.5373097 0 0)
  triclinic box = (0 0 0) to (3.1066466 2.6904349 4.9906028) with tilt (-1.5373097 0 0)
  triclinic box = (0 0 0) to (3.1066466 2.6904349 5.0425882) with tilt (-1.5373097 0 0)
  triclinic box = (0 0 0) to (3.1066466 2.6904349 5.0425882) with tilt (-1.5533233 0 0)
  triclinic box = (0 0 0) to (3.1066466 2.6904349 5.0425882) with tilt (-1.5533233 0 0)
  triclinic box = (0 0 0) to (3.1066466 2.6904349 5.0425882) with tilt (-1.5533233 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0538437      38758.932      0              0            
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
Nghost:            574 ave         574 max         574 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            146 ave         146 max         146 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 146
Ave neighs/atom = 73
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.1386739039966 y final 0 2.7181713350563355 z final 0 5.094573667019207      xy final -1.5693369519983 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1386739 2.6904349 5.0425882) with tilt (-1.5533233 0 0)
  triclinic box = (0 0 0) to (3.1386739 2.7181713 5.0425882) with tilt (-1.5533233 0 0)
  triclinic box = (0 0 0) to (3.1386739 2.7181713 5.0945737) with tilt (-1.5533233 0 0)
  triclinic box = (0 0 0) to (3.1386739 2.7181713 5.0945737) with tilt (-1.569337 0 0)
  triclinic box = (0 0 0) to (3.1386739 2.7181713 5.0945737) with tilt (-1.569337 0 0)
  triclinic box = (0 0 0) to (3.1386739 2.7181713 5.0945737) with tilt (-1.569337 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0801472      23549.917      0              0            
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
Nghost:            574 ave         574 max         574 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            146 ave         146 max         146 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 146
Ave neighs/atom = 73
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.1707011887312593 y final 0 2.745907777250788 z final 0 5.146559112601036      xy final -1.5853505943656296 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1707012 2.7181713 5.0945737) with tilt (-1.569337 0 0)
  triclinic box = (0 0 0) to (3.1707012 2.7459078 5.0945737) with tilt (-1.569337 0 0)
  triclinic box = (0 0 0) to (3.1707012 2.7459078 5.1465591) with tilt (-1.569337 0 0)
  triclinic box = (0 0 0) to (3.1707012 2.7459078 5.1465591) with tilt (-1.5853506 0 0)
  triclinic box = (0 0 0) to (3.1707012 2.7459078 5.1465591) with tilt (-1.5853506 0 0)
  triclinic box = (0 0 0) to (3.1707012 2.7459078 5.1465591) with tilt (-1.5853506 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0951582      10720.346      0              0            
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
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            146 ave         146 max         146 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 146
Ave neighs/atom = 73
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.202728473465918 y final 0 2.77364421944524 z final 0 5.198544558182864      xy final -1.601364236732959 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2027285 2.7459078 5.1465591) with tilt (-1.5853506 0 0)
  triclinic box = (0 0 0) to (3.2027285 2.7736442 5.1465591) with tilt (-1.5853506 0 0)
  triclinic box = (0 0 0) to (3.2027285 2.7736442 5.1985446) with tilt (-1.5853506 0 0)
  triclinic box = (0 0 0) to (3.2027285 2.7736442 5.1985446) with tilt (-1.6013642 0 0)
  triclinic box = (0 0 0) to (3.2027285 2.7736442 5.1985446) with tilt (-1.6013642 0 0)
  triclinic box = (0 0 0) to (3.2027285 2.7736442 5.1985446) with tilt (-1.6013642 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0999416     -435.03273      0              0            
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
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            134 ave         134 max         134 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 134
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.234755758200578 y final 0 2.8013806616396932 z final 0 5.250530003764695      xy final -1.617377879100289 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2347558 2.7736442 5.1985446) with tilt (-1.6013642 0 0)
  triclinic box = (0 0 0) to (3.2347558 2.8013807 5.1985446) with tilt (-1.6013642 0 0)
  triclinic box = (0 0 0) to (3.2347558 2.8013807 5.25053) with tilt (-1.6013642 0 0)
  triclinic box = (0 0 0) to (3.2347558 2.8013807 5.25053) with tilt (-1.6173779 0 0)
  triclinic box = (0 0 0) to (3.2347558 2.8013807 5.25053) with tilt (-1.6173779 0 0)
  triclinic box = (0 0 0) to (3.2347558 2.8013807 5.25053) with tilt (-1.6173779 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0955836     -10090.971      0              0            
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
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            134 ave         134 max         134 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 134
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.2667830429352365 y final 0 2.829117103834145 z final 0 5.302515449346521      xy final -1.6333915214676182 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.266783 2.8013807 5.25053) with tilt (-1.6173779 0 0)
  triclinic box = (0 0 0) to (3.266783 2.8291171 5.25053) with tilt (-1.6173779 0 0)
  triclinic box = (0 0 0) to (3.266783 2.8291171 5.3025154) with tilt (-1.6173779 0 0)
  triclinic box = (0 0 0) to (3.266783 2.8291171 5.3025154) with tilt (-1.6333915 0 0)
  triclinic box = (0 0 0) to (3.266783 2.8291171 5.3025154) with tilt (-1.6333915 0 0)
  triclinic box = (0 0 0) to (3.266783 2.8291171 5.3025154) with tilt (-1.6333915 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0829995     -18327.464      0              0            
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
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            134 ave         134 max         134 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 134
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.2988103276698966 y final 0 2.856853546028598 z final 0 5.354500894928352      xy final -1.6494051638349483 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2988103 2.8291171 5.3025154) with tilt (-1.6333915 0 0)
  triclinic box = (0 0 0) to (3.2988103 2.8568535 5.3025154) with tilt (-1.6333915 0 0)
  triclinic box = (0 0 0) to (3.2988103 2.8568535 5.3545009) with tilt (-1.6333915 0 0)
  triclinic box = (0 0 0) to (3.2988103 2.8568535 5.3545009) with tilt (-1.6494052 0 0)
  triclinic box = (0 0 0) to (3.2988103 2.8568535 5.3545009) with tilt (-1.6494052 0 0)
  triclinic box = (0 0 0) to (3.2988103 2.8568535 5.3545009) with tilt (-1.6494052 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0631499     -25297.507      0              0            
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
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            134 ave         134 max         134 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 134
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.330837612404555 y final 0 2.88458998822305 z final 0 5.406486340510179      xy final -1.6654188062022774 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3308376 2.8568535 5.3545009) with tilt (-1.6494052 0 0)
  triclinic box = (0 0 0) to (3.3308376 2.88459 5.3545009) with tilt (-1.6494052 0 0)
  triclinic box = (0 0 0) to (3.3308376 2.88459 5.4064863) with tilt (-1.6494052 0 0)
  triclinic box = (0 0 0) to (3.3308376 2.88459 5.4064863) with tilt (-1.6654188 0 0)
  triclinic box = (0 0 0) to (3.3308376 2.88459 5.4064863) with tilt (-1.6654188 0 0)
  triclinic box = (0 0 0) to (3.3308376 2.88459 5.4064863) with tilt (-1.6654188 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0366595     -31871.947      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 2 atoms

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

Nlocal:              2 ave           2 max           2 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            134 ave         134 max         134 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 134
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.362864897139215 y final 0 2.912326430417503 z final 0 5.458471786092009      xy final -1.6814324485696075 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3628649 2.88459 5.4064863) with tilt (-1.6654188 0 0)
  triclinic box = (0 0 0) to (3.3628649 2.9123264 5.4064863) with tilt (-1.6654188 0 0)
  triclinic box = (0 0 0) to (3.3628649 2.9123264 5.4584718) with tilt (-1.6654188 0 0)
  triclinic box = (0 0 0) to (3.3628649 2.9123264 5.4584718) with tilt (-1.6814324 0 0)
  triclinic box = (0 0 0) to (3.3628649 2.9123264 5.4584718) with tilt (-1.6814324 0 0)
  triclinic box = (0 0 0) to (3.3628649 2.9123264 5.4584718) with tilt (-1.6814324 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-3.0035365     -38225.061      0              0            
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
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            134 ave         134 max         134 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 134
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.3948921818738733 y final 0 2.940062872611955 z final 0 5.510457231673836      xy final -1.6974460909369367 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3948922 2.9123264 5.4584718) with tilt (-1.6814324 0 0)
  triclinic box = (0 0 0) to (3.3948922 2.9400629 5.4584718) with tilt (-1.6814324 0 0)
  triclinic box = (0 0 0) to (3.3948922 2.9400629 5.5104572) with tilt (-1.6814324 0 0)
  triclinic box = (0 0 0) to (3.3948922 2.9400629 5.5104572) with tilt (-1.6974461 0 0)
  triclinic box = (0 0 0) to (3.3948922 2.9400629 5.5104572) with tilt (-1.6974461 0 0)
  triclinic box = (0 0 0) to (3.3948922 2.9400629 5.5104572) with tilt (-1.6974461 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2.9639252     -44132.607      0              0            
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
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            128 ave         128 max         128 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 128
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.4269194666085334 y final 0 2.967799314806408 z final 0 5.5624426772556665      xy final -1.7134597333042667 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.4269195 2.9400629 5.5104572) with tilt (-1.6974461 0 0)
  triclinic box = (0 0 0) to (3.4269195 2.9677993 5.5104572) with tilt (-1.6974461 0 0)
  triclinic box = (0 0 0) to (3.4269195 2.9677993 5.5624427) with tilt (-1.6974461 0 0)
  triclinic box = (0 0 0) to (3.4269195 2.9677993 5.5624427) with tilt (-1.7134597 0 0)
  triclinic box = (0 0 0) to (3.4269195 2.9677993 5.5624427) with tilt (-1.7134597 0 0)
  triclinic box = (0 0 0) to (3.4269195 2.9677993 5.5624427) with tilt (-1.7134597 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2.9181391     -49460.053      0              0            
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
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            128 ave         128 max         128 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 128
Ave neighs/atom = 64
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.4589467513431917 y final 0 2.9955357570008596 z final 0 5.614428122837494      xy final -1.7294733756715959 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.4589468 2.9677993 5.5624427) with tilt (-1.7134597 0 0)
  triclinic box = (0 0 0) to (3.4589468 2.9955358 5.5624427) with tilt (-1.7134597 0 0)
  triclinic box = (0 0 0) to (3.4589468 2.9955358 5.6144281) with tilt (-1.7134597 0 0)
  triclinic box = (0 0 0) to (3.4589468 2.9955358 5.6144281) with tilt (-1.7294734 0 0)
  triclinic box = (0 0 0) to (3.4589468 2.9955358 5.6144281) with tilt (-1.7294734 0 0)
  triclinic box = (0 0 0) to (3.4589468 2.9955358 5.6144281) with tilt (-1.7294734 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2.8665968     -54135.074      0              0            
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
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            116 ave         116 max         116 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 116
Ave neighs/atom = 58
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.490974036077852 y final 0 3.0232721991953126 z final 0 5.6664135684193235      xy final -1.745487018038926 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.490974 2.9955358 5.6144281) with tilt (-1.7294734 0 0)
  triclinic box = (0 0 0) to (3.490974 3.0232722 5.6144281) with tilt (-1.7294734 0 0)
  triclinic box = (0 0 0) to (3.490974 3.0232722 5.6664136) with tilt (-1.7294734 0 0)
  triclinic box = (0 0 0) to (3.490974 3.0232722 5.6664136) with tilt (-1.745487 0 0)
  triclinic box = (0 0 0) to (3.490974 3.0232722 5.6664136) with tilt (-1.745487 0 0)
  triclinic box = (0 0 0) to (3.490974 3.0232722 5.6664136) with tilt (-1.745487 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.252 | 3.252 | 3.252 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2.8097835     -58123.935      0              0            
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
Nghost:            499 ave         499 max         499 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            104 ave         104 max         104 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 104
Ave neighs/atom = 52
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 3.52300132081251 y final 0 3.0510086413897644 z final 0 5.718399014001151      xy final -1.761500660406255 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.5230013 3.0232722 5.6664136) with tilt (-1.745487 0 0)
  triclinic box = (0 0 0) to (3.5230013 3.0510086 5.6664136) with tilt (-1.745487 0 0)
  triclinic box = (0 0 0) to (3.5230013 3.0510086 5.718399) with tilt (-1.745487 0 0)
  triclinic box = (0 0 0) to (3.5230013 3.0510086 5.718399) with tilt (-1.7615007 0 0)
  triclinic box = (0 0 0) to (3.5230013 3.0510086 5.718399) with tilt (-1.7615007 0 0)
  triclinic box = (0 0 0) to (3.5230013 3.0510086 5.718399) with tilt (-1.7615007 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.236 | 3.236 | 3.236 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-2.7482316     -61412.988      0              0            
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
Nghost:            387 ave         387 max         387 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:            104 ave         104 max         104 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 104
Ave neighs/atom = 52
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  2 settings made for x
  2 settings made for y
  2 settings made for z
change_box all     x final 0 9.608185420397755 y final 0 8.320932658335721 z final 0 15.595633674548594      xy final -4.804092710198877 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6081854 3.0510086 5.718399) with tilt (-1.7615007 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 5.718399) with tilt (-1.7615007 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-1.7615007 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
echo none
Created 52 atoms
  using lattice units in triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
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
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.698423     -435.03273      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.4160217119898 y final 0 8.320932658335721 z final 0 15.595633674548594      xy final -4.7080108559949 xz final 0.0 yz final 0.0 units box
Changing box ...
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (9.4160217 8.3209327 15.595634) with tilt (-4.8040927 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (9.4160217 8.3209327 15.595634) with tilt (-4.8040927 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (9.4160217 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.4160217 8.3209327 15.595634) with tilt (-4.7080109 0 0)
  triclinic box = (0 0 0) to (9.4160217 8.3209327 15.595634) with tilt (-4.7080109 0 0)
  triclinic box = (0 0 0) to (9.4160217 8.3209327 15.595634) with tilt (-4.7080109 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.605577      13053.199      0              0            
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
Nghost:           1176 ave        1176 max        1176 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.512103566193778 y final 0 8.320932658335721 z final 0 15.595633674548594      xy final -4.756051783096889 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.5121036 8.3209327 15.595634) with tilt (-4.7080109 0 0)
  triclinic box = (0 0 0) to (9.5121036 8.3209327 15.595634) with tilt (-4.7080109 0 0)
  triclinic box = (0 0 0) to (9.5121036 8.3209327 15.595634) with tilt (-4.7080109 0 0)
  triclinic box = (0 0 0) to (9.5121036 8.3209327 15.595634) with tilt (-4.7560518 0 0)
  triclinic box = (0 0 0) to (9.5121036 8.3209327 15.595634) with tilt (-4.7560518 0 0)
  triclinic box = (0 0 0) to (9.5121036 8.3209327 15.595634) with tilt (-4.7560518 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.678653      5874.9749      0              0            
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
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608185420397755 y final 0 8.320932658335721 z final 0 15.595633674548594      xy final -4.804092710198877 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.7560518 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.7560518 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.7560518 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.698423     -435.03273      0              0            
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
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.704267274601731 y final 0 8.320932658335721 z final 0 15.595633674548594      xy final -4.852133637300866 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.7042673 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.7042673 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.7042673 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.7042673 8.3209327 15.595634) with tilt (-4.8521336 0 0)
  triclinic box = (0 0 0) to (9.7042673 8.3209327 15.595634) with tilt (-4.8521336 0 0)
  triclinic box = (0 0 0) to (9.7042673 8.3209327 15.595634) with tilt (-4.8521336 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.672752     -6048.1274      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.80034912880571 y final 0 8.320932658335721 z final 0 15.595633674548594      xy final -4.900174564402855 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.8003491 8.3209327 15.595634) with tilt (-4.8521336 0 0)
  triclinic box = (0 0 0) to (9.8003491 8.3209327 15.595634) with tilt (-4.8521336 0 0)
  triclinic box = (0 0 0) to (9.8003491 8.3209327 15.595634) with tilt (-4.8521336 0 0)
  triclinic box = (0 0 0) to (9.8003491 8.3209327 15.595634) with tilt (-4.9001746 0 0)
  triclinic box = (0 0 0) to (9.8003491 8.3209327 15.595634) with tilt (-4.9001746 0 0)
  triclinic box = (0 0 0) to (9.8003491 8.3209327 15.595634) with tilt (-4.9001746 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.605934     -11025.973      0              0            
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
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608185420397755 y final 0 8.320932658335721 z final 0 15.283721001057621      xy final -4.804092710198877 xz final 0.0 yz final 0.0 units box
Changing box ...
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.9001746 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.9001746 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.283721) with tilt (-4.9001746 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.283721) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.283721) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.283721) with tilt (-4.8040927 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.566581      4026.5786      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3942 ave        3942 max        3942 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3942
Ave neighs/atom = 73
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608185420397755 y final 0 8.320932658335721 z final 0 15.439677337803108      xy final -4.804092710198877 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.283721) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.283721) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.439677) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.439677) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.439677) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.439677) with tilt (-4.8040927 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.662638      1678.4004      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3942 ave        3942 max        3942 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3942
Ave neighs/atom = 73
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608185420397755 y final 0 8.320932658335721 z final 0 15.595633674548594      xy final -4.804092710198877 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.439677) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.439677) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.698423     -435.03273      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608185420397755 y final 0 8.320932658335721 z final 0 15.75159001129408      xy final -4.804092710198877 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.75159) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.75159) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.75159) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.75159) with tilt (-4.8040927 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.680883     -2387.6124      0              0            
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
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608185420397755 y final 0 8.320932658335721 z final 0 15.907546348039567      xy final -4.804092710198877 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.75159) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.75159) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.907546) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.907546) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.907546) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.907546) with tilt (-4.8040927 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.613311     -4155.51        0              0            
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
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608185420397755 y final 0 8.320949300184395 z final 0 15.595602483374819      xy final -4.804092710198877 xz final 0.0 yz final 0.031191204966749635 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.907546) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209493 15.907546) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209493 15.595602) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209493 15.595602) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209493 15.595602) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209493 15.595602) with tilt (-4.8040927 0 0.031191205)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.698201     -434.97389      0              0            
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
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608185420397755 y final 0 8.321108436181495 z final 0 15.595304227226546      xy final -4.804092710198877 xz final 0.0 yz final 0.10136947747697254 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6081854 8.3209493 15.595602) with tilt (-4.8040927 0 0.031191205)
  triclinic box = (0 0 0) to (9.6081854 8.3211084 15.595602) with tilt (-4.8040927 0 0.031191205)
  triclinic box = (0 0 0) to (9.6081854 8.3211084 15.595304) with tilt (-4.8040927 0 0.031191205)
  triclinic box = (0 0 0) to (9.6081854 8.3211084 15.595304) with tilt (-4.8040927 0 0.031191205)
  triclinic box = (0 0 0) to (9.6081854 8.3211084 15.595304) with tilt (-4.8040927 0 0.031191205)
  triclinic box = (0 0 0) to (9.6081854 8.3211084 15.595304) with tilt (-4.8040927 0 0.10136948)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.696503     -440.64307      0              0            
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
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608185420397755 y final 0 8.321436059534125 z final 0 15.594690224328529      xy final -4.804092710198877 xz final 0.0 yz final 0.17154159246761383 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6081854 8.3211084 15.595304) with tilt (-4.8040927 0 0.10136948)
  triclinic box = (0 0 0) to (9.6081854 8.3214361 15.595304) with tilt (-4.8040927 0 0.10136948)
  triclinic box = (0 0 0) to (9.6081854 8.3214361 15.59469) with tilt (-4.8040927 0 0.10136948)
  triclinic box = (0 0 0) to (9.6081854 8.3214361 15.59469) with tilt (-4.8040927 0 0.10136948)
  triclinic box = (0 0 0) to (9.6081854 8.3214361 15.59469) with tilt (-4.8040927 0 0.10136948)
  triclinic box = (0 0 0) to (9.6081854 8.3214361 15.59469) with tilt (-4.8040927 0 0.17154159)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.692162     -439.8342       0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608185420397755 y final 0 8.32193215034292 z final 0 15.593760586553726      xy final -4.804092710198877 xz final 0.0 yz final 0.24170328909158276 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6081854 8.3214361 15.59469) with tilt (-4.8040927 0 0.17154159)
  triclinic box = (0 0 0) to (9.6081854 8.3219322 15.59469) with tilt (-4.8040927 0 0.17154159)
  triclinic box = (0 0 0) to (9.6081854 8.3219322 15.593761) with tilt (-4.8040927 0 0.17154159)
  triclinic box = (0 0 0) to (9.6081854 8.3219322 15.593761) with tilt (-4.8040927 0 0.17154159)
  triclinic box = (0 0 0) to (9.6081854 8.3219322 15.593761) with tilt (-4.8040927 0 0.17154159)
  triclinic box = (0 0 0) to (9.6081854 8.3219322 15.593761) with tilt (-4.8040927 0 0.24170329)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.685781     -448.51607      0              0            
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
Nghost:           1113 ave        1113 max        1113 min
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
change_box all     x final 0 9.608185420397755 y final 0 8.322596678482011 z final 0 15.592515483239904      xy final -4.804092710198877 xz final 0.0 yz final 0.31185030966479804 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6081854 8.3219322 15.593761) with tilt (-4.8040927 0 0.24170329)
  triclinic box = (0 0 0) to (9.6081854 8.3225967 15.593761) with tilt (-4.8040927 0 0.24170329)
  triclinic box = (0 0 0) to (9.6081854 8.3225967 15.592515) with tilt (-4.8040927 0 0.24170329)
  triclinic box = (0 0 0) to (9.6081854 8.3225967 15.592515) with tilt (-4.8040927 0 0.24170329)
  triclinic box = (0 0 0) to (9.6081854 8.3225967 15.592515) with tilt (-4.8040927 0 0.24170329)
  triclinic box = (0 0 0) to (9.6081854 8.3225967 15.592515) with tilt (-4.8040927 0 0.31185031)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.676945     -446.70874      0              0            
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
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3726 ave        3726 max        3726 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3726
Ave neighs/atom = 69
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.60820463674938 y final 0 8.32091601652033 z final 0 15.595633674548594      xy final -4.787460486341648 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6082046 8.3225967 15.592515) with tilt (-4.8040927 0 0.31185031)
  triclinic box = (0 0 0) to (9.6082046 8.320916 15.592515) with tilt (-4.8040927 0 0.31185031)
  triclinic box = (0 0 0) to (9.6082046 8.320916 15.595634) with tilt (-4.8040927 0 0.31185031)
  triclinic box = (0 0 0) to (9.6082046 8.320916 15.595634) with tilt (-4.7874605 0 0.31185031)
  triclinic box = (0 0 0) to (9.6082046 8.320916 15.595634) with tilt (-4.7874605 0 0.31185031)
  triclinic box = (0 0 0) to (9.6082046 8.320916 15.595634) with tilt (-4.7874605 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.698146     -435.94784      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608388391170905 y final 0 8.320756884203137 z final 0 15.595633674548594      xy final -4.750109275838132 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6083884 8.320916 15.595634) with tilt (-4.7874605 0 0)
  triclinic box = (0 0 0) to (9.6083884 8.3207569 15.595634) with tilt (-4.7874605 0 0)
  triclinic box = (0 0 0) to (9.6083884 8.3207569 15.595634) with tilt (-4.7874605 0 0)
  triclinic box = (0 0 0) to (9.6083884 8.3207569 15.595634) with tilt (-4.7501093 0 0)
  triclinic box = (0 0 0) to (9.6083884 8.3207569 15.595634) with tilt (-4.7501093 0 0)
  triclinic box = (0 0 0) to (9.6083884 8.3207569 15.595634) with tilt (-4.7501093 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.696125     -461.29594      0              0            
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
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608766698032571 y final 0 8.320429287590326 z final 0 15.595633674548594      xy final -4.712858626852792 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6087667 8.3207569 15.595634) with tilt (-4.7501093 0 0)
  triclinic box = (0 0 0) to (9.6087667 8.3204293 15.595634) with tilt (-4.7501093 0 0)
  triclinic box = (0 0 0) to (9.6087667 8.3204293 15.595634) with tilt (-4.7501093 0 0)
  triclinic box = (0 0 0) to (9.6087667 8.3204293 15.595634) with tilt (-4.7128586 0 0)
  triclinic box = (0 0 0) to (9.6087667 8.3204293 15.595634) with tilt (-4.7128586 0 0)
  triclinic box = (0 0 0) to (9.6087667 8.3204293 15.595634) with tilt (-4.7128586 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.690699     -480.39681      0              0            
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
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.609339534356574 y final 0 8.319933286370869 z final 0 15.595633674548594      xy final -4.675710801239537 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6093395 8.3204293 15.595634) with tilt (-4.7128586 0 0)
  triclinic box = (0 0 0) to (9.6093395 8.3199333 15.595634) with tilt (-4.7128586 0 0)
  triclinic box = (0 0 0) to (9.6093395 8.3199333 15.595634) with tilt (-4.7128586 0 0)
  triclinic box = (0 0 0) to (9.6093395 8.3199333 15.595634) with tilt (-4.6757108 0 0)
  triclinic box = (0 0 0) to (9.6093395 8.3199333 15.595634) with tilt (-4.6757108 0 0)
  triclinic box = (0 0 0) to (9.6093395 8.3199333 15.595634) with tilt (-4.6757108 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.682457     -508.79816      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.61010686535655 y final 0 8.319268970893654 z final 0 15.595633674548594      xy final -4.638668053260401 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6101069 8.3199333 15.595634) with tilt (-4.6757108 0 0)
  triclinic box = (0 0 0) to (9.6101069 8.319269 15.595634) with tilt (-4.6757108 0 0)
  triclinic box = (0 0 0) to (9.6101069 8.319269 15.595634) with tilt (-4.6757108 0 0)
  triclinic box = (0 0 0) to (9.6101069 8.319269 15.595634) with tilt (-4.6386681 0 0)
  triclinic box = (0 0 0) to (9.6101069 8.319269 15.595634) with tilt (-4.6386681 0 0)
  triclinic box = (0 0 0) to (9.6101069 8.319269 15.595634) with tilt (-4.6386681 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-83.671392     -546.54258      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 54 atoms

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

Nlocal:             54 ave          54 max          54 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1113 ave        1113 max        1113 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           3618 ave        3618 max        3618 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 3618
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  54 settings made for x
  54 settings made for y
  54 settings made for z
change_box all     x final 0 9.608185420397755 y final 0 8.320932658335721 z final 0 10.397089116365729      xy final -4.804092710198877 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6081854 8.319269 15.595634) with tilt (-4.6386681 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 15.595634) with tilt (-4.6386681 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 10.397089) with tilt (-4.6386681 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 10.397089) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 10.397089) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (9.6081854 8.3209327 10.397089) with tilt (-4.8040927 0 0)
echo none
18 atoms in group delatoms
Deleted 18 atoms, new total = 36
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-55.798814     -434.20439      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 36 atoms

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

Nlocal:             36 ave          36 max          36 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            950 ave         950 max         950 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           2412 ave        2412 max        2412 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 2412
Ave neighs/atom = 67
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  36 settings made for x
  36 settings made for y
  36 settings made for z
change_box all     x final 0 12.810913893863672 y final 0 13.8682210972262 z final 0 31.19126734909719      xy final -8.006821183664796 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.810914 8.3209327 10.397089) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 10.397089) with tilt (-4.8040927 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-4.8040927 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
echo none
Created 203 atoms
  using lattice units in triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
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
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.256 | 3.256 | 3.256 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.78322     -582.73034      0              0            
Loop time of 1.334e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.556266154896742 y final 0 13.592556833524164 z final 0 30.571265858823715      xy final -7.847666346810464 xz final 0.0 yz final 0.0 units box
Changing box ...
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.556266 13.868221 31.191267) with tilt (-8.0068212 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.556266 13.592557 31.191267) with tilt (-8.0068212 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.556266 13.592557 30.571266) with tilt (-8.0068212 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.556266 13.592557 30.571266) with tilt (-7.8476663 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.556266 13.592557 30.571266) with tilt (-7.8476663 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.556266 13.592557 30.571266) with tilt (-7.8476663 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.49065      23076.799      0              0            
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
Nghost:           2261 ave        2261 max        2261 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          17374 ave       17374 max       17374 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 17374
Ave neighs/atom = 72.694561
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809967497393766 y final 0 13.867196592994961 z final 0 31.18896311806764      xy final -8.006229685871103 xz final 0.0 yz final 0.0 units box
Changing box ...
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809967 13.592557 30.571266) with tilt (-7.8476663 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809967 13.867197 30.571266) with tilt (-7.8476663 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809967 13.867197 31.188963) with tilt (-7.8476663 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809967 13.867197 31.188963) with tilt (-8.0062297 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809967 13.867197 31.188963) with tilt (-8.0062297 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809967 13.867197 31.188963) with tilt (-8.0062297 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.78325     -506.35329      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809967497393766 y final 0 13.867196592994961 z final 0 31.18896311806764      xy final -8.006229685871103 xz final 0.0 yz final 0.0 units box
Changing box ...
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809967 13.867197 31.188963) with tilt (-8.0062297 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809967 13.867197 31.188963) with tilt (-8.0062297 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809967 13.867197 31.188963) with tilt (-8.0062297 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809967 13.867197 31.188963) with tilt (-8.0062297 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809967 13.867197 31.188963) with tilt (-8.0062297 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809967 13.867197 31.188963) with tilt (-8.0062297 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.78325     -506.35329      0              0            
Loop time of 2.9417e-05 on 1 procs for 0 steps with 239 atoms

30.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 2.942e-05  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.80974365281584 y final 0 13.866954274131233 z final 0 31.188418114320637      xy final -8.006089783009902 xz final 0.0 yz final 0.0 units box
Changing box ...
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809744 13.867197 31.188963) with tilt (-8.0062297 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809744 13.866954 31.188963) with tilt (-8.0062297 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809744 13.866954 31.188418) with tilt (-8.0062297 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809744 13.866954 31.188418) with tilt (-8.0060898 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809744 13.866954 31.188418) with tilt (-8.0060898 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809744 13.866954 31.188418) with tilt (-8.0060898 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.78822     -510.93213      0              0            
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
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809368000087467 y final 0 13.866547618124017 z final 0 31.187503496926578      xy final -8.005855000054668 xz final 0.0 yz final 0.0 units box
Changing box ...
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809368 13.866954 31.188418) with tilt (-8.0060898 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809368 13.866548 31.188418) with tilt (-8.0060898 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809368 13.866548 31.187503) with tilt (-8.0060898 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809368 13.866548 31.187503) with tilt (-8.005855 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809368 13.866548 31.187503) with tilt (-8.005855 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809368 13.866548 31.187503) with tilt (-8.005855 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79159     -513.46098      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809368000087467 y final 0 13.866547618124015 z final 0 31.187503496926578      xy final -8.00585500005467 xz final 0.0 yz final 0.0 units box
Changing box ...
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809368 13.866548 31.187503) with tilt (-8.005855 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809368 13.866548 31.187503) with tilt (-8.005855 0 0)
WARNING: Triclinic box skew is large (src/src/domain.cpp:224)
  triclinic box = (0 0 0) to (12.809368 13.866548 31.187503) with tilt (-8.005855 0 0)
  triclinic box = (0 0 0) to (12.809368 13.866548 31.187503) with tilt (-8.005855 0 0)
  triclinic box = (0 0 0) to (12.809368 13.866548 31.187503) with tilt (-8.005855 0 0)
  triclinic box = (0 0 0) to (12.809368 13.866548 31.187503) with tilt (-8.005855 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
WARNING: Too many warnings: 104 vs 100. All future warnings will be suppressed (src/src/thermo.cpp:465)
-369.79159     -513.46098      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809474249267009 y final 0 13.866662636234784 z final 0 31.187762186243525      xy final -8.005921405791883 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809474 13.866548 31.187503) with tilt (-8.005855 0 0)
  triclinic box = (0 0 0) to (12.809474 13.866663 31.187503) with tilt (-8.005855 0 0)
  triclinic box = (0 0 0) to (12.809474 13.866663 31.187762) with tilt (-8.005855 0 0)
  triclinic box = (0 0 0) to (12.809474 13.866663 31.187762) with tilt (-8.0059214 0 0)
  triclinic box = (0 0 0) to (12.809474 13.866663 31.187762) with tilt (-8.0059214 0 0)
  triclinic box = (0 0 0) to (12.809474 13.866663 31.187762) with tilt (-8.0059214 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79227     -524.13475      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809747158211493 y final 0 13.86695806883334 z final 0 31.188426649053902      xy final -8.006091973882185 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809747 13.866663 31.187762) with tilt (-8.0059214 0 0)
  triclinic box = (0 0 0) to (12.809747 13.866958 31.187762) with tilt (-8.0059214 0 0)
  triclinic box = (0 0 0) to (12.809747 13.866958 31.188427) with tilt (-8.0059214 0 0)
  triclinic box = (0 0 0) to (12.809747 13.866958 31.188427) with tilt (-8.006092 0 0)
  triclinic box = (0 0 0) to (12.809747 13.866958 31.188427) with tilt (-8.006092 0 0)
  triclinic box = (0 0 0) to (12.809747 13.866958 31.188427) with tilt (-8.006092 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79301     -550.12253      0              0            
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
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809747158211493 y final 0 13.86695806883334 z final 0 31.188426649053902      xy final -8.006091973882185 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809747 13.866958 31.188427) with tilt (-8.006092 0 0)
  triclinic box = (0 0 0) to (12.809747 13.866958 31.188427) with tilt (-8.006092 0 0)
  triclinic box = (0 0 0) to (12.809747 13.866958 31.188427) with tilt (-8.006092 0 0)
  triclinic box = (0 0 0) to (12.809747 13.866958 31.188427) with tilt (-8.006092 0 0)
  triclinic box = (0 0 0) to (12.809747 13.866958 31.188427) with tilt (-8.006092 0 0)
  triclinic box = (0 0 0) to (12.809747 13.866958 31.188427) with tilt (-8.006092 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79301     -550.12253      0              0            
Loop time of 1.083e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809421560509909 y final 0 13.866605598982108 z final 0 31.187633902725068      xy final -8.005888475318695 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809422 13.866958 31.188427) with tilt (-8.006092 0 0)
  triclinic box = (0 0 0) to (12.809422 13.866606 31.188427) with tilt (-8.006092 0 0)
  triclinic box = (0 0 0) to (12.809422 13.866606 31.187634) with tilt (-8.006092 0 0)
  triclinic box = (0 0 0) to (12.809422 13.866606 31.187634) with tilt (-8.0058885 0 0)
  triclinic box = (0 0 0) to (12.809422 13.866606 31.187634) with tilt (-8.0058885 0 0)
  triclinic box = (0 0 0) to (12.809422 13.866606 31.187634) with tilt (-8.0058885 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79314     -524.46744      0              0            
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
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.808700910015967 y final 0 13.86582547193835 z final 0 31.185879304855778      xy final -8.00543806875998 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808701 13.866606 31.187634) with tilt (-8.0058885 0 0)
  triclinic box = (0 0 0) to (12.808701 13.865825 31.187634) with tilt (-8.0058885 0 0)
  triclinic box = (0 0 0) to (12.808701 13.865825 31.185879) with tilt (-8.0058885 0 0)
  triclinic box = (0 0 0) to (12.808701 13.865825 31.185879) with tilt (-8.0054381 0 0)
  triclinic box = (0 0 0) to (12.808701 13.865825 31.185879) with tilt (-8.0054381 0 0)
  triclinic box = (0 0 0) to (12.808701 13.865825 31.185879) with tilt (-8.0054381 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79326     -467.44481      0              0            
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
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.808700910015967 y final 0 13.86582547193835 z final 0 31.185879304855774      xy final -8.00543806875998 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808701 13.865825 31.185879) with tilt (-8.0054381 0 0)
  triclinic box = (0 0 0) to (12.808701 13.865825 31.185879) with tilt (-8.0054381 0 0)
  triclinic box = (0 0 0) to (12.808701 13.865825 31.185879) with tilt (-8.0054381 0 0)
  triclinic box = (0 0 0) to (12.808701 13.865825 31.185879) with tilt (-8.0054381 0 0)
  triclinic box = (0 0 0) to (12.808701 13.865825 31.185879) with tilt (-8.0054381 0 0)
  triclinic box = (0 0 0) to (12.808701 13.865825 31.185879) with tilt (-8.0054381 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79326     -467.44481      0              0            
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
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.80921514209263 y final 0 13.86638214449064 z final 0 31.187131327179518      xy final -8.005759463807896 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809215 13.865825 31.185879) with tilt (-8.0054381 0 0)
  triclinic box = (0 0 0) to (12.809215 13.866382 31.185879) with tilt (-8.0054381 0 0)
  triclinic box = (0 0 0) to (12.809215 13.866382 31.187131) with tilt (-8.0054381 0 0)
  triclinic box = (0 0 0) to (12.809215 13.866382 31.187131) with tilt (-8.0057595 0 0)
  triclinic box = (0 0 0) to (12.809215 13.866382 31.187131) with tilt (-8.0057595 0 0)
  triclinic box = (0 0 0) to (12.809215 13.866382 31.187131) with tilt (-8.0057595 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.7933      -507.71846      0              0            
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
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.811272276856787 y final 0 13.8686090581966 z final 0 31.19213991914521      xy final -8.007045173035495 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.811272 13.866382 31.187131) with tilt (-8.0057595 0 0)
  triclinic box = (0 0 0) to (12.811272 13.868609 31.187131) with tilt (-8.0057595 0 0)
  triclinic box = (0 0 0) to (12.811272 13.868609 31.19214) with tilt (-8.0057595 0 0)
  triclinic box = (0 0 0) to (12.811272 13.868609 31.19214) with tilt (-8.0070452 0 0)
  triclinic box = (0 0 0) to (12.811272 13.868609 31.19214) with tilt (-8.0070452 0 0)
  triclinic box = (0 0 0) to (12.811272 13.868609 31.19214) with tilt (-8.0070452 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79311     -668.3816       0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809624912452643 y final 0 13.8668257339175 z final 0 31.1881290121969      xy final -8.006015570282905 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809625 13.868609 31.19214) with tilt (-8.0070452 0 0)
  triclinic box = (0 0 0) to (12.809625 13.866826 31.19214) with tilt (-8.0070452 0 0)
  triclinic box = (0 0 0) to (12.809625 13.866826 31.188129) with tilt (-8.0070452 0 0)
  triclinic box = (0 0 0) to (12.809625 13.866826 31.188129) with tilt (-8.0060156 0 0)
  triclinic box = (0 0 0) to (12.809625 13.866826 31.188129) with tilt (-8.0060156 0 0)
  triclinic box = (0 0 0) to (12.809625 13.866826 31.188129) with tilt (-8.0060156 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79331     -539.81764      0              0            
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
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809624912452643 y final 0 13.8668257339175 z final 0 31.188129012196896      xy final -8.006015570282907 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809625 13.866826 31.188129) with tilt (-8.0060156 0 0)
  triclinic box = (0 0 0) to (12.809625 13.866826 31.188129) with tilt (-8.0060156 0 0)
  triclinic box = (0 0 0) to (12.809625 13.866826 31.188129) with tilt (-8.0060156 0 0)
  triclinic box = (0 0 0) to (12.809625 13.866826 31.188129) with tilt (-8.0060156 0 0)
  triclinic box = (0 0 0) to (12.809625 13.866826 31.188129) with tilt (-8.0060156 0 0)
  triclinic box = (0 0 0) to (12.809625 13.866826 31.188129) with tilt (-8.0060156 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79331     -539.81764      0              0            
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
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809614390536524 y final 0 13.86681434360918 z final 0 31.188103394048074      xy final -8.006008994085331 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809614 13.866826 31.188129) with tilt (-8.0060156 0 0)
  triclinic box = (0 0 0) to (12.809614 13.866814 31.188129) with tilt (-8.0060156 0 0)
  triclinic box = (0 0 0) to (12.809614 13.866814 31.188103) with tilt (-8.0060156 0 0)
  triclinic box = (0 0 0) to (12.809614 13.866814 31.188103) with tilt (-8.006009 0 0)
  triclinic box = (0 0 0) to (12.809614 13.866814 31.188103) with tilt (-8.006009 0 0)
  triclinic box = (0 0 0) to (12.809614 13.866814 31.188103) with tilt (-8.006009 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79334     -538.85069      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.80958579502127 y final 0 13.866783388055874 z final 0 31.188033771352245      xy final -8.0059911218883 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809586 13.866814 31.188103) with tilt (-8.006009 0 0)
  triclinic box = (0 0 0) to (12.809586 13.866783 31.188103) with tilt (-8.006009 0 0)
  triclinic box = (0 0 0) to (12.809586 13.866783 31.188034) with tilt (-8.006009 0 0)
  triclinic box = (0 0 0) to (12.809586 13.866783 31.188034) with tilt (-8.0059911 0 0)
  triclinic box = (0 0 0) to (12.809586 13.866783 31.188034) with tilt (-8.0059911 0 0)
  triclinic box = (0 0 0) to (12.809586 13.866783 31.188034) with tilt (-8.0059911 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79339     -536.17223      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.80958579502127 y final 0 13.866783388055874 z final 0 31.188033771352245      xy final -8.005991121888302 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809586 13.866783 31.188034) with tilt (-8.0059911 0 0)
  triclinic box = (0 0 0) to (12.809586 13.866783 31.188034) with tilt (-8.0059911 0 0)
  triclinic box = (0 0 0) to (12.809586 13.866783 31.188034) with tilt (-8.0059911 0 0)
  triclinic box = (0 0 0) to (12.809586 13.866783 31.188034) with tilt (-8.0059911 0 0)
  triclinic box = (0 0 0) to (12.809586 13.866783 31.188034) with tilt (-8.0059911 0 0)
  triclinic box = (0 0 0) to (12.809586 13.866783 31.188034) with tilt (-8.0059911 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79339     -536.17223      0              0            
Loop time of 1.541e-06 on 1 procs for 0 steps with 239 atoms

129.8% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.541e-06  |            |       |100.00

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809389542716506 y final 0 13.866570938704028 z final 0 31.187555947665622      xy final -8.005868464197825 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.80939 13.866783 31.188034) with tilt (-8.0059911 0 0)
  triclinic box = (0 0 0) to (12.80939 13.866571 31.188034) with tilt (-8.0059911 0 0)
  triclinic box = (0 0 0) to (12.80939 13.866571 31.187556) with tilt (-8.0059911 0 0)
  triclinic box = (0 0 0) to (12.80939 13.866571 31.187556) with tilt (-8.0058685 0 0)
  triclinic box = (0 0 0) to (12.80939 13.866571 31.187556) with tilt (-8.0058685 0 0)
  triclinic box = (0 0 0) to (12.80939 13.866571 31.187556) with tilt (-8.0058685 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.7934      -520.45086      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.808992416252602 y final 0 13.866141036696208 z final 0 31.186589047270406      xy final -8.005620260157885 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808992 13.866571 31.187556) with tilt (-8.0058685 0 0)
  triclinic box = (0 0 0) to (12.808992 13.866141 31.187556) with tilt (-8.0058685 0 0)
  triclinic box = (0 0 0) to (12.808992 13.866141 31.186589) with tilt (-8.0058685 0 0)
  triclinic box = (0 0 0) to (12.808992 13.866141 31.186589) with tilt (-8.0056203 0 0)
  triclinic box = (0 0 0) to (12.808992 13.866141 31.186589) with tilt (-8.0056203 0 0)
  triclinic box = (0 0 0) to (12.808992 13.866141 31.186589) with tilt (-8.0056203 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79343     -488.66452      0              0            
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
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.808992416252602 y final 0 13.866141036696208 z final 0 31.186589047270402      xy final -8.005620260157887 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808992 13.866141 31.186589) with tilt (-8.0056203 0 0)
  triclinic box = (0 0 0) to (12.808992 13.866141 31.186589) with tilt (-8.0056203 0 0)
  triclinic box = (0 0 0) to (12.808992 13.866141 31.186589) with tilt (-8.0056203 0 0)
  triclinic box = (0 0 0) to (12.808992 13.866141 31.186589) with tilt (-8.0056203 0 0)
  triclinic box = (0 0 0) to (12.808992 13.866141 31.186589) with tilt (-8.0056203 0 0)
  triclinic box = (0 0 0) to (12.808992 13.866141 31.186589) with tilt (-8.0056203 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79343     -488.66452      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809173643432986 y final 0 13.866337220873794 z final 0 31.187030288658885      xy final -8.005733527145628 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809174 13.866141 31.186589) with tilt (-8.0056203 0 0)
  triclinic box = (0 0 0) to (12.809174 13.866337 31.186589) with tilt (-8.0056203 0 0)
  triclinic box = (0 0 0) to (12.809174 13.866337 31.18703) with tilt (-8.0056203 0 0)
  triclinic box = (0 0 0) to (12.809174 13.866337 31.18703) with tilt (-8.0057335 0 0)
  triclinic box = (0 0 0) to (12.809174 13.866337 31.18703) with tilt (-8.0057335 0 0)
  triclinic box = (0 0 0) to (12.809174 13.866337 31.18703) with tilt (-8.0057335 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79345     -503.05132      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809461471910508 y final 0 13.866648804340628 z final 0 31.18773107667974      xy final -8.005913419944081 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809461 13.866337 31.18703) with tilt (-8.0057335 0 0)
  triclinic box = (0 0 0) to (12.809461 13.866649 31.18703) with tilt (-8.0057335 0 0)
  triclinic box = (0 0 0) to (12.809461 13.866649 31.187731) with tilt (-8.0057335 0 0)
  triclinic box = (0 0 0) to (12.809461 13.866649 31.187731) with tilt (-8.0059134 0 0)
  triclinic box = (0 0 0) to (12.809461 13.866649 31.187731) with tilt (-8.0059134 0 0)
  triclinic box = (0 0 0) to (12.809461 13.866649 31.187731) with tilt (-8.0059134 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79346     -525.88357      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809461471910508 y final 0 13.866648804340628 z final 0 31.187731076679736      xy final -8.005913419944083 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809461 13.866649 31.187731) with tilt (-8.0059134 0 0)
  triclinic box = (0 0 0) to (12.809461 13.866649 31.187731) with tilt (-8.0059134 0 0)
  triclinic box = (0 0 0) to (12.809461 13.866649 31.187731) with tilt (-8.0059134 0 0)
  triclinic box = (0 0 0) to (12.809461 13.866649 31.187731) with tilt (-8.0059134 0 0)
  triclinic box = (0 0 0) to (12.809461 13.866649 31.187731) with tilt (-8.0059134 0 0)
  triclinic box = (0 0 0) to (12.809461 13.866649 31.187731) with tilt (-8.0059134 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79346     -525.88357      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809435941533696 y final 0 13.86662116689702 z final 0 31.187668916804554      xy final -8.005897463458576 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809436 13.866649 31.187731) with tilt (-8.0059134 0 0)
  triclinic box = (0 0 0) to (12.809436 13.866621 31.187731) with tilt (-8.0059134 0 0)
  triclinic box = (0 0 0) to (12.809436 13.866621 31.187669) with tilt (-8.0059134 0 0)
  triclinic box = (0 0 0) to (12.809436 13.866621 31.187669) with tilt (-8.0058975 0 0)
  triclinic box = (0 0 0) to (12.809436 13.866621 31.187669) with tilt (-8.0058975 0 0)
  triclinic box = (0 0 0) to (12.809436 13.866621 31.187669) with tilt (-8.0058975 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79346     -523.89679      0              0            
Loop time of 1e-06 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.809362905135268 y final 0 13.86654210267647 z final 0 31.187491092033543      xy final -8.00585181570956 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809363 13.866621 31.187669) with tilt (-8.0058975 0 0)
  triclinic box = (0 0 0) to (12.809363 13.866542 31.187669) with tilt (-8.0058975 0 0)
  triclinic box = (0 0 0) to (12.809363 13.866542 31.187491) with tilt (-8.0058975 0 0)
  triclinic box = (0 0 0) to (12.809363 13.866542 31.187491) with tilt (-8.0058518 0 0)
  triclinic box = (0 0 0) to (12.809363 13.866542 31.187491) with tilt (-8.0058518 0 0)
  triclinic box = (0 0 0) to (12.809363 13.866542 31.187491) with tilt (-8.0058518 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-369.79348     -518.25042      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 239 atoms

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

Nlocal:            239 ave         239 max         239 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2041 ave        2041 max        2041 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15946 ave       15946 max       15946 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15946
Ave neighs/atom = 66.719665
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  239 settings made for x
  239 settings made for y
  239 settings made for z
change_box all     x final 0 12.810913893863672 y final 0 13.8682210972262 z final 0 31.19126734909719      xy final -8.006821183664796 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.810914 13.866542 31.187491) with tilt (-8.0058518 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.187491) with tilt (-8.0058518 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0058518 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
echo none
1 atoms in group delatoms
Deleted 1 atoms, new total = 238
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.65402     -715.38861      0              0            
Loop time of 1.5e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.047020404164604 y final 0 13.041282137395026 z final 0 29.3313839511726      xy final -7.529387752602878 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.04702 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.04702 13.041282 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.04702 13.041282 29.331384) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.04702 13.041282 29.331384) with tilt (-7.5293878 0 0)
  triclinic box = (0 0 0) to (12.04702 13.041282 29.331384) with tilt (-7.5293878 0 0)
  triclinic box = (0 0 0) to (12.04702 13.041282 29.331384) with tilt (-7.5293878 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-343.06972      101424.83      0              0            
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
Nghost:           2254 ave        2254 max        2254 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          18645 ave       18645 max       18645 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 18645
Ave neighs/atom = 78.340336
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808706294438656 y final 0 13.865831300746896 z final 0 31.185892414534685      xy final -8.00544143402416 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808706 13.041282 29.331384) with tilt (-7.5293878 0 0)
  triclinic box = (0 0 0) to (12.808706 13.865831 29.331384) with tilt (-7.5293878 0 0)
  triclinic box = (0 0 0) to (12.808706 13.865831 31.185892) with tilt (-7.5293878 0 0)
  triclinic box = (0 0 0) to (12.808706 13.865831 31.185892) with tilt (-8.0054414 0 0)
  triclinic box = (0 0 0) to (12.808706 13.865831 31.185892) with tilt (-8.0054414 0 0)
  triclinic box = (0 0 0) to (12.808706 13.865831 31.185892) with tilt (-8.0054414 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.65419     -538.34773      0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808706294438656 y final 0 13.865831300746896 z final 0 31.185892414534685      xy final -8.00544143402416 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808706 13.865831 31.185892) with tilt (-8.0054414 0 0)
  triclinic box = (0 0 0) to (12.808706 13.865831 31.185892) with tilt (-8.0054414 0 0)
  triclinic box = (0 0 0) to (12.808706 13.865831 31.185892) with tilt (-8.0054414 0 0)
  triclinic box = (0 0 0) to (12.808706 13.865831 31.185892) with tilt (-8.0054414 0 0)
  triclinic box = (0 0 0) to (12.808706 13.865831 31.185892) with tilt (-8.0054414 0 0)
  triclinic box = (0 0 0) to (12.808706 13.865831 31.185892) with tilt (-8.0054414 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.65419     -538.34773      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808195850809778 y final 0 13.865278729309635 z final 0 31.18464961610306      xy final -8.005122406756112 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808196 13.865831 31.185892) with tilt (-8.0054414 0 0)
  triclinic box = (0 0 0) to (12.808196 13.865279 31.185892) with tilt (-8.0054414 0 0)
  triclinic box = (0 0 0) to (12.808196 13.865279 31.18465) with tilt (-8.0054414 0 0)
  triclinic box = (0 0 0) to (12.808196 13.865279 31.18465) with tilt (-8.0051224 0 0)
  triclinic box = (0 0 0) to (12.808196 13.865279 31.18465) with tilt (-8.0051224 0 0)
  triclinic box = (0 0 0) to (12.808196 13.865279 31.18465) with tilt (-8.0051224 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.66111     -528.73431      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806728082460038 y final 0 13.863689823462455 z final 0 31.18107597917266      xy final -8.004205051537525 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806728 13.865279 31.18465) with tilt (-8.0051224 0 0)
  triclinic box = (0 0 0) to (12.806728 13.86369 31.18465) with tilt (-8.0051224 0 0)
  triclinic box = (0 0 0) to (12.806728 13.86369 31.181076) with tilt (-8.0051224 0 0)
  triclinic box = (0 0 0) to (12.806728 13.86369 31.181076) with tilt (-8.0042051 0 0)
  triclinic box = (0 0 0) to (12.806728 13.86369 31.181076) with tilt (-8.0042051 0 0)
  triclinic box = (0 0 0) to (12.806728 13.86369 31.181076) with tilt (-8.0042051 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.66992     -486.75991      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806728082460038 y final 0 13.863689823462455 z final 0 31.18107597917266      xy final -8.004205051537527 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806728 13.86369 31.181076) with tilt (-8.0042051 0 0)
  triclinic box = (0 0 0) to (12.806728 13.86369 31.181076) with tilt (-8.0042051 0 0)
  triclinic box = (0 0 0) to (12.806728 13.86369 31.181076) with tilt (-8.0042051 0 0)
  triclinic box = (0 0 0) to (12.806728 13.86369 31.181076) with tilt (-8.0042051 0 0)
  triclinic box = (0 0 0) to (12.806728 13.86369 31.181076) with tilt (-8.0042051 0 0)
  triclinic box = (0 0 0) to (12.806728 13.86369 31.181076) with tilt (-8.0042051 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.66992     -486.75991      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.807073818636999 y final 0 13.864064093852773 z final 0 31.18191775748896      xy final -8.004421136648126 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.807074 13.86369 31.181076) with tilt (-8.0042051 0 0)
  triclinic box = (0 0 0) to (12.807074 13.864064 31.181076) with tilt (-8.0042051 0 0)
  triclinic box = (0 0 0) to (12.807074 13.864064 31.181918) with tilt (-8.0042051 0 0)
  triclinic box = (0 0 0) to (12.807074 13.864064 31.181918) with tilt (-8.0044211 0 0)
  triclinic box = (0 0 0) to (12.807074 13.864064 31.181918) with tilt (-8.0044211 0 0)
  triclinic box = (0 0 0) to (12.807074 13.864064 31.181918) with tilt (-8.0044211 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67179     -517.09478      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808456856683847 y final 0 13.865561276456486 z final 0 31.185285098010535      xy final -8.005285535427406 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808457 13.864064 31.181918) with tilt (-8.0044211 0 0)
  triclinic box = (0 0 0) to (12.808457 13.865561 31.181918) with tilt (-8.0044211 0 0)
  triclinic box = (0 0 0) to (12.808457 13.865561 31.185285) with tilt (-8.0044211 0 0)
  triclinic box = (0 0 0) to (12.808457 13.865561 31.185285) with tilt (-8.0052855 0 0)
  triclinic box = (0 0 0) to (12.808457 13.865561 31.185285) with tilt (-8.0052855 0 0)
  triclinic box = (0 0 0) to (12.808457 13.865561 31.185285) with tilt (-8.0052855 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67409     -633.25815      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15819 ave       15819 max       15819 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15819
Ave neighs/atom = 66.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808456856683847 y final 0 13.865561276456486 z final 0 31.185285098010535      xy final -8.005285535427406 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808457 13.865561 31.185285) with tilt (-8.0052855 0 0)
  triclinic box = (0 0 0) to (12.808457 13.865561 31.185285) with tilt (-8.0052855 0 0)
  triclinic box = (0 0 0) to (12.808457 13.865561 31.185285) with tilt (-8.0052855 0 0)
  triclinic box = (0 0 0) to (12.808457 13.865561 31.185285) with tilt (-8.0052855 0 0)
  triclinic box = (0 0 0) to (12.808457 13.865561 31.185285) with tilt (-8.0052855 0 0)
  triclinic box = (0 0 0) to (12.808457 13.865561 31.185285) with tilt (-8.0052855 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67409     -633.25815      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15819 ave       15819 max       15819 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15819
Ave neighs/atom = 66.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806895154762906 y final 0 13.863870684535646 z final 0 31.18148275708941      xy final -8.004309471726819 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806895 13.865561 31.185285) with tilt (-8.0052855 0 0)
  triclinic box = (0 0 0) to (12.806895 13.863871 31.185285) with tilt (-8.0052855 0 0)
  triclinic box = (0 0 0) to (12.806895 13.863871 31.181483) with tilt (-8.0052855 0 0)
  triclinic box = (0 0 0) to (12.806895 13.863871 31.181483) with tilt (-8.0043095 0 0)
  triclinic box = (0 0 0) to (12.806895 13.863871 31.181483) with tilt (-8.0043095 0 0)
  triclinic box = (0 0 0) to (12.806895 13.863871 31.181483) with tilt (-8.0043095 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67485     -505.44145      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15819 ave       15819 max       15819 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15819
Ave neighs/atom = 66.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806895154762906 y final 0 13.863870684535646 z final 0 31.18148275708941      xy final -8.004309471726819 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806895 13.863871 31.181483) with tilt (-8.0043095 0 0)
  triclinic box = (0 0 0) to (12.806895 13.863871 31.181483) with tilt (-8.0043095 0 0)
  triclinic box = (0 0 0) to (12.806895 13.863871 31.181483) with tilt (-8.0043095 0 0)
  triclinic box = (0 0 0) to (12.806895 13.863871 31.181483) with tilt (-8.0043095 0 0)
  triclinic box = (0 0 0) to (12.806895 13.863871 31.181483) with tilt (-8.0043095 0 0)
  triclinic box = (0 0 0) to (12.806895 13.863871 31.181483) with tilt (-8.0043095 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67485     -505.44145      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15819 ave       15819 max       15819 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15819
Ave neighs/atom = 66.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806219280134806 y final 0 13.863139028788513 z final 0 31.179837176891606      xy final -8.003887050084256 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806219 13.863871 31.181483) with tilt (-8.0043095 0 0)
  triclinic box = (0 0 0) to (12.806219 13.863139 31.181483) with tilt (-8.0043095 0 0)
  triclinic box = (0 0 0) to (12.806219 13.863139 31.179837) with tilt (-8.0043095 0 0)
  triclinic box = (0 0 0) to (12.806219 13.863139 31.179837) with tilt (-8.0038871 0 0)
  triclinic box = (0 0 0) to (12.806219 13.863139 31.179837) with tilt (-8.0038871 0 0)
  triclinic box = (0 0 0) to (12.806219 13.863139 31.179837) with tilt (-8.0038871 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.6753      -445.82686      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15819 ave       15819 max       15819 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15819
Ave neighs/atom = 66.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.805201982297932 y final 0 13.862037771576086 z final 0 31.177360319341368      xy final -8.00325123893621 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805202 13.863139 31.179837) with tilt (-8.0038871 0 0)
  triclinic box = (0 0 0) to (12.805202 13.862038 31.179837) with tilt (-8.0038871 0 0)
  triclinic box = (0 0 0) to (12.805202 13.862038 31.17736) with tilt (-8.0038871 0 0)
  triclinic box = (0 0 0) to (12.805202 13.862038 31.17736) with tilt (-8.0032512 0 0)
  triclinic box = (0 0 0) to (12.805202 13.862038 31.17736) with tilt (-8.0032512 0 0)
  triclinic box = (0 0 0) to (12.805202 13.862038 31.17736) with tilt (-8.0032512 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67544     -355.88298      0              0            
Loop time of 5.41e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15819 ave       15819 max       15819 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15819
Ave neighs/atom = 66.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.805201982297932 y final 0 13.862037771576086 z final 0 31.177360319341364      xy final -8.00325123893621 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805202 13.862038 31.17736) with tilt (-8.0032512 0 0)
  triclinic box = (0 0 0) to (12.805202 13.862038 31.17736) with tilt (-8.0032512 0 0)
  triclinic box = (0 0 0) to (12.805202 13.862038 31.17736) with tilt (-8.0032512 0 0)
  triclinic box = (0 0 0) to (12.805202 13.862038 31.17736) with tilt (-8.0032512 0 0)
  triclinic box = (0 0 0) to (12.805202 13.862038 31.17736) with tilt (-8.0032512 0 0)
  triclinic box = (0 0 0) to (12.805202 13.862038 31.17736) with tilt (-8.0032512 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67544     -355.88298      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15819 ave       15819 max       15819 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15819
Ave neighs/atom = 66.466387
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806083725040045 y final 0 13.862992286093924 z final 0 31.179507135238353      xy final -8.00380232815003 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806084 13.862038 31.17736) with tilt (-8.0032512 0 0)
  triclinic box = (0 0 0) to (12.806084 13.862992 31.17736) with tilt (-8.0032512 0 0)
  triclinic box = (0 0 0) to (12.806084 13.862992 31.179507) with tilt (-8.0032512 0 0)
  triclinic box = (0 0 0) to (12.806084 13.862992 31.179507) with tilt (-8.0038023 0 0)
  triclinic box = (0 0 0) to (12.806084 13.862992 31.179507) with tilt (-8.0038023 0 0)
  triclinic box = (0 0 0) to (12.806084 13.862992 31.179507) with tilt (-8.0038023 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67575     -426.58692      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15823 ave       15823 max       15823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15823
Ave neighs/atom = 66.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.809611303202193 y final 0 13.866811001471739 z final 0 31.18809587718619      xy final -8.006007064501373 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.809611 13.862992 31.179507) with tilt (-8.0038023 0 0)
  triclinic box = (0 0 0) to (12.809611 13.866811 31.179507) with tilt (-8.0038023 0 0)
  triclinic box = (0 0 0) to (12.809611 13.866811 31.188096) with tilt (-8.0038023 0 0)
  triclinic box = (0 0 0) to (12.809611 13.866811 31.188096) with tilt (-8.0060071 0 0)
  triclinic box = (0 0 0) to (12.809611 13.866811 31.188096) with tilt (-8.0060071 0 0)
  triclinic box = (0 0 0) to (12.809611 13.866811 31.188096) with tilt (-8.0060071 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67481     -706.40765      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15823 ave       15823 max       15823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15823
Ave neighs/atom = 66.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806877806669752 y final 0 13.863851904673929 z final 0 31.18144051896214      xy final -8.004298629168597 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806878 13.866811 31.188096) with tilt (-8.0060071 0 0)
  triclinic box = (0 0 0) to (12.806878 13.863852 31.188096) with tilt (-8.0060071 0 0)
  triclinic box = (0 0 0) to (12.806878 13.863852 31.181441) with tilt (-8.0060071 0 0)
  triclinic box = (0 0 0) to (12.806878 13.863852 31.181441) with tilt (-8.0042986 0 0)
  triclinic box = (0 0 0) to (12.806878 13.863852 31.181441) with tilt (-8.0042986 0 0)
  triclinic box = (0 0 0) to (12.806878 13.863852 31.181441) with tilt (-8.0042986 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67583     -489.93404      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15823 ave       15823 max       15823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15823
Ave neighs/atom = 66.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806877806669752 y final 0 13.863851904673929 z final 0 31.18144051896214      xy final -8.004298629168597 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806878 13.863852 31.181441) with tilt (-8.0042986 0 0)
  triclinic box = (0 0 0) to (12.806878 13.863852 31.181441) with tilt (-8.0042986 0 0)
  triclinic box = (0 0 0) to (12.806878 13.863852 31.181441) with tilt (-8.0042986 0 0)
  triclinic box = (0 0 0) to (12.806878 13.863852 31.181441) with tilt (-8.0042986 0 0)
  triclinic box = (0 0 0) to (12.806878 13.863852 31.181441) with tilt (-8.0042986 0 0)
  triclinic box = (0 0 0) to (12.806878 13.863852 31.181441) with tilt (-8.0042986 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67583     -489.93404      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 238 atoms

671.8% CPU use with 1 MPI tasks x no OpenMP threads

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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15823 ave       15823 max       15823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15823
Ave neighs/atom = 66.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806883914545756 y final 0 13.86385851664366 z final 0 31.181455390063064      xy final -8.0043024465911 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806884 13.863852 31.181441) with tilt (-8.0042986 0 0)
  triclinic box = (0 0 0) to (12.806884 13.863859 31.181441) with tilt (-8.0042986 0 0)
  triclinic box = (0 0 0) to (12.806884 13.863859 31.181455) with tilt (-8.0042986 0 0)
  triclinic box = (0 0 0) to (12.806884 13.863859 31.181455) with tilt (-8.0043024 0 0)
  triclinic box = (0 0 0) to (12.806884 13.863859 31.181455) with tilt (-8.0043024 0 0)
  triclinic box = (0 0 0) to (12.806884 13.863859 31.181455) with tilt (-8.0043024 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67599     -485.57564      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15823 ave       15823 max       15823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15823
Ave neighs/atom = 66.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806894769006355 y final 0 13.863870266941943 z final 0 31.181481817871834      xy final -8.004309230628975 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806895 13.863859 31.181455) with tilt (-8.0043024 0 0)
  triclinic box = (0 0 0) to (12.806895 13.86387 31.181455) with tilt (-8.0043024 0 0)
  triclinic box = (0 0 0) to (12.806895 13.86387 31.181482) with tilt (-8.0043024 0 0)
  triclinic box = (0 0 0) to (12.806895 13.86387 31.181482) with tilt (-8.0043092 0 0)
  triclinic box = (0 0 0) to (12.806895 13.86387 31.181482) with tilt (-8.0043092 0 0)
  triclinic box = (0 0 0) to (12.806895 13.86387 31.181482) with tilt (-8.0043092 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.6761      -477.79915      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15823 ave       15823 max       15823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15823
Ave neighs/atom = 66.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806894769006355 y final 0 13.863870266941943 z final 0 31.18148181787183      xy final -8.004309230628975 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806895 13.86387 31.181482) with tilt (-8.0043092 0 0)
  triclinic box = (0 0 0) to (12.806895 13.86387 31.181482) with tilt (-8.0043092 0 0)
  triclinic box = (0 0 0) to (12.806895 13.86387 31.181482) with tilt (-8.0043092 0 0)
  triclinic box = (0 0 0) to (12.806895 13.86387 31.181482) with tilt (-8.0043092 0 0)
  triclinic box = (0 0 0) to (12.806895 13.86387 31.181482) with tilt (-8.0043092 0 0)
  triclinic box = (0 0 0) to (12.806895 13.86387 31.181482) with tilt (-8.0043092 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.6761      -477.79915      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15823 ave       15823 max       15823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15823
Ave neighs/atom = 66.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806744834520622 y final 0 13.863707958100008 z final 0 31.18111676611455      xy final -8.004215521575391 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806745 13.86387 31.181482) with tilt (-8.0043092 0 0)
  triclinic box = (0 0 0) to (12.806745 13.863708 31.181482) with tilt (-8.0043092 0 0)
  triclinic box = (0 0 0) to (12.806745 13.863708 31.181117) with tilt (-8.0043092 0 0)
  triclinic box = (0 0 0) to (12.806745 13.863708 31.181117) with tilt (-8.0042155 0 0)
  triclinic box = (0 0 0) to (12.806745 13.863708 31.181117) with tilt (-8.0042155 0 0)
  triclinic box = (0 0 0) to (12.806745 13.863708 31.181117) with tilt (-8.0042155 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.6762      -464.5335       0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15823 ave       15823 max       15823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15823
Ave neighs/atom = 66.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80639327349301 y final 0 13.863327382123893 z final 0 31.180260805790983      xy final -8.003995795933132 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806393 13.863708 31.181117) with tilt (-8.0042155 0 0)
  triclinic box = (0 0 0) to (12.806393 13.863327 31.181117) with tilt (-8.0042155 0 0)
  triclinic box = (0 0 0) to (12.806393 13.863327 31.180261) with tilt (-8.0042155 0 0)
  triclinic box = (0 0 0) to (12.806393 13.863327 31.180261) with tilt (-8.0039958 0 0)
  triclinic box = (0 0 0) to (12.806393 13.863327 31.180261) with tilt (-8.0039958 0 0)
  triclinic box = (0 0 0) to (12.806393 13.863327 31.180261) with tilt (-8.0039958 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67629     -433.46586      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15823 ave       15823 max       15823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15823
Ave neighs/atom = 66.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80639327349301 y final 0 13.863327382123893 z final 0 31.18026080579098      xy final -8.003995795933132 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806393 13.863327 31.180261) with tilt (-8.0039958 0 0)
  triclinic box = (0 0 0) to (12.806393 13.863327 31.180261) with tilt (-8.0039958 0 0)
  triclinic box = (0 0 0) to (12.806393 13.863327 31.180261) with tilt (-8.0039958 0 0)
  triclinic box = (0 0 0) to (12.806393 13.863327 31.180261) with tilt (-8.0039958 0 0)
  triclinic box = (0 0 0) to (12.806393 13.863327 31.180261) with tilt (-8.0039958 0 0)
  triclinic box = (0 0 0) to (12.806393 13.863327 31.180261) with tilt (-8.0039958 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67629     -433.46586      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15823 ave       15823 max       15823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15823
Ave neighs/atom = 66.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80649085834727 y final 0 13.863433020827413 z final 0 31.18049839971314      xy final -8.004056786467046 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806491 13.863327 31.180261) with tilt (-8.0039958 0 0)
  triclinic box = (0 0 0) to (12.806491 13.863433 31.180261) with tilt (-8.0039958 0 0)
  triclinic box = (0 0 0) to (12.806491 13.863433 31.180498) with tilt (-8.0039958 0 0)
  triclinic box = (0 0 0) to (12.806491 13.863433 31.180498) with tilt (-8.0040568 0 0)
  triclinic box = (0 0 0) to (12.806491 13.863433 31.180498) with tilt (-8.0040568 0 0)
  triclinic box = (0 0 0) to (12.806491 13.863433 31.180498) with tilt (-8.0040568 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67633     -440.84991      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15823 ave       15823 max       15823 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15823
Ave neighs/atom = 66.483193
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806881205200355 y final 0 13.863855583691237 z final 0 31.18144879350661      xy final -8.004300753250224 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806881 13.863433 31.180498) with tilt (-8.0040568 0 0)
  triclinic box = (0 0 0) to (12.806881 13.863856 31.180498) with tilt (-8.0040568 0 0)
  triclinic box = (0 0 0) to (12.806881 13.863856 31.181449) with tilt (-8.0040568 0 0)
  triclinic box = (0 0 0) to (12.806881 13.863856 31.181449) with tilt (-8.0043008 0 0)
  triclinic box = (0 0 0) to (12.806881 13.863856 31.181449) with tilt (-8.0043008 0 0)
  triclinic box = (0 0 0) to (12.806881 13.863856 31.181449) with tilt (-8.0043008 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67638     -470.61821      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806881205200355 y final 0 13.863855583691237 z final 0 31.181448793506608      xy final -8.004300753250224 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806881 13.863856 31.181449) with tilt (-8.0043008 0 0)
  triclinic box = (0 0 0) to (12.806881 13.863856 31.181449) with tilt (-8.0043008 0 0)
  triclinic box = (0 0 0) to (12.806881 13.863856 31.181449) with tilt (-8.0043008 0 0)
  triclinic box = (0 0 0) to (12.806881 13.863856 31.181449) with tilt (-8.0043008 0 0)
  triclinic box = (0 0 0) to (12.806881 13.863856 31.181449) with tilt (-8.0043008 0 0)
  triclinic box = (0 0 0) to (12.806881 13.863856 31.181449) with tilt (-8.0043008 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67638     -470.61821      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806638481774936 y final 0 13.863592827875603 z final 0 31.18085782464078      xy final -8.004149051109337 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806638 13.863856 31.181449) with tilt (-8.0043008 0 0)
  triclinic box = (0 0 0) to (12.806638 13.863593 31.181449) with tilt (-8.0043008 0 0)
  triclinic box = (0 0 0) to (12.806638 13.863593 31.180858) with tilt (-8.0043008 0 0)
  triclinic box = (0 0 0) to (12.806638 13.863593 31.180858) with tilt (-8.0041491 0 0)
  triclinic box = (0 0 0) to (12.806638 13.863593 31.180858) with tilt (-8.0041491 0 0)
  triclinic box = (0 0 0) to (12.806638 13.863593 31.180858) with tilt (-8.0041491 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67642     -452.37514      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806638481774936 y final 0 13.863592827875602 z final 0 31.18085782464078      xy final -8.004149051109337 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806638 13.863593 31.180858) with tilt (-8.0041491 0 0)
  triclinic box = (0 0 0) to (12.806638 13.863593 31.180858) with tilt (-8.0041491 0 0)
  triclinic box = (0 0 0) to (12.806638 13.863593 31.180858) with tilt (-8.0041491 0 0)
  triclinic box = (0 0 0) to (12.806638 13.863593 31.180858) with tilt (-8.0041491 0 0)
  triclinic box = (0 0 0) to (12.806638 13.863593 31.180858) with tilt (-8.0041491 0 0)
  triclinic box = (0 0 0) to (12.806638 13.863593 31.180858) with tilt (-8.0041491 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67642     -452.37514      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806502008723056 y final 0 13.863445091463275 z final 0 31.18052554799897      xy final -8.004063755451913 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806502 13.863593 31.180858) with tilt (-8.0041491 0 0)
  triclinic box = (0 0 0) to (12.806502 13.863445 31.180858) with tilt (-8.0041491 0 0)
  triclinic box = (0 0 0) to (12.806502 13.863445 31.180526) with tilt (-8.0041491 0 0)
  triclinic box = (0 0 0) to (12.806502 13.863445 31.180526) with tilt (-8.0040638 0 0)
  triclinic box = (0 0 0) to (12.806502 13.863445 31.180526) with tilt (-8.0040638 0 0)
  triclinic box = (0 0 0) to (12.806502 13.863445 31.180526) with tilt (-8.0040638 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67644     -443.102        0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806502008723056 y final 0 13.863445091463275 z final 0 31.180525547998965      xy final -8.004063755451915 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806502 13.863445 31.180526) with tilt (-8.0040638 0 0)
  triclinic box = (0 0 0) to (12.806502 13.863445 31.180526) with tilt (-8.0040638 0 0)
  triclinic box = (0 0 0) to (12.806502 13.863445 31.180526) with tilt (-8.0040638 0 0)
  triclinic box = (0 0 0) to (12.806502 13.863445 31.180526) with tilt (-8.0040638 0 0)
  triclinic box = (0 0 0) to (12.806502 13.863445 31.180526) with tilt (-8.0040638 0 0)
  triclinic box = (0 0 0) to (12.806502 13.863445 31.180526) with tilt (-8.0040638 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67644     -443.102        0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806470882554995 y final 0 13.863411396397945 z final 0 31.180449763817006      xy final -8.004044301596878 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806471 13.863445 31.180526) with tilt (-8.0040638 0 0)
  triclinic box = (0 0 0) to (12.806471 13.863411 31.180526) with tilt (-8.0040638 0 0)
  triclinic box = (0 0 0) to (12.806471 13.863411 31.18045) with tilt (-8.0040638 0 0)
  triclinic box = (0 0 0) to (12.806471 13.863411 31.18045) with tilt (-8.0040443 0 0)
  triclinic box = (0 0 0) to (12.806471 13.863411 31.18045) with tilt (-8.0040443 0 0)
  triclinic box = (0 0 0) to (12.806471 13.863411 31.18045) with tilt (-8.0040443 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67645     -442.59889      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806470882554995 y final 0 13.863411396397945 z final 0 31.180449763817006      xy final -8.004044301596878 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806471 13.863411 31.18045) with tilt (-8.0040443 0 0)
  triclinic box = (0 0 0) to (12.806471 13.863411 31.18045) with tilt (-8.0040443 0 0)
  triclinic box = (0 0 0) to (12.806471 13.863411 31.18045) with tilt (-8.0040443 0 0)
  triclinic box = (0 0 0) to (12.806471 13.863411 31.18045) with tilt (-8.0040443 0 0)
  triclinic box = (0 0 0) to (12.806471 13.863411 31.18045) with tilt (-8.0040443 0 0)
  triclinic box = (0 0 0) to (12.806471 13.863411 31.18045) with tilt (-8.0040443 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67645     -442.59889      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806522213300282 y final 0 13.863466963559711 z final 0 31.18057474092735      xy final -8.004076383312684 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806522 13.863411 31.18045) with tilt (-8.0040443 0 0)
  triclinic box = (0 0 0) to (12.806522 13.863467 31.18045) with tilt (-8.0040443 0 0)
  triclinic box = (0 0 0) to (12.806522 13.863467 31.180575) with tilt (-8.0040443 0 0)
  triclinic box = (0 0 0) to (12.806522 13.863467 31.180575) with tilt (-8.0040764 0 0)
  triclinic box = (0 0 0) to (12.806522 13.863467 31.180575) with tilt (-8.0040764 0 0)
  triclinic box = (0 0 0) to (12.806522 13.863467 31.180575) with tilt (-8.0040764 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -448.25862      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806522213300282 y final 0 13.86346696355971 z final 0 31.180574740927348      xy final -8.004076383312686 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806522 13.863467 31.180575) with tilt (-8.0040764 0 0)
  triclinic box = (0 0 0) to (12.806522 13.863467 31.180575) with tilt (-8.0040764 0 0)
  triclinic box = (0 0 0) to (12.806522 13.863467 31.180575) with tilt (-8.0040764 0 0)
  triclinic box = (0 0 0) to (12.806522 13.863467 31.180575) with tilt (-8.0040764 0 0)
  triclinic box = (0 0 0) to (12.806522 13.863467 31.180575) with tilt (-8.0040764 0 0)
  triclinic box = (0 0 0) to (12.806522 13.863467 31.180575) with tilt (-8.0040764 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -448.25862      0              0            
Loop time of 1.5e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806584494882793 y final 0 13.863534385350512 z final 0 31.180726380498655      xy final -8.00411530930176 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806584 13.863467 31.180575) with tilt (-8.0040764 0 0)
  triclinic box = (0 0 0) to (12.806584 13.863534 31.180575) with tilt (-8.0040764 0 0)
  triclinic box = (0 0 0) to (12.806584 13.863534 31.180726) with tilt (-8.0040764 0 0)
  triclinic box = (0 0 0) to (12.806584 13.863534 31.180726) with tilt (-8.0041153 0 0)
  triclinic box = (0 0 0) to (12.806584 13.863534 31.180726) with tilt (-8.0041153 0 0)
  triclinic box = (0 0 0) to (12.806584 13.863534 31.180726) with tilt (-8.0041153 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -454.41174      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806537290814633 y final 0 13.863483285447778 z final 0 31.180611450781537      xy final -8.004085806759155 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806537 13.863534 31.180726) with tilt (-8.0041153 0 0)
  triclinic box = (0 0 0) to (12.806537 13.863483 31.180726) with tilt (-8.0041153 0 0)
  triclinic box = (0 0 0) to (12.806537 13.863483 31.180611) with tilt (-8.0041153 0 0)
  triclinic box = (0 0 0) to (12.806537 13.863483 31.180611) with tilt (-8.0040858 0 0)
  triclinic box = (0 0 0) to (12.806537 13.863483 31.180611) with tilt (-8.0040858 0 0)
  triclinic box = (0 0 0) to (12.806537 13.863483 31.180611) with tilt (-8.0040858 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -449.79233      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806568445480098 y final 0 13.863517011362449 z final 0 31.180687304347305      xy final -8.004105278425072 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806568 13.863483 31.180611) with tilt (-8.0040858 0 0)
  triclinic box = (0 0 0) to (12.806568 13.863517 31.180611) with tilt (-8.0040858 0 0)
  triclinic box = (0 0 0) to (12.806568 13.863517 31.180687) with tilt (-8.0040858 0 0)
  triclinic box = (0 0 0) to (12.806568 13.863517 31.180687) with tilt (-8.0041053 0 0)
  triclinic box = (0 0 0) to (12.806568 13.863517 31.180687) with tilt (-8.0041053 0 0)
  triclinic box = (0 0 0) to (12.806568 13.863517 31.180687) with tilt (-8.0041053 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -452.8254       0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806540364183117 y final 0 13.863486612466755 z final 0 31.1806189336402      xy final -8.00408772761446 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.80654 13.863517 31.180687) with tilt (-8.0041053 0 0)
  triclinic box = (0 0 0) to (12.80654 13.863487 31.180687) with tilt (-8.0041053 0 0)
  triclinic box = (0 0 0) to (12.80654 13.863487 31.180619) with tilt (-8.0041053 0 0)
  triclinic box = (0 0 0) to (12.80654 13.863487 31.180619) with tilt (-8.0040877 0 0)
  triclinic box = (0 0 0) to (12.80654 13.863487 31.180619) with tilt (-8.0040877 0 0)
  triclinic box = (0 0 0) to (12.80654 13.863487 31.180619) with tilt (-8.0040877 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.09625      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806558897832215 y final 0 13.863506675730434 z final 0 31.18066405829007      xy final -8.004099311145145 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806559 13.863487 31.180619) with tilt (-8.0040877 0 0)
  triclinic box = (0 0 0) to (12.806559 13.863507 31.180619) with tilt (-8.0040877 0 0)
  triclinic box = (0 0 0) to (12.806559 13.863507 31.180664) with tilt (-8.0040877 0 0)
  triclinic box = (0 0 0) to (12.806559 13.863507 31.180664) with tilt (-8.0040993 0 0)
  triclinic box = (0 0 0) to (12.806559 13.863507 31.180664) with tilt (-8.0040993 0 0)
  triclinic box = (0 0 0) to (12.806559 13.863507 31.180664) with tilt (-8.0040993 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -451.88155      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80654139038946 y final 0 13.863487723367708 z final 0 31.180621432187667      xy final -8.004088368993424 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806541 13.863507 31.180664) with tilt (-8.0040993 0 0)
  triclinic box = (0 0 0) to (12.806541 13.863488 31.180664) with tilt (-8.0040993 0 0)
  triclinic box = (0 0 0) to (12.806541 13.863488 31.180621) with tilt (-8.0040993 0 0)
  triclinic box = (0 0 0) to (12.806541 13.863488 31.180621) with tilt (-8.0040884 0 0)
  triclinic box = (0 0 0) to (12.806541 13.863488 31.180621) with tilt (-8.0040884 0 0)
  triclinic box = (0 0 0) to (12.806541 13.863488 31.180621) with tilt (-8.0040884 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.19773      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806552945298993 y final 0 13.863500231924201 z final 0 31.180649565408714      xy final -8.004095590811882 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806553 13.863488 31.180621) with tilt (-8.0040884 0 0)
  triclinic box = (0 0 0) to (12.806553 13.8635 31.180621) with tilt (-8.0040884 0 0)
  triclinic box = (0 0 0) to (12.806553 13.8635 31.18065) with tilt (-8.0040884 0 0)
  triclinic box = (0 0 0) to (12.806553 13.8635 31.18065) with tilt (-8.0040956 0 0)
  triclinic box = (0 0 0) to (12.806553 13.8635 31.18065) with tilt (-8.0040956 0 0)
  triclinic box = (0 0 0) to (12.806553 13.8635 31.18065) with tilt (-8.0040956 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -451.29304      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806541778609944 y final 0 13.863488143628711 z final 0 31.180622377404298      xy final -8.004088611631227 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806542 13.8635 31.18065) with tilt (-8.0040956 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.18065) with tilt (-8.0040956 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180622) with tilt (-8.0040956 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180622) with tilt (-8.0040886 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180622) with tilt (-8.0040886 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180622) with tilt (-8.0040886 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.23612      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806549148623624 y final 0 13.863496121902552 z final 0 31.180640321484553      xy final -8.004093217889777 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806549 13.863488 31.180622) with tilt (-8.0040886 0 0)
  triclinic box = (0 0 0) to (12.806549 13.863496 31.180622) with tilt (-8.0040886 0 0)
  triclinic box = (0 0 0) to (12.806549 13.863496 31.18064) with tilt (-8.0040886 0 0)
  triclinic box = (0 0 0) to (12.806549 13.863496 31.18064) with tilt (-8.0040932 0 0)
  triclinic box = (0 0 0) to (12.806549 13.863496 31.18064) with tilt (-8.0040932 0 0)
  triclinic box = (0 0 0) to (12.806549 13.863496 31.18064) with tilt (-8.0040932 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -450.91765      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806541934261372 y final 0 13.863488312126323 z final 0 31.180622756375335      xy final -8.004088708913368 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806542 13.863496 31.18064) with tilt (-8.0040932 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.18064) with tilt (-8.0040932 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180623) with tilt (-8.0040932 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180623) with tilt (-8.0040887 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180623) with tilt (-8.0040887 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180623) with tilt (-8.0040887 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.25151      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806546695740002 y final 0 13.86349346657814 z final 0 31.18063434934631      xy final -8.004091684837512 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806547 13.863488 31.180623) with tilt (-8.0040887 0 0)
  triclinic box = (0 0 0) to (12.806547 13.863493 31.180623) with tilt (-8.0040887 0 0)
  triclinic box = (0 0 0) to (12.806547 13.863493 31.180634) with tilt (-8.0040887 0 0)
  triclinic box = (0 0 0) to (12.806547 13.863493 31.180634) with tilt (-8.0040917 0 0)
  triclinic box = (0 0 0) to (12.806547 13.863493 31.180634) with tilt (-8.0040917 0 0)
  triclinic box = (0 0 0) to (12.806547 13.863493 31.180634) with tilt (-8.0040917 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -450.67512      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80654199868264 y final 0 13.863488381864391 z final 0 31.180622913224486      xy final -8.004088749176661 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806542 13.863493 31.180634) with tilt (-8.0040917 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180634) with tilt (-8.0040917 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180623) with tilt (-8.0040917 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180623) with tilt (-8.0040887 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180623) with tilt (-8.0040887 0 0)
  triclinic box = (0 0 0) to (12.806542 13.863488 31.180623) with tilt (-8.0040887 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.25788      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544055553193 y final 0 13.863490608492082 z final 0 31.180627921173166      xy final -8.004090034720758 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806544 13.863488 31.180623) with tilt (-8.0040887 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180623) with tilt (-8.0040887 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040887 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.00409 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.00409 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.00409 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.46127      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806545798076426 y final 0 13.863492494828815 z final 0 31.180632163767292      xy final -8.004091123797778 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806546 13.863491 31.180628) with tilt (-8.00409 0 0)
  triclinic box = (0 0 0) to (12.806546 13.863492 31.180628) with tilt (-8.00409 0 0)
  triclinic box = (0 0 0) to (12.806546 13.863492 31.180632) with tilt (-8.00409 0 0)
  triclinic box = (0 0 0) to (12.806546 13.863492 31.180632) with tilt (-8.0040911 0 0)
  triclinic box = (0 0 0) to (12.806546 13.863492 31.180632) with tilt (-8.0040911 0 0)
  triclinic box = (0 0 0) to (12.806546 13.863492 31.180632) with tilt (-8.0040911 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -450.58636      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544063910398 y final 0 13.863490617539021 z final 0 31.180627941520804      xy final -8.00409003994401 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806544 13.863492 31.180632) with tilt (-8.0040911 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180632) with tilt (-8.0040911 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040911 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.00409 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.00409 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.00409 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.4621       0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544113736223 y final 0 13.863490671477058 z final 0 31.18062806283382      xy final -8.00409007108515 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.00409 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.00409 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.00409 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040901 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.46703      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544955906297 y final 0 13.863491583152905 z final 0 31.18063011330049      xy final -8.004090597441447 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180628) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863492 31.180628) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863492 31.18063) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863492 31.18063) with tilt (-8.0040906 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863492 31.18063) with tilt (-8.0040906 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863492 31.18063) with tilt (-8.0040906 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -450.50308      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544115682314 y final 0 13.863490673583765 z final 0 31.18062806757205      xy final -8.004090072301459 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806544 13.863492 31.18063) with tilt (-8.0040906 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.18063) with tilt (-8.0040906 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040906 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040901 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.46722      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544670230137 y final 0 13.863491273899392 z final 0 31.180629417752804      xy final -8.004090418893847 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180628) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180628) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -450.47483      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80654411652544 y final 0 13.863490674496475 z final 0 31.180628069624845      xy final -8.004090072828411 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040901 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.4673       0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544481970539 y final 0 13.863491070102398 z final 0 31.180628959389292      xy final -8.004090301231598 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180628) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180629) with tilt (-8.0040901 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180629) with tilt (-8.0040903 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180629) with tilt (-8.0040903 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180629) with tilt (-8.0040903 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.50344      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544606221873 y final 0 13.863491204608414 z final 0 31.18062926190921      xy final -8.004090378888682 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040903 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040903 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040903 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -450.46851      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80654448201264 y final 0 13.863491070147974 z final 0 31.180628959491802      xy final -8.00409030125791 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180629) with tilt (-8.0040903 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180629) with tilt (-8.0040903 0 0)
  triclinic box = (0 0 0) to (12.806544 13.863491 31.180629) with tilt (-8.0040903 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.50344      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544563990734 y final 0 13.863491158891863 z final 0 31.180629159087292      xy final -8.00409035249422 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040903 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040903 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040903 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.51155      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544591863286 y final 0 13.863491189064787 z final 0 31.18062922694976      xy final -8.004090369914564 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67646     -450.46709      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80654456399285 y final 0 13.863491158894155 z final 0 31.18062915909244      xy final -8.004090352495542 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.51155      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544563990895 y final 0 13.863491158892039 z final 0 31.180629159087687      xy final -8.004090352494321 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.51155      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544563990734 y final 0 13.863491158891865 z final 0 31.180629159087292      xy final -8.00409035249422 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.51155      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544563990816 y final 0 13.86349115889195 z final 0 31.180629159087488      xy final -8.004090352494272 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.51155      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544563990734 y final 0 13.863491158891865 z final 0 31.180629159087292      xy final -8.00409035249422 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.51155      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544563990734 y final 0 13.863491158891865 z final 0 31.180629159087292      xy final -8.00409035249422 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.51155      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544563990734 y final 0 13.863491158891865 z final 0 31.180629159087292      xy final -8.00409035249422 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.51155      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544563990734 y final 0 13.863491158891865 z final 0 31.180629159087292      xy final -8.00409035249422 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.51155      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806544563990734 y final 0 13.863491158891865 z final 0 31.180629159087292      xy final -8.00409035249422 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.806545 13.863491 31.180629) with tilt (-8.0040904 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67647     -450.51155      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15827 ave       15827 max       15827 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15827
Ave neighs/atom = 66.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.810913893863672 y final 0 13.8682210972262 z final 0 31.19126734909719      xy final -8.006821183664796 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.810914 13.863491 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.180629) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0040904 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.65463     -676.0198       0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.03248685092035 y final 0 13.025549104499053 z final 0 29.295998501818282      xy final -7.520304281825219 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.032487 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.032487 13.025549 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.032487 13.025549 29.295999) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.032487 13.025549 29.295999) with tilt (-7.5203043 0 0)
  triclinic box = (0 0 0) to (12.032487 13.025549 29.295999) with tilt (-7.5203043 0 0)
  triclinic box = (0 0 0) to (12.032487 13.025549 29.295999) with tilt (-7.5203043 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-341.98551      104789.79      0              0            
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
Nghost:           2250 ave        2250 max        2250 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          18646 ave       18646 max       18646 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 18646
Ave neighs/atom = 78.344538
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808668242259904 y final 0 13.86579010805506 z final 0 31.185799767305102      xy final -8.00541765141244 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808668 13.025549 29.295999) with tilt (-7.5203043 0 0)
  triclinic box = (0 0 0) to (12.808668 13.86579 29.295999) with tilt (-7.5203043 0 0)
  triclinic box = (0 0 0) to (12.808668 13.86579 31.1858) with tilt (-7.5203043 0 0)
  triclinic box = (0 0 0) to (12.808668 13.86579 31.1858) with tilt (-8.0054177 0 0)
  triclinic box = (0 0 0) to (12.808668 13.86579 31.1858) with tilt (-8.0054177 0 0)
  triclinic box = (0 0 0) to (12.808668 13.86579 31.1858) with tilt (-8.0054177 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.65481     -495.4413       0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808668242259904 y final 0 13.86579010805506 z final 0 31.1857997673051      xy final -8.00541765141244 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808668 13.86579 31.1858) with tilt (-8.0054177 0 0)
  triclinic box = (0 0 0) to (12.808668 13.86579 31.1858) with tilt (-8.0054177 0 0)
  triclinic box = (0 0 0) to (12.808668 13.86579 31.1858) with tilt (-8.0054177 0 0)
  triclinic box = (0 0 0) to (12.808668 13.86579 31.1858) with tilt (-8.0054177 0 0)
  triclinic box = (0 0 0) to (12.808668 13.86579 31.1858) with tilt (-8.0054177 0 0)
  triclinic box = (0 0 0) to (12.808668 13.86579 31.1858) with tilt (-8.0054177 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.65481     -495.4413       0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808146744425837 y final 0 13.86522557008966 z final 0 31.184530054738133      xy final -8.005091715266149 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808147 13.86579 31.1858) with tilt (-8.0054177 0 0)
  triclinic box = (0 0 0) to (12.808147 13.865226 31.1858) with tilt (-8.0054177 0 0)
  triclinic box = (0 0 0) to (12.808147 13.865226 31.18453) with tilt (-8.0054177 0 0)
  triclinic box = (0 0 0) to (12.808147 13.865226 31.18453) with tilt (-8.0050917 0 0)
  triclinic box = (0 0 0) to (12.808147 13.865226 31.18453) with tilt (-8.0050917 0 0)
  triclinic box = (0 0 0) to (12.808147 13.865226 31.18453) with tilt (-8.0050917 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.66184     -484.26061      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80665176513409 y final 0 13.863607207533677 z final 0 31.180890166190014      xy final -8.004157353208807 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806652 13.865226 31.18453) with tilt (-8.0050917 0 0)
  triclinic box = (0 0 0) to (12.806652 13.863607 31.18453) with tilt (-8.0050917 0 0)
  triclinic box = (0 0 0) to (12.806652 13.863607 31.18089) with tilt (-8.0050917 0 0)
  triclinic box = (0 0 0) to (12.806652 13.863607 31.18089) with tilt (-8.0041574 0 0)
  triclinic box = (0 0 0) to (12.806652 13.863607 31.18089) with tilt (-8.0041574 0 0)
  triclinic box = (0 0 0) to (12.806652 13.863607 31.18089) with tilt (-8.0041574 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67067     -441.47704      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80665176513409 y final 0 13.863607207533676 z final 0 31.18089016619001      xy final -8.004157353208809 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806652 13.863607 31.18089) with tilt (-8.0041574 0 0)
  triclinic box = (0 0 0) to (12.806652 13.863607 31.18089) with tilt (-8.0041574 0 0)
  triclinic box = (0 0 0) to (12.806652 13.863607 31.18089) with tilt (-8.0041574 0 0)
  triclinic box = (0 0 0) to (12.806652 13.863607 31.18089) with tilt (-8.0041574 0 0)
  triclinic box = (0 0 0) to (12.806652 13.863607 31.18089) with tilt (-8.0041574 0 0)
  triclinic box = (0 0 0) to (12.806652 13.863607 31.18089) with tilt (-8.0041574 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67067     -441.47704      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.807019362280034 y final 0 13.86400514311711 z final 0 31.181785170320914      xy final -8.004387101425024 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.807019 13.863607 31.18089) with tilt (-8.0041574 0 0)
  triclinic box = (0 0 0) to (12.807019 13.864005 31.18089) with tilt (-8.0041574 0 0)
  triclinic box = (0 0 0) to (12.807019 13.864005 31.181785) with tilt (-8.0041574 0 0)
  triclinic box = (0 0 0) to (12.807019 13.864005 31.181785) with tilt (-8.0043871 0 0)
  triclinic box = (0 0 0) to (12.807019 13.864005 31.181785) with tilt (-8.0043871 0 0)
  triclinic box = (0 0 0) to (12.807019 13.864005 31.181785) with tilt (-8.0043871 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.6725      -476.22215      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808155323561753 y final 0 13.865234857276715 z final 0 31.184550942718502      xy final -8.005097077226099 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808155 13.864005 31.181785) with tilt (-8.0043871 0 0)
  triclinic box = (0 0 0) to (12.808155 13.865235 31.181785) with tilt (-8.0043871 0 0)
  triclinic box = (0 0 0) to (12.808155 13.865235 31.184551) with tilt (-8.0043871 0 0)
  triclinic box = (0 0 0) to (12.808155 13.865235 31.184551) with tilt (-8.0050971 0 0)
  triclinic box = (0 0 0) to (12.808155 13.865235 31.184551) with tilt (-8.0050971 0 0)
  triclinic box = (0 0 0) to (12.808155 13.865235 31.184551) with tilt (-8.0050971 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67502     -580.77764      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15817 ave       15817 max       15817 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15817
Ave neighs/atom = 66.457983
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808155323561753 y final 0 13.865234857276715 z final 0 31.1845509427185      xy final -8.005097077226099 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808155 13.865235 31.184551) with tilt (-8.0050971 0 0)
  triclinic box = (0 0 0) to (12.808155 13.865235 31.184551) with tilt (-8.0050971 0 0)
  triclinic box = (0 0 0) to (12.808155 13.865235 31.184551) with tilt (-8.0050971 0 0)
  triclinic box = (0 0 0) to (12.808155 13.865235 31.184551) with tilt (-8.0050971 0 0)
  triclinic box = (0 0 0) to (12.808155 13.865235 31.184551) with tilt (-8.0050971 0 0)
  triclinic box = (0 0 0) to (12.808155 13.865235 31.184551) with tilt (-8.0050971 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67502     -580.77764      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15817 ave       15817 max       15817 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15817
Ave neighs/atom = 66.457983
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806859792383829 y final 0 13.863832403637367 z final 0 31.18139665882708      xy final -8.004287370239897 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.80686 13.865235 31.184551) with tilt (-8.0050971 0 0)
  triclinic box = (0 0 0) to (12.80686 13.863832 31.184551) with tilt (-8.0050971 0 0)
  triclinic box = (0 0 0) to (12.80686 13.863832 31.181397) with tilt (-8.0050971 0 0)
  triclinic box = (0 0 0) to (12.80686 13.863832 31.181397) with tilt (-8.0042874 0 0)
  triclinic box = (0 0 0) to (12.80686 13.863832 31.181397) with tilt (-8.0042874 0 0)
  triclinic box = (0 0 0) to (12.80686 13.863832 31.181397) with tilt (-8.0042874 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67562     -476.5573       0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.804408975269872 y final 0 13.861179316286476 z final 0 31.175429551994686      xy final -8.002755609543675 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.804409 13.863832 31.181397) with tilt (-8.0042874 0 0)
  triclinic box = (0 0 0) to (12.804409 13.861179 31.181397) with tilt (-8.0042874 0 0)
  triclinic box = (0 0 0) to (12.804409 13.861179 31.17543) with tilt (-8.0042874 0 0)
  triclinic box = (0 0 0) to (12.804409 13.861179 31.17543) with tilt (-8.0027556 0 0)
  triclinic box = (0 0 0) to (12.804409 13.861179 31.17543) with tilt (-8.0027556 0 0)
  triclinic box = (0 0 0) to (12.804409 13.861179 31.17543) with tilt (-8.0027556 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67609     -277.87588      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.804408975269872 y final 0 13.861179316286476 z final 0 31.175429551994682      xy final -8.002755609543676 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.804409 13.861179 31.17543) with tilt (-8.0027556 0 0)
  triclinic box = (0 0 0) to (12.804409 13.861179 31.17543) with tilt (-8.0027556 0 0)
  triclinic box = (0 0 0) to (12.804409 13.861179 31.17543) with tilt (-8.0027556 0 0)
  triclinic box = (0 0 0) to (12.804409 13.861179 31.17543) with tilt (-8.0027556 0 0)
  triclinic box = (0 0 0) to (12.804409 13.861179 31.17543) with tilt (-8.0027556 0 0)
  triclinic box = (0 0 0) to (12.804409 13.861179 31.17543) with tilt (-8.0027556 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67609     -277.87588      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80561601274467 y final 0 13.862485972682093 z final 0 31.178368376569466      xy final -8.003510007965424 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805616 13.861179 31.17543) with tilt (-8.0027556 0 0)
  triclinic box = (0 0 0) to (12.805616 13.862486 31.17543) with tilt (-8.0027556 0 0)
  triclinic box = (0 0 0) to (12.805616 13.862486 31.178368) with tilt (-8.0027556 0 0)
  triclinic box = (0 0 0) to (12.805616 13.862486 31.178368) with tilt (-8.00351 0 0)
  triclinic box = (0 0 0) to (12.805616 13.862486 31.178368) with tilt (-8.00351 0 0)
  triclinic box = (0 0 0) to (12.805616 13.862486 31.178368) with tilt (-8.00351 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67648     -373.99311      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.810445300593154 y final 0 13.867713830130809 z final 0 31.19012644548129      xy final -8.006528312870728 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.810445 13.862486 31.178368) with tilt (-8.00351 0 0)
  triclinic box = (0 0 0) to (12.810445 13.867714 31.178368) with tilt (-8.00351 0 0)
  triclinic box = (0 0 0) to (12.810445 13.867714 31.190126) with tilt (-8.00351 0 0)
  triclinic box = (0 0 0) to (12.810445 13.867714 31.190126) with tilt (-8.0065283 0 0)
  triclinic box = (0 0 0) to (12.810445 13.867714 31.190126) with tilt (-8.0065283 0 0)
  triclinic box = (0 0 0) to (12.810445 13.867714 31.190126) with tilt (-8.0065283 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67577     -756.01196      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.807034044828356 y final 0 13.86402103744191 z final 0 31.18182091853471      xy final -8.00439627801773 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.807034 13.867714 31.190126) with tilt (-8.0065283 0 0)
  triclinic box = (0 0 0) to (12.807034 13.864021 31.190126) with tilt (-8.0065283 0 0)
  triclinic box = (0 0 0) to (12.807034 13.864021 31.181821) with tilt (-8.0065283 0 0)
  triclinic box = (0 0 0) to (12.807034 13.864021 31.181821) with tilt (-8.0043963 0 0)
  triclinic box = (0 0 0) to (12.807034 13.864021 31.181821) with tilt (-8.0043963 0 0)
  triclinic box = (0 0 0) to (12.807034 13.864021 31.181821) with tilt (-8.0043963 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67664     -486.66506      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.807034044828356 y final 0 13.86402103744191 z final 0 31.181820918534708      xy final -8.00439627801773 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.807034 13.864021 31.181821) with tilt (-8.0043963 0 0)
  triclinic box = (0 0 0) to (12.807034 13.864021 31.181821) with tilt (-8.0043963 0 0)
  triclinic box = (0 0 0) to (12.807034 13.864021 31.181821) with tilt (-8.0043963 0 0)
  triclinic box = (0 0 0) to (12.807034 13.864021 31.181821) with tilt (-8.0043963 0 0)
  triclinic box = (0 0 0) to (12.807034 13.864021 31.181821) with tilt (-8.0043963 0 0)
  triclinic box = (0 0 0) to (12.807034 13.864021 31.181821) with tilt (-8.0043963 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67664     -486.66506      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.807024519230085 y final 0 13.864010725679298 z final 0 31.181797726162575      xy final -8.00439032451881 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.807025 13.864021 31.181821) with tilt (-8.0043963 0 0)
  triclinic box = (0 0 0) to (12.807025 13.864011 31.181821) with tilt (-8.0043963 0 0)
  triclinic box = (0 0 0) to (12.807025 13.864011 31.181798) with tilt (-8.0043963 0 0)
  triclinic box = (0 0 0) to (12.807025 13.864011 31.181798) with tilt (-8.0043903 0 0)
  triclinic box = (0 0 0) to (12.807025 13.864011 31.181798) with tilt (-8.0043903 0 0)
  triclinic box = (0 0 0) to (12.807025 13.864011 31.181798) with tilt (-8.0043903 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67681     -484.60392      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806986416907852 y final 0 13.863969478705554 z final 0 31.181704956846545      xy final -8.004366510567417 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806986 13.864011 31.181798) with tilt (-8.0043903 0 0)
  triclinic box = (0 0 0) to (12.806986 13.863969 31.181798) with tilt (-8.0043903 0 0)
  triclinic box = (0 0 0) to (12.806986 13.863969 31.181705) with tilt (-8.0043903 0 0)
  triclinic box = (0 0 0) to (12.806986 13.863969 31.181705) with tilt (-8.0043665 0 0)
  triclinic box = (0 0 0) to (12.806986 13.863969 31.181705) with tilt (-8.0043665 0 0)
  triclinic box = (0 0 0) to (12.806986 13.863969 31.181705) with tilt (-8.0043665 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67715     -476.86764      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806986416907852 y final 0 13.863969478705554 z final 0 31.181704956846545      xy final -8.004366510567417 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806986 13.863969 31.181705) with tilt (-8.0043665 0 0)
  triclinic box = (0 0 0) to (12.806986 13.863969 31.181705) with tilt (-8.0043665 0 0)
  triclinic box = (0 0 0) to (12.806986 13.863969 31.181705) with tilt (-8.0043665 0 0)
  triclinic box = (0 0 0) to (12.806986 13.863969 31.181705) with tilt (-8.0043665 0 0)
  triclinic box = (0 0 0) to (12.806986 13.863969 31.181705) with tilt (-8.0043665 0 0)
  triclinic box = (0 0 0) to (12.806986 13.863969 31.181705) with tilt (-8.0043665 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67715     -476.86764      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806558658847177 y final 0 13.863506417021531 z final 0 31.180663476422765      xy final -8.004099161779495 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806559 13.863969 31.181705) with tilt (-8.0043665 0 0)
  triclinic box = (0 0 0) to (12.806559 13.863506 31.181705) with tilt (-8.0043665 0 0)
  triclinic box = (0 0 0) to (12.806559 13.863506 31.180663) with tilt (-8.0043665 0 0)
  triclinic box = (0 0 0) to (12.806559 13.863506 31.180663) with tilt (-8.0040992 0 0)
  triclinic box = (0 0 0) to (12.806559 13.863506 31.180663) with tilt (-8.0040992 0 0)
  triclinic box = (0 0 0) to (12.806559 13.863506 31.180663) with tilt (-8.0040992 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67722     -441.67659      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80484776947247 y final 0 13.86165432494458 z final 0 31.176497902574333      xy final -8.003029855920301 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.804848 13.863506 31.180663) with tilt (-8.0040992 0 0)
  triclinic box = (0 0 0) to (12.804848 13.861654 31.180663) with tilt (-8.0040992 0 0)
  triclinic box = (0 0 0) to (12.804848 13.861654 31.176498) with tilt (-8.0040992 0 0)
  triclinic box = (0 0 0) to (12.804848 13.861654 31.176498) with tilt (-8.0030299 0 0)
  triclinic box = (0 0 0) to (12.804848 13.861654 31.176498) with tilt (-8.0030299 0 0)
  triclinic box = (0 0 0) to (12.804848 13.861654 31.176498) with tilt (-8.0030299 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67695     -301.03808      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806117054085492 y final 0 13.863028365843968 z final 0 31.17958828285747      xy final -8.003823158803442 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806117 13.861654 31.176498) with tilt (-8.0030299 0 0)
  triclinic box = (0 0 0) to (12.806117 13.863028 31.176498) with tilt (-8.0030299 0 0)
  triclinic box = (0 0 0) to (12.806117 13.863028 31.179588) with tilt (-8.0030299 0 0)
  triclinic box = (0 0 0) to (12.806117 13.863028 31.179588) with tilt (-8.0038232 0 0)
  triclinic box = (0 0 0) to (12.806117 13.863028 31.179588) with tilt (-8.0038232 0 0)
  triclinic box = (0 0 0) to (12.806117 13.863028 31.179588) with tilt (-8.0038232 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67725     -405.44296      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806117054085492 y final 0 13.863028365843968 z final 0 31.179588282857466      xy final -8.003823158803442 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806117 13.863028 31.179588) with tilt (-8.0038232 0 0)
  triclinic box = (0 0 0) to (12.806117 13.863028 31.179588) with tilt (-8.0038232 0 0)
  triclinic box = (0 0 0) to (12.806117 13.863028 31.179588) with tilt (-8.0038232 0 0)
  triclinic box = (0 0 0) to (12.806117 13.863028 31.179588) with tilt (-8.0038232 0 0)
  triclinic box = (0 0 0) to (12.806117 13.863028 31.179588) with tilt (-8.0038232 0 0)
  triclinic box = (0 0 0) to (12.806117 13.863028 31.179588) with tilt (-8.0038232 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67725     -405.44296      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806227282846038 y final 0 13.863147691977543 z final 0 31.179856661426562      xy final -8.003892051778784 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806227 13.863028 31.179588) with tilt (-8.0038232 0 0)
  triclinic box = (0 0 0) to (12.806227 13.863148 31.179588) with tilt (-8.0038232 0 0)
  triclinic box = (0 0 0) to (12.806227 13.863148 31.179857) with tilt (-8.0038232 0 0)
  triclinic box = (0 0 0) to (12.806227 13.863148 31.179857) with tilt (-8.0038921 0 0)
  triclinic box = (0 0 0) to (12.806227 13.863148 31.179857) with tilt (-8.0038921 0 0)
  triclinic box = (0 0 0) to (12.806227 13.863148 31.179857) with tilt (-8.0038921 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.6773      -412.96175      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806668207376255 y final 0 13.863625006782943 z final 0 31.180930198803868      xy final -8.004167629610173 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806668 13.863148 31.179857) with tilt (-8.0038921 0 0)
  triclinic box = (0 0 0) to (12.806668 13.863625 31.179857) with tilt (-8.0038921 0 0)
  triclinic box = (0 0 0) to (12.806668 13.863625 31.18093) with tilt (-8.0038921 0 0)
  triclinic box = (0 0 0) to (12.806668 13.863625 31.18093) with tilt (-8.0041676 0 0)
  triclinic box = (0 0 0) to (12.806668 13.863625 31.18093) with tilt (-8.0041676 0 0)
  triclinic box = (0 0 0) to (12.806668 13.863625 31.18093) with tilt (-8.0041676 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67733     -443.71924      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806465008524855 y final 0 13.863405037573777 z final 0 31.180435462069905      xy final -8.004040630328046 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806465 13.863625 31.18093) with tilt (-8.0041676 0 0)
  triclinic box = (0 0 0) to (12.806465 13.863405 31.18093) with tilt (-8.0041676 0 0)
  triclinic box = (0 0 0) to (12.806465 13.863405 31.180435) with tilt (-8.0041676 0 0)
  triclinic box = (0 0 0) to (12.806465 13.863405 31.180435) with tilt (-8.0040406 0 0)
  triclinic box = (0 0 0) to (12.806465 13.863405 31.180435) with tilt (-8.0040406 0 0)
  triclinic box = (0 0 0) to (12.806465 13.863405 31.180435) with tilt (-8.0040406 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.67739     -429.6446       0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15821 ave       15821 max       15821 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15821
Ave neighs/atom = 66.47479
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.810913893863672 y final 0 13.8682210972262 z final 0 31.19126734909719      xy final -8.006821183664796 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.810914 13.863405 31.180435) with tilt (-8.0040406 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.180435) with tilt (-8.0040406 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0040406 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.58697     -756.64603      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.826985116157559 y final 0 12.803086950966119 z final 0 28.79565484149911      xy final -7.391865697598475 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.826985 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (11.826985 12.803087 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (11.826985 12.803087 28.795655) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (11.826985 12.803087 28.795655) with tilt (-7.3918657 0 0)
  triclinic box = (0 0 0) to (11.826985 12.803087 28.795655) with tilt (-7.3918657 0 0)
  triclinic box = (0 0 0) to (11.826985 12.803087 28.795655) with tilt (-7.3918657 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-323.10109      154516.72      0              0            
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
Nghost:           2250 ave        2250 max        2250 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          18645 ave       18645 max       18645 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 18645
Ave neighs/atom = 78.340336
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808136358961397 y final 0 13.865214327494614 z final 0 31.184504768813998      xy final -8.005085224350873 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808136 12.803087 28.795655) with tilt (-7.3918657 0 0)
  triclinic box = (0 0 0) to (12.808136 13.865214 28.795655) with tilt (-7.3918657 0 0)
  triclinic box = (0 0 0) to (12.808136 13.865214 31.184505) with tilt (-7.3918657 0 0)
  triclinic box = (0 0 0) to (12.808136 13.865214 31.184505) with tilt (-8.0050852 0 0)
  triclinic box = (0 0 0) to (12.808136 13.865214 31.184505) with tilt (-8.0050852 0 0)
  triclinic box = (0 0 0) to (12.808136 13.865214 31.184505) with tilt (-8.0050852 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.58726     -533.78079      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808136358961397 y final 0 13.865214327494614 z final 0 31.184504768813998      xy final -8.005085224350873 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808136 13.865214 31.184505) with tilt (-8.0050852 0 0)
  triclinic box = (0 0 0) to (12.808136 13.865214 31.184505) with tilt (-8.0050852 0 0)
  triclinic box = (0 0 0) to (12.808136 13.865214 31.184505) with tilt (-8.0050852 0 0)
  triclinic box = (0 0 0) to (12.808136 13.865214 31.184505) with tilt (-8.0050852 0 0)
  triclinic box = (0 0 0) to (12.808136 13.865214 31.184505) with tilt (-8.0050852 0 0)
  triclinic box = (0 0 0) to (12.808136 13.865214 31.184505) with tilt (-8.0050852 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.58726     -533.78079      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.807547589236778 y final 0 13.864576965571466 z final 0 31.18307126656429      xy final -8.004717243272987 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.807548 13.865214 31.184505) with tilt (-8.0050852 0 0)
  triclinic box = (0 0 0) to (12.807548 13.864577 31.184505) with tilt (-8.0050852 0 0)
  triclinic box = (0 0 0) to (12.807548 13.864577 31.183071) with tilt (-8.0050852 0 0)
  triclinic box = (0 0 0) to (12.807548 13.864577 31.183071) with tilt (-8.0047172 0 0)
  triclinic box = (0 0 0) to (12.807548 13.864577 31.183071) with tilt (-8.0047172 0 0)
  triclinic box = (0 0 0) to (12.807548 13.864577 31.183071) with tilt (-8.0047172 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59504     -521.98517      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80595146679097 y final 0 13.862849112339372 z final 0 31.179185120553043      xy final -8.003719666744358 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805951 13.864577 31.183071) with tilt (-8.0047172 0 0)
  triclinic box = (0 0 0) to (12.805951 13.862849 31.183071) with tilt (-8.0047172 0 0)
  triclinic box = (0 0 0) to (12.805951 13.862849 31.179185) with tilt (-8.0047172 0 0)
  triclinic box = (0 0 0) to (12.805951 13.862849 31.179185) with tilt (-8.0037197 0 0)
  triclinic box = (0 0 0) to (12.805951 13.862849 31.179185) with tilt (-8.0037197 0 0)
  triclinic box = (0 0 0) to (12.805951 13.862849 31.179185) with tilt (-8.0037197 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60416     -477.72499      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80595146679097 y final 0 13.862849112339372 z final 0 31.17918512055304      xy final -8.003719666744358 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805951 13.862849 31.179185) with tilt (-8.0037197 0 0)
  triclinic box = (0 0 0) to (12.805951 13.862849 31.179185) with tilt (-8.0037197 0 0)
  triclinic box = (0 0 0) to (12.805951 13.862849 31.179185) with tilt (-8.0037197 0 0)
  triclinic box = (0 0 0) to (12.805951 13.862849 31.179185) with tilt (-8.0037197 0 0)
  triclinic box = (0 0 0) to (12.805951 13.862849 31.179185) with tilt (-8.0037197 0 0)
  triclinic box = (0 0 0) to (12.805951 13.862849 31.179185) with tilt (-8.0037197 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60416     -477.72499      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806525406279535 y final 0 13.863470420061033 z final 0 31.180582515007355      xy final -8.004078378924712 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806525 13.862849 31.179185) with tilt (-8.0037197 0 0)
  triclinic box = (0 0 0) to (12.806525 13.86347 31.179185) with tilt (-8.0037197 0 0)
  triclinic box = (0 0 0) to (12.806525 13.86347 31.180583) with tilt (-8.0037197 0 0)
  triclinic box = (0 0 0) to (12.806525 13.86347 31.180583) with tilt (-8.0040784 0 0)
  triclinic box = (0 0 0) to (12.806525 13.86347 31.180583) with tilt (-8.0040784 0 0)
  triclinic box = (0 0 0) to (12.806525 13.86347 31.180583) with tilt (-8.0040784 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60571     -528.00661      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808821421474574 y final 0 13.865955929418993 z final 0 31.186172719139506      xy final -8.005513388421612 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808821 13.86347 31.180583) with tilt (-8.0040784 0 0)
  triclinic box = (0 0 0) to (12.808821 13.865956 31.180583) with tilt (-8.0040784 0 0)
  triclinic box = (0 0 0) to (12.808821 13.865956 31.186173) with tilt (-8.0040784 0 0)
  triclinic box = (0 0 0) to (12.808821 13.865956 31.186173) with tilt (-8.0055134 0 0)
  triclinic box = (0 0 0) to (12.808821 13.865956 31.186173) with tilt (-8.0055134 0 0)
  triclinic box = (0 0 0) to (12.808821 13.865956 31.186173) with tilt (-8.0055134 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60694     -723.13424      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808821421474574 y final 0 13.865955929418993 z final 0 31.186172719139503      xy final -8.005513388421614 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808821 13.865956 31.186173) with tilt (-8.0055134 0 0)
  triclinic box = (0 0 0) to (12.808821 13.865956 31.186173) with tilt (-8.0055134 0 0)
  triclinic box = (0 0 0) to (12.808821 13.865956 31.186173) with tilt (-8.0055134 0 0)
  triclinic box = (0 0 0) to (12.808821 13.865956 31.186173) with tilt (-8.0055134 0 0)
  triclinic box = (0 0 0) to (12.808821 13.865956 31.186173) with tilt (-8.0055134 0 0)
  triclinic box = (0 0 0) to (12.808821 13.865956 31.186173) with tilt (-8.0055134 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60694     -723.13424      0              0            
Loop time of 6.66e-07 on 1 procs for 0 steps with 238 atoms

300.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 6.66e-07   |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806548378383857 y final 0 13.86349528809342 z final 0 31.180638446149615      xy final -8.004092736489914 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806548 13.865956 31.186173) with tilt (-8.0055134 0 0)
  triclinic box = (0 0 0) to (12.806548 13.863495 31.186173) with tilt (-8.0055134 0 0)
  triclinic box = (0 0 0) to (12.806548 13.863495 31.180638) with tilt (-8.0055134 0 0)
  triclinic box = (0 0 0) to (12.806548 13.863495 31.180638) with tilt (-8.0040927 0 0)
  triclinic box = (0 0 0) to (12.806548 13.863495 31.180638) with tilt (-8.0040927 0 0)
  triclinic box = (0 0 0) to (12.806548 13.863495 31.180638) with tilt (-8.0040927 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60785     -540.60137      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806548378383857 y final 0 13.86349528809342 z final 0 31.180638446149615      xy final -8.004092736489914 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806548 13.863495 31.180638) with tilt (-8.0040927 0 0)
  triclinic box = (0 0 0) to (12.806548 13.863495 31.180638) with tilt (-8.0040927 0 0)
  triclinic box = (0 0 0) to (12.806548 13.863495 31.180638) with tilt (-8.0040927 0 0)
  triclinic box = (0 0 0) to (12.806548 13.863495 31.180638) with tilt (-8.0040927 0 0)
  triclinic box = (0 0 0) to (12.806548 13.863495 31.180638) with tilt (-8.0040927 0 0)
  triclinic box = (0 0 0) to (12.806548 13.863495 31.180638) with tilt (-8.0040927 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60785     -540.60137      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806107632756389 y final 0 13.863018166955927 z final 0 31.179565344354284      xy final -8.003817270472748 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806108 13.863495 31.180638) with tilt (-8.0040927 0 0)
  triclinic box = (0 0 0) to (12.806108 13.863018 31.180638) with tilt (-8.0040927 0 0)
  triclinic box = (0 0 0) to (12.806108 13.863018 31.179565) with tilt (-8.0040927 0 0)
  triclinic box = (0 0 0) to (12.806108 13.863018 31.179565) with tilt (-8.0038173 0 0)
  triclinic box = (0 0 0) to (12.806108 13.863018 31.179565) with tilt (-8.0038173 0 0)
  triclinic box = (0 0 0) to (12.806108 13.863018 31.179565) with tilt (-8.0038173 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60809     -505.09143      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.805426774996885 y final 0 13.86228111681081 z final 0 31.177907631518238      xy final -8.003391734373057 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805427 13.863018 31.179565) with tilt (-8.0038173 0 0)
  triclinic box = (0 0 0) to (12.805427 13.862281 31.179565) with tilt (-8.0038173 0 0)
  triclinic box = (0 0 0) to (12.805427 13.862281 31.177908) with tilt (-8.0038173 0 0)
  triclinic box = (0 0 0) to (12.805427 13.862281 31.177908) with tilt (-8.0033917 0 0)
  triclinic box = (0 0 0) to (12.805427 13.862281 31.177908) with tilt (-8.0033917 0 0)
  triclinic box = (0 0 0) to (12.805427 13.862281 31.177908) with tilt (-8.0033917 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60827     -450.07672      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.805426774996885 y final 0 13.86228111681081 z final 0 31.177907631518234      xy final -8.003391734373057 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805427 13.862281 31.177908) with tilt (-8.0033917 0 0)
  triclinic box = (0 0 0) to (12.805427 13.862281 31.177908) with tilt (-8.0033917 0 0)
  triclinic box = (0 0 0) to (12.805427 13.862281 31.177908) with tilt (-8.0033917 0 0)
  triclinic box = (0 0 0) to (12.805427 13.862281 31.177908) with tilt (-8.0033917 0 0)
  triclinic box = (0 0 0) to (12.805427 13.862281 31.177908) with tilt (-8.0033917 0 0)
  triclinic box = (0 0 0) to (12.805427 13.862281 31.177908) with tilt (-8.0033917 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60827     -450.07672      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.805917796487309 y final 0 13.86281266316645 z final 0 31.179103142057834      xy final -8.003698622804572 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805918 13.862281 31.177908) with tilt (-8.0033917 0 0)
  triclinic box = (0 0 0) to (12.805918 13.862813 31.177908) with tilt (-8.0033917 0 0)
  triclinic box = (0 0 0) to (12.805918 13.862813 31.179103) with tilt (-8.0033917 0 0)
  triclinic box = (0 0 0) to (12.805918 13.862813 31.179103) with tilt (-8.0036986 0 0)
  triclinic box = (0 0 0) to (12.805918 13.862813 31.179103) with tilt (-8.0036986 0 0)
  triclinic box = (0 0 0) to (12.805918 13.862813 31.179103) with tilt (-8.0036986 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60843     -489.59399      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15817 ave       15817 max       15817 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15817
Ave neighs/atom = 66.457983
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806587899327543 y final 0 13.863538070769934 z final 0 31.18073466944255      xy final -8.004117437079717 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806588 13.862813 31.179103) with tilt (-8.0036986 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863538 31.179103) with tilt (-8.0036986 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863538 31.180735) with tilt (-8.0036986 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863538 31.180735) with tilt (-8.0041174 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863538 31.180735) with tilt (-8.0041174 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863538 31.180735) with tilt (-8.0041174 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60848     -543.18592      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15817 ave       15817 max       15817 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15817
Ave neighs/atom = 66.457983
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806587899327543 y final 0 13.863538070769934 z final 0 31.18073466944255      xy final -8.004117437079717 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806588 13.863538 31.180735) with tilt (-8.0041174 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863538 31.180735) with tilt (-8.0041174 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863538 31.180735) with tilt (-8.0041174 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863538 31.180735) with tilt (-8.0041174 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863538 31.180735) with tilt (-8.0041174 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863538 31.180735) with tilt (-8.0041174 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60848     -543.18592      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15817 ave       15817 max       15817 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15817
Ave neighs/atom = 66.457983
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806532727264784 y final 0 13.863478345260024 z final 0 31.180600339716044      xy final -8.004082954540495 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806533 13.863538 31.180735) with tilt (-8.0041174 0 0)
  triclinic box = (0 0 0) to (12.806533 13.863478 31.180735) with tilt (-8.0041174 0 0)
  triclinic box = (0 0 0) to (12.806533 13.863478 31.1806) with tilt (-8.0041174 0 0)
  triclinic box = (0 0 0) to (12.806533 13.863478 31.1806) with tilt (-8.004083 0 0)
  triclinic box = (0 0 0) to (12.806533 13.863478 31.1806) with tilt (-8.004083 0 0)
  triclinic box = (0 0 0) to (12.806533 13.863478 31.1806) with tilt (-8.004083 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60856     -538.47533      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15817 ave       15817 max       15817 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15817
Ave neighs/atom = 66.457983
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806312041390614 y final 0 13.863239445793413 z final 0 31.180063026597054      xy final -8.003945025869138 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806312 13.863478 31.1806) with tilt (-8.004083 0 0)
  triclinic box = (0 0 0) to (12.806312 13.863239 31.1806) with tilt (-8.004083 0 0)
  triclinic box = (0 0 0) to (12.806312 13.863239 31.180063) with tilt (-8.004083 0 0)
  triclinic box = (0 0 0) to (12.806312 13.863239 31.180063) with tilt (-8.003945 0 0)
  triclinic box = (0 0 0) to (12.806312 13.863239 31.180063) with tilt (-8.003945 0 0)
  triclinic box = (0 0 0) to (12.806312 13.863239 31.180063) with tilt (-8.003945 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.60871     -519.58976      0              0            
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
Nghost:           2030 ave        2030 max        2030 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15817 ave       15817 max       15817 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15817
Ave neighs/atom = 66.457983
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.810913893863672 y final 0 13.8682210972262 z final 0 31.19126734909719      xy final -8.006821183664796 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.810914 13.863239 31.180063) with tilt (-8.003945 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.180063) with tilt (-8.003945 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.003945 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.57442     -739.59513      0              0            
Loop time of 7.09e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.862141969617172 y final 0 12.841145361232558 z final 0 28.881252701612517      xy final -7.413838731010733 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.862142 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (11.862142 12.841145 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (11.862142 12.841145 28.881253) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (11.862142 12.841145 28.881253) with tilt (-7.4138387 0 0)
  triclinic box = (0 0 0) to (11.862142 12.841145 28.881253) with tilt (-7.4138387 0 0)
  triclinic box = (0 0 0) to (11.862142 12.841145 28.881253) with tilt (-7.4138387 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-326.78815      145078.52      0              0            
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
Nghost:           2254 ave        2254 max        2254 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          18645 ave       18645 max       18645 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 18645
Ave neighs/atom = 78.340336
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808230308546845 y final 0 13.865316030904207 z final 0 31.18473351179528      xy final -8.005143942841778 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.80823 12.841145 28.881253) with tilt (-7.4138387 0 0)
  triclinic box = (0 0 0) to (12.80823 13.865316 28.881253) with tilt (-7.4138387 0 0)
  triclinic box = (0 0 0) to (12.80823 13.865316 31.184734) with tilt (-7.4138387 0 0)
  triclinic box = (0 0 0) to (12.80823 13.865316 31.184734) with tilt (-8.0051439 0 0)
  triclinic box = (0 0 0) to (12.80823 13.865316 31.184734) with tilt (-8.0051439 0 0)
  triclinic box = (0 0 0) to (12.80823 13.865316 31.184734) with tilt (-8.0051439 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.57468     -524.1881       0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808230308546845 y final 0 13.865316030904207 z final 0 31.184733511795276      xy final -8.005143942841778 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.80823 13.865316 31.184734) with tilt (-8.0051439 0 0)
  triclinic box = (0 0 0) to (12.80823 13.865316 31.184734) with tilt (-8.0051439 0 0)
  triclinic box = (0 0 0) to (12.80823 13.865316 31.184734) with tilt (-8.0051439 0 0)
  triclinic box = (0 0 0) to (12.80823 13.865316 31.184734) with tilt (-8.0051439 0 0)
  triclinic box = (0 0 0) to (12.80823 13.865316 31.184734) with tilt (-8.0051439 0 0)
  triclinic box = (0 0 0) to (12.80823 13.865316 31.184734) with tilt (-8.0051439 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.57468     -524.1881       0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.807684607620367 y final 0 13.864725292322708 z final 0 31.18340487094793      xy final -8.00480287976273 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.807685 13.865316 31.184734) with tilt (-8.0051439 0 0)
  triclinic box = (0 0 0) to (12.807685 13.864725 31.184734) with tilt (-8.0051439 0 0)
  triclinic box = (0 0 0) to (12.807685 13.864725 31.183405) with tilt (-8.0051439 0 0)
  triclinic box = (0 0 0) to (12.807685 13.864725 31.183405) with tilt (-8.0048029 0 0)
  triclinic box = (0 0 0) to (12.807685 13.864725 31.183405) with tilt (-8.0048029 0 0)
  triclinic box = (0 0 0) to (12.807685 13.864725 31.183405) with tilt (-8.0048029 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.58218     -513.5146       0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806223800521417 y final 0 13.863143922250558 z final 0 31.17984818286554      xy final -8.003889875325886 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806224 13.864725 31.183405) with tilt (-8.0048029 0 0)
  triclinic box = (0 0 0) to (12.806224 13.863144 31.183405) with tilt (-8.0048029 0 0)
  triclinic box = (0 0 0) to (12.806224 13.863144 31.179848) with tilt (-8.0048029 0 0)
  triclinic box = (0 0 0) to (12.806224 13.863144 31.179848) with tilt (-8.0038899 0 0)
  triclinic box = (0 0 0) to (12.806224 13.863144 31.179848) with tilt (-8.0038899 0 0)
  triclinic box = (0 0 0) to (12.806224 13.863144 31.179848) with tilt (-8.0038899 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59087     -470.22323      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806223800521417 y final 0 13.863143922250558 z final 0 31.17984818286554      xy final -8.003889875325887 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806224 13.863144 31.179848) with tilt (-8.0038899 0 0)
  triclinic box = (0 0 0) to (12.806224 13.863144 31.179848) with tilt (-8.0038899 0 0)
  triclinic box = (0 0 0) to (12.806224 13.863144 31.179848) with tilt (-8.0038899 0 0)
  triclinic box = (0 0 0) to (12.806224 13.863144 31.179848) with tilt (-8.0038899 0 0)
  triclinic box = (0 0 0) to (12.806224 13.863144 31.179848) with tilt (-8.0038899 0 0)
  triclinic box = (0 0 0) to (12.806224 13.863144 31.179848) with tilt (-8.0038899 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59087     -470.22323      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806764796317955 y final 0 13.863729567379485 z final 0 31.18116536793635      xy final -8.004227997698724 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806765 13.863144 31.179848) with tilt (-8.0038899 0 0)
  triclinic box = (0 0 0) to (12.806765 13.86373 31.179848) with tilt (-8.0038899 0 0)
  triclinic box = (0 0 0) to (12.806765 13.86373 31.181165) with tilt (-8.0038899 0 0)
  triclinic box = (0 0 0) to (12.806765 13.86373 31.181165) with tilt (-8.004228 0 0)
  triclinic box = (0 0 0) to (12.806765 13.86373 31.181165) with tilt (-8.004228 0 0)
  triclinic box = (0 0 0) to (12.806765 13.86373 31.181165) with tilt (-8.004228 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59234     -519.20306      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808200184466719 y final 0 13.865283420630883 z final 0 31.184660167438437      xy final -8.0051251152917 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.8082 13.86373 31.181165) with tilt (-8.004228 0 0)
  triclinic box = (0 0 0) to (12.8082 13.865283 31.181165) with tilt (-8.004228 0 0)
  triclinic box = (0 0 0) to (12.8082 13.865283 31.18466) with tilt (-8.004228 0 0)
  triclinic box = (0 0 0) to (12.8082 13.865283 31.18466) with tilt (-8.0051251 0 0)
  triclinic box = (0 0 0) to (12.8082 13.865283 31.18466) with tilt (-8.0051251 0 0)
  triclinic box = (0 0 0) to (12.8082 13.865283 31.18466) with tilt (-8.0051251 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59388     -644.96493      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808200184466719 y final 0 13.865283420630883 z final 0 31.184660167438437      xy final -8.0051251152917 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.8082 13.865283 31.18466) with tilt (-8.0051251 0 0)
  triclinic box = (0 0 0) to (12.8082 13.865283 31.18466) with tilt (-8.0051251 0 0)
  triclinic box = (0 0 0) to (12.8082 13.865283 31.18466) with tilt (-8.0051251 0 0)
  triclinic box = (0 0 0) to (12.8082 13.865283 31.18466) with tilt (-8.0051251 0 0)
  triclinic box = (0 0 0) to (12.8082 13.865283 31.18466) with tilt (-8.0051251 0 0)
  triclinic box = (0 0 0) to (12.8082 13.865283 31.18466) with tilt (-8.0051251 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59388     -644.96493      0              0            
Loop time of 1.084e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806766745399011 y final 0 13.863731677321619 z final 0 31.181170113445408      xy final -8.004229215874382 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806767 13.865283 31.18466) with tilt (-8.0051251 0 0)
  triclinic box = (0 0 0) to (12.806767 13.863732 31.18466) with tilt (-8.0051251 0 0)
  triclinic box = (0 0 0) to (12.806767 13.863732 31.18117) with tilt (-8.0051251 0 0)
  triclinic box = (0 0 0) to (12.806767 13.863732 31.18117) with tilt (-8.0042292 0 0)
  triclinic box = (0 0 0) to (12.806767 13.863732 31.18117) with tilt (-8.0042292 0 0)
  triclinic box = (0 0 0) to (12.806767 13.863732 31.18117) with tilt (-8.0042292 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59434     -534.53577      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.805517200494174 y final 0 13.86237900528317 z final 0 31.17812779425455      xy final -8.003448250308859 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805517 13.863732 31.18117) with tilt (-8.0042292 0 0)
  triclinic box = (0 0 0) to (12.805517 13.862379 31.18117) with tilt (-8.0042292 0 0)
  triclinic box = (0 0 0) to (12.805517 13.862379 31.178128) with tilt (-8.0042292 0 0)
  triclinic box = (0 0 0) to (12.805517 13.862379 31.178128) with tilt (-8.0034483 0 0)
  triclinic box = (0 0 0) to (12.805517 13.862379 31.178128) with tilt (-8.0034483 0 0)
  triclinic box = (0 0 0) to (12.805517 13.862379 31.178128) with tilt (-8.0034483 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59454     -438.19687      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.805517200494174 y final 0 13.86237900528317 z final 0 31.178127794254547      xy final -8.003448250308859 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805517 13.862379 31.178128) with tilt (-8.0034483 0 0)
  triclinic box = (0 0 0) to (12.805517 13.862379 31.178128) with tilt (-8.0034483 0 0)
  triclinic box = (0 0 0) to (12.805517 13.862379 31.178128) with tilt (-8.0034483 0 0)
  triclinic box = (0 0 0) to (12.805517 13.862379 31.178128) with tilt (-8.0034483 0 0)
  triclinic box = (0 0 0) to (12.805517 13.862379 31.178128) with tilt (-8.0034483 0 0)
  triclinic box = (0 0 0) to (12.805517 13.862379 31.178128) with tilt (-8.0034483 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59454     -438.19687      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80590807143104 y final 0 13.86280213548433 z final 0 31.179079464056702      xy final -8.0036925446444 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805908 13.862379 31.178128) with tilt (-8.0034483 0 0)
  triclinic box = (0 0 0) to (12.805908 13.862802 31.178128) with tilt (-8.0034483 0 0)
  triclinic box = (0 0 0) to (12.805908 13.862802 31.179079) with tilt (-8.0034483 0 0)
  triclinic box = (0 0 0) to (12.805908 13.862802 31.179079) with tilt (-8.0036925 0 0)
  triclinic box = (0 0 0) to (12.805908 13.862802 31.179079) with tilt (-8.0036925 0 0)
  triclinic box = (0 0 0) to (12.805908 13.862802 31.179079) with tilt (-8.0036925 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59469     -468.09936      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.807471674490165 y final 0 13.864494785447624 z final 0 31.182886433758405      xy final -8.004669796556353 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.807472 13.862802 31.179079) with tilt (-8.0036925 0 0)
  triclinic box = (0 0 0) to (12.807472 13.864495 31.179079) with tilt (-8.0036925 0 0)
  triclinic box = (0 0 0) to (12.807472 13.864495 31.182886) with tilt (-8.0036925 0 0)
  triclinic box = (0 0 0) to (12.807472 13.864495 31.182886) with tilt (-8.0046698 0 0)
  triclinic box = (0 0 0) to (12.807472 13.864495 31.182886) with tilt (-8.0046698 0 0)
  triclinic box = (0 0 0) to (12.807472 13.864495 31.182886) with tilt (-8.0046698 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59463     -587.39652      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806588302548734 y final 0 13.863538507269787 z final 0 31.180735651181756      xy final -8.00411768909296 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806588 13.864495 31.182886) with tilt (-8.0046698 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863539 31.182886) with tilt (-8.0046698 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863539 31.180736) with tilt (-8.0046698 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863539 31.180736) with tilt (-8.0041177 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863539 31.180736) with tilt (-8.0041177 0 0)
  triclinic box = (0 0 0) to (12.806588 13.863539 31.180736) with tilt (-8.0041177 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59478     -520.02582      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.810913893863672 y final 0 13.8682210972262 z final 0 31.19126734909719      xy final -8.006821183664796 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.810914 13.863539 31.180736) with tilt (-8.0041177 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.180736) with tilt (-8.0041177 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0041177 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (12.810914 13.868221 31.191267) with tilt (-8.0068212 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.57299     -738.45727      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.877800885035525 y final 0 12.858096634417565 z final 0 28.919378117274288      xy final -7.423625553147203 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.877801 13.868221 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (11.877801 12.858097 31.191267) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (11.877801 12.858097 28.919378) with tilt (-8.0068212 0 0)
  triclinic box = (0 0 0) to (11.877801 12.858097 28.919378) with tilt (-7.4236256 0 0)
  triclinic box = (0 0 0) to (11.877801 12.858097 28.919378) with tilt (-7.4236256 0 0)
  triclinic box = (0 0 0) to (11.877801 12.858097 28.919378) with tilt (-7.4236256 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-328.36942      140978.32      0              0            
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
Nghost:           2250 ave        2250 max        2250 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          18645 ave       18645 max       18645 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 18645
Ave neighs/atom = 78.340336
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808269040806572 y final 0 13.865357959805293 z final 0 31.184827814846425      xy final -8.005168150504108 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808269 12.858097 28.919378) with tilt (-7.4236256 0 0)
  triclinic box = (0 0 0) to (12.808269 13.865358 28.919378) with tilt (-7.4236256 0 0)
  triclinic box = (0 0 0) to (12.808269 13.865358 31.184828) with tilt (-7.4236256 0 0)
  triclinic box = (0 0 0) to (12.808269 13.865358 31.184828) with tilt (-8.0051682 0 0)
  triclinic box = (0 0 0) to (12.808269 13.865358 31.184828) with tilt (-8.0051682 0 0)
  triclinic box = (0 0 0) to (12.808269 13.865358 31.184828) with tilt (-8.0051682 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.57325     -526.1751       0              0            
Loop time of 1.5e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808269040806572 y final 0 13.86535795980529 z final 0 31.18482781484642      xy final -8.005168150504108 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808269 13.865358 31.184828) with tilt (-8.0051682 0 0)
  triclinic box = (0 0 0) to (12.808269 13.865358 31.184828) with tilt (-8.0051682 0 0)
  triclinic box = (0 0 0) to (12.808269 13.865358 31.184828) with tilt (-8.0051682 0 0)
  triclinic box = (0 0 0) to (12.808269 13.865358 31.184828) with tilt (-8.0051682 0 0)
  triclinic box = (0 0 0) to (12.808269 13.865358 31.184828) with tilt (-8.0051682 0 0)
  triclinic box = (0 0 0) to (12.808269 13.865358 31.184828) with tilt (-8.0051682 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.57325     -526.1751       0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.807725196710441 y final 0 13.86476923130158 z final 0 31.183503694901507      xy final -8.004828247944026 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.807725 13.865358 31.184828) with tilt (-8.0051682 0 0)
  triclinic box = (0 0 0) to (12.807725 13.864769 31.184828) with tilt (-8.0051682 0 0)
  triclinic box = (0 0 0) to (12.807725 13.864769 31.183504) with tilt (-8.0051682 0 0)
  triclinic box = (0 0 0) to (12.807725 13.864769 31.183504) with tilt (-8.0048282 0 0)
  triclinic box = (0 0 0) to (12.807725 13.864769 31.183504) with tilt (-8.0048282 0 0)
  triclinic box = (0 0 0) to (12.807725 13.864769 31.183504) with tilt (-8.0048282 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.58078     -516.86039      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806342163422903 y final 0 13.8632720538499 z final 0 31.180136365967467      xy final -8.003963852139314 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806342 13.864769 31.183504) with tilt (-8.0048282 0 0)
  triclinic box = (0 0 0) to (12.806342 13.863272 31.183504) with tilt (-8.0048282 0 0)
  triclinic box = (0 0 0) to (12.806342 13.863272 31.180136) with tilt (-8.0048282 0 0)
  triclinic box = (0 0 0) to (12.806342 13.863272 31.180136) with tilt (-8.0039639 0 0)
  triclinic box = (0 0 0) to (12.806342 13.863272 31.180136) with tilt (-8.0039639 0 0)
  triclinic box = (0 0 0) to (12.806342 13.863272 31.180136) with tilt (-8.0039639 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.58904     -481.97381      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806342163422903 y final 0 13.8632720538499 z final 0 31.180136365967464      xy final -8.003963852139314 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806342 13.863272 31.180136) with tilt (-8.0039639 0 0)
  triclinic box = (0 0 0) to (12.806342 13.863272 31.180136) with tilt (-8.0039639 0 0)
  triclinic box = (0 0 0) to (12.806342 13.863272 31.180136) with tilt (-8.0039639 0 0)
  triclinic box = (0 0 0) to (12.806342 13.863272 31.180136) with tilt (-8.0039639 0 0)
  triclinic box = (0 0 0) to (12.806342 13.863272 31.180136) with tilt (-8.0039639 0 0)
  triclinic box = (0 0 0) to (12.806342 13.863272 31.180136) with tilt (-8.0039639 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.58904     -481.97381      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806836058475836 y final 0 13.8638067109282 z final 0 31.18133887288976      xy final -8.004272536547399 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806836 13.863272 31.180136) with tilt (-8.0039639 0 0)
  triclinic box = (0 0 0) to (12.806836 13.863807 31.180136) with tilt (-8.0039639 0 0)
  triclinic box = (0 0 0) to (12.806836 13.863807 31.181339) with tilt (-8.0039639 0 0)
  triclinic box = (0 0 0) to (12.806836 13.863807 31.181339) with tilt (-8.0042725 0 0)
  triclinic box = (0 0 0) to (12.806836 13.863807 31.181339) with tilt (-8.0042725 0 0)
  triclinic box = (0 0 0) to (12.806836 13.863807 31.181339) with tilt (-8.0042725 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59047     -525.58547      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808811829172665 y final 0 13.86594554544757 z final 0 31.186149364360976      xy final -8.005507393232916 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808812 13.863807 31.181339) with tilt (-8.0042725 0 0)
  triclinic box = (0 0 0) to (12.808812 13.865946 31.181339) with tilt (-8.0042725 0 0)
  triclinic box = (0 0 0) to (12.808812 13.865946 31.186149) with tilt (-8.0042725 0 0)
  triclinic box = (0 0 0) to (12.808812 13.865946 31.186149) with tilt (-8.0055074 0 0)
  triclinic box = (0 0 0) to (12.808812 13.865946 31.186149) with tilt (-8.0055074 0 0)
  triclinic box = (0 0 0) to (12.808812 13.865946 31.186149) with tilt (-8.0055074 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59158     -694.38947      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.808811829172665 y final 0 13.86594554544757 z final 0 31.186149364360976      xy final -8.005507393232916 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.808812 13.865946 31.186149) with tilt (-8.0055074 0 0)
  triclinic box = (0 0 0) to (12.808812 13.865946 31.186149) with tilt (-8.0055074 0 0)
  triclinic box = (0 0 0) to (12.808812 13.865946 31.186149) with tilt (-8.0055074 0 0)
  triclinic box = (0 0 0) to (12.808812 13.865946 31.186149) with tilt (-8.0055074 0 0)
  triclinic box = (0 0 0) to (12.808812 13.865946 31.186149) with tilt (-8.0055074 0 0)
  triclinic box = (0 0 0) to (12.808812 13.865946 31.186149) with tilt (-8.0055074 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59158     -694.38947      0              0            
Loop time of 1.459e-06 on 1 procs for 0 steps with 238 atoms

342.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.459e-06  |            |       |100.00

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806843695906679 y final 0 13.8638149786896 z final 0 31.18135746806173      xy final -8.004277309941676 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806844 13.865946 31.186149) with tilt (-8.0055074 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.186149) with tilt (-8.0055074 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181357) with tilt (-8.0055074 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181357) with tilt (-8.0042773 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181357) with tilt (-8.0042773 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181357) with tilt (-8.0042773 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59237     -535.36739      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806843695906679 y final 0 13.8638149786896 z final 0 31.18135746806173      xy final -8.004277309941678 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181357) with tilt (-8.0042773 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181357) with tilt (-8.0042773 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181357) with tilt (-8.0042773 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181357) with tilt (-8.0042773 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181357) with tilt (-8.0042773 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181357) with tilt (-8.0042773 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59237     -535.36739      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806455047554579 y final 0 13.863394254507027 z final 0 31.180411209679573      xy final -8.004034404721615 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806455 13.863815 31.181357) with tilt (-8.0042773 0 0)
  triclinic box = (0 0 0) to (12.806455 13.863394 31.181357) with tilt (-8.0042773 0 0)
  triclinic box = (0 0 0) to (12.806455 13.863394 31.180411) with tilt (-8.0042773 0 0)
  triclinic box = (0 0 0) to (12.806455 13.863394 31.180411) with tilt (-8.0040344 0 0)
  triclinic box = (0 0 0) to (12.806455 13.863394 31.180411) with tilt (-8.0040344 0 0)
  triclinic box = (0 0 0) to (12.806455 13.863394 31.180411) with tilt (-8.0040344 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59258     -503.45542      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80579153171306 y final 0 13.862675977538814 z final 0 31.178795719935856      xy final -8.003619707320667 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805792 13.863394 31.180411) with tilt (-8.0040344 0 0)
  triclinic box = (0 0 0) to (12.805792 13.862676 31.180411) with tilt (-8.0040344 0 0)
  triclinic box = (0 0 0) to (12.805792 13.862676 31.178796) with tilt (-8.0040344 0 0)
  triclinic box = (0 0 0) to (12.805792 13.862676 31.178796) with tilt (-8.0036197 0 0)
  triclinic box = (0 0 0) to (12.805792 13.862676 31.178796) with tilt (-8.0036197 0 0)
  triclinic box = (0 0 0) to (12.805792 13.862676 31.178796) with tilt (-8.0036197 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59275     -448.75556      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80579153171306 y final 0 13.862675977538814 z final 0 31.178795719935856      xy final -8.003619707320667 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.805792 13.862676 31.178796) with tilt (-8.0036197 0 0)
  triclinic box = (0 0 0) to (12.805792 13.862676 31.178796) with tilt (-8.0036197 0 0)
  triclinic box = (0 0 0) to (12.805792 13.862676 31.178796) with tilt (-8.0036197 0 0)
  triclinic box = (0 0 0) to (12.805792 13.862676 31.178796) with tilt (-8.0036197 0 0)
  triclinic box = (0 0 0) to (12.805792 13.862676 31.178796) with tilt (-8.0036197 0 0)
  triclinic box = (0 0 0) to (12.805792 13.862676 31.178796) with tilt (-8.0036197 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59275     -448.75556      0              0            
Loop time of 8.75e-07 on 1 procs for 0 steps with 238 atoms

571.4% CPU use with 1 MPI tasks x no OpenMP threads

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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806243108462292 y final 0 13.863164823709551 z final 0 31.179895192715904      xy final -8.003901942788936 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806243 13.862676 31.178796) with tilt (-8.0036197 0 0)
  triclinic box = (0 0 0) to (12.806243 13.863165 31.178796) with tilt (-8.0036197 0 0)
  triclinic box = (0 0 0) to (12.806243 13.863165 31.179895) with tilt (-8.0036197 0 0)
  triclinic box = (0 0 0) to (12.806243 13.863165 31.179895) with tilt (-8.0039019 0 0)
  triclinic box = (0 0 0) to (12.806243 13.863165 31.179895) with tilt (-8.0039019 0 0)
  triclinic box = (0 0 0) to (12.806243 13.863165 31.179895) with tilt (-8.0039019 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59295     -484.87293      0              0            
Loop time of 1.708e-06 on 1 procs for 0 steps with 238 atoms

292.7% CPU use with 1 MPI tasks x no OpenMP threads

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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806903081885844 y final 0 13.863879265897824 z final 0 31.181502057586577      xy final -8.004314426178656 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806903 13.863165 31.179895) with tilt (-8.0039019 0 0)
  triclinic box = (0 0 0) to (12.806903 13.863879 31.179895) with tilt (-8.0039019 0 0)
  triclinic box = (0 0 0) to (12.806903 13.863879 31.181502) with tilt (-8.0039019 0 0)
  triclinic box = (0 0 0) to (12.806903 13.863879 31.181502) with tilt (-8.0043144 0 0)
  triclinic box = (0 0 0) to (12.806903 13.863879 31.181502) with tilt (-8.0043144 0 0)
  triclinic box = (0 0 0) to (12.806903 13.863879 31.181502) with tilt (-8.0043144 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59308     -537.43996      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.806903081885844 y final 0 13.863879265897822 z final 0 31.181502057586577      xy final -8.004314426178656 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806903 13.863879 31.181502) with tilt (-8.0043144 0 0)
  triclinic box = (0 0 0) to (12.806903 13.863879 31.181502) with tilt (-8.0043144 0 0)
  triclinic box = (0 0 0) to (12.806903 13.863879 31.181502) with tilt (-8.0043144 0 0)
  triclinic box = (0 0 0) to (12.806903 13.863879 31.181502) with tilt (-8.0043144 0 0)
  triclinic box = (0 0 0) to (12.806903 13.863879 31.181502) with tilt (-8.0043144 0 0)
  triclinic box = (0 0 0) to (12.806903 13.863879 31.181502) with tilt (-8.0043144 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59308     -537.43996      0              0            
Loop time of 1.458e-06 on 1 procs for 0 steps with 238 atoms

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

Nlocal:            238 ave         238 max         238 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80684413404451 y final 0 13.863815452987707 z final 0 31.181358534814205      xy final -8.004277583777823 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806844 13.863879 31.181502) with tilt (-8.0043144 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181502) with tilt (-8.0043144 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181359) with tilt (-8.0043144 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181359) with tilt (-8.0042776 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181359) with tilt (-8.0042776 0 0)
  triclinic box = (0 0 0) to (12.806844 13.863815 31.181359) with tilt (-8.0042776 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59316     -532.23144      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.80660834539243 y final 0 13.863560204284424 z final 0 31.180784450330783      xy final -8.004130215870273 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.806608 13.863815 31.181359) with tilt (-8.0042776 0 0)
  triclinic box = (0 0 0) to (12.806608 13.86356 31.181359) with tilt (-8.0042776 0 0)
  triclinic box = (0 0 0) to (12.806608 13.86356 31.180784) with tilt (-8.0042776 0 0)
  triclinic box = (0 0 0) to (12.806608 13.86356 31.180784) with tilt (-8.0041302 0 0)
  triclinic box = (0 0 0) to (12.806608 13.86356 31.180784) with tilt (-8.0041302 0 0)
  triclinic box = (0 0 0) to (12.806608 13.86356 31.180784) with tilt (-8.0041302 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.258 | 3.258 | 3.258 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.59336     -511.08227      0              0            
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
Nghost:           2034 ave        2034 max        2034 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:          15813 ave       15813 max       15813 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 15813
Ave neighs/atom = 66.441176
Neighbor list builds = 0
Dangerous builds = 0
change_box all boundary p p f
Changing box ...
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 11.094576877780963 y final 0 10.397089116365729 z final 0 34.41227813059663      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.094577 13.86356 31.180784) with tilt (-8.0041302 0 0)
  triclinic box = (0 0 0) to (11.094577 10.397089 31.180784) with tilt (-8.0041302 0 0)
  triclinic box = (0 0 0) to (11.094577 10.397089 34.412278) with tilt (-8.0041302 0 0)
  triclinic box = (0 0 0) to (11.094577 10.397089 34.412278) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.094577 10.397089 34.412278) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.094577 10.397089 34.412278) with tilt (0 0 0)
echo none
158 atoms in group delatoms
Deleted 158 atoms, new total = 80
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.47836     -3309.9112      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4224 ave        4224 max        4224 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4224
Ave neighs/atom = 52.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 9.083471282227173 y final 0 8.512416602061002 z final 0 28.174390388992347      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.0834713 10.397089 34.412278) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0834713 8.5124166 34.412278) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0834713 8.5124166 28.17439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0834713 8.5124166 28.17439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0834713 8.5124166 28.17439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.0834713 8.5124166 28.17439) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.266 | 3.266 | 3.266 Mbytes
    PotEng          Pxx           E_mol          E_coul    
 17.667475      462411.28      0              0            
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
Nghost:            735 ave         735 max         735 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           7584 ave        7584 max        7584 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 7584
Ave neighs/atom = 94.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.030797110563332 y final 0 10.33731902049927 z final 0 34.21445110999224      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.030797 8.5124166 28.17439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.030797 10.337319 28.17439) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.030797 10.337319 34.214451) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.030797 10.337319 34.214451) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.030797 10.337319 34.214451) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.030797 10.337319 34.214451) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.53154     -477.73275      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4224 ave        4224 max        4224 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4224
Ave neighs/atom = 52.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.030797110563334 y final 0 10.33731902049927 z final 0 34.21445110999224      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.030797 10.337319 34.214451) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.030797 10.337319 34.214451) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.030797 10.337319 34.214451) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.030797 10.337319 34.214451) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.030797 10.337319 34.214451) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.030797 10.337319 34.214451) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.53154     -477.73275      0              0            
Loop time of 5.84e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4224 ave        4224 max        4224 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4224
Ave neighs/atom = 52.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.038920239400065 y final 0 10.344931468936881 z final 0 34.23964678639339      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.03892 10.337319 34.214451) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.03892 10.344931 34.214451) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.03892 10.344931 34.239647) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.03892 10.344931 34.239647) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.03892 10.344931 34.239647) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.03892 10.344931 34.239647) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.54755     -719.3287       0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4224 ave        4224 max        4224 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4224
Ave neighs/atom = 52.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.055038107877719 y final 0 10.360036048117694 z final 0 34.289639911776675      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.055038 10.344931 34.239647) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.055038 10.360036 34.239647) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.055038 10.360036 34.28964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.055038 10.360036 34.28964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.055038 10.360036 34.28964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.055038 10.360036 34.28964) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.55994     -1169.2079      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4224 ave        4224 max        4224 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4224
Ave neighs/atom = 52.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.055038107877719 y final 0 10.360036048117694 z final 0 34.289639911776675      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.055038 10.360036 34.28964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.055038 10.360036 34.28964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.055038 10.360036 34.28964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.055038 10.360036 34.28964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.055038 10.360036 34.28964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.055038 10.360036 34.28964) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.55994     -1169.2079      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4224 ave        4224 max        4224 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4224
Ave neighs/atom = 52.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.047382888866819 y final 0 10.352862093208154 z final 0 34.265895560941516      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.047383 10.360036 34.28964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.047383 10.352862 34.28964) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.047383 10.352862 34.265896) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.047383 10.352862 34.265896) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.047383 10.352862 34.265896) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.047383 10.352862 34.265896) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56311     -778.08728      0              0            
Loop time of 1.708e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4224 ave        4224 max        4224 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4224
Ave neighs/atom = 52.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.044833091161498 y final 0 10.350472594783566 z final 0 34.25798680076266      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.044833 10.352862 34.265896) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.044833 10.350473 34.265896) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.044833 10.350473 34.257987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.044833 10.350473 34.257987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.044833 10.350473 34.257987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.044833 10.350473 34.257987) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56453     -652.04645      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4224 ave        4224 max        4224 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4224
Ave neighs/atom = 52.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.044833091161498 y final 0 10.350472594783566 z final 0 34.25798680076266      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.044833 10.350473 34.257987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.044833 10.350473 34.257987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.044833 10.350473 34.257987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.044833 10.350473 34.257987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.044833 10.350473 34.257987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.044833 10.350473 34.257987) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56453     -652.04645      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4224 ave        4224 max        4224 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4224
Ave neighs/atom = 52.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.045210504484675 y final 0 10.350826281093374 z final 0 34.25915743145821      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.045211 10.350473 34.257987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045211 10.350826 34.257987) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045211 10.350826 34.259157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045211 10.350826 34.259157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045211 10.350826 34.259157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045211 10.350826 34.259157) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.5651      -682.01137      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4256 ave        4256 max        4256 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4256
Ave neighs/atom = 53.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.045210504484675 y final 0 10.350826281093374 z final 0 34.25915743145821      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.045211 10.350826 34.259157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045211 10.350826 34.259157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045211 10.350826 34.259157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045211 10.350826 34.259157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045211 10.350826 34.259157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045211 10.350826 34.259157) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.5651      -682.01137      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4256 ave        4256 max        4256 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4256
Ave neighs/atom = 53.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.046036730793624 y final 0 10.351600564660838 z final 0 34.26172015465672      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.046037 10.350826 34.259157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046037 10.351601 34.259157) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046037 10.351601 34.26172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046037 10.351601 34.26172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046037 10.351601 34.26172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046037 10.351601 34.26172) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56515     -744.72335      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4256 ave        4256 max        4256 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4256
Ave neighs/atom = 53.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.04934225412627 y final 0 10.354698278169367 z final 0 34.27197296461451      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.049342 10.351601 34.26172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.049342 10.354698 34.26172) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.049342 10.354698 34.271973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.049342 10.354698 34.271973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.049342 10.354698 34.271973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.049342 10.354698 34.271973) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56424     -996.22849      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4256 ave        4256 max        4256 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4256
Ave neighs/atom = 53.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.046703832797261 y final 0 10.35222572766236 z final 0 34.26378931871259      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.046704 10.354698 34.271973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046704 10.352226 34.271973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046704 10.352226 34.263789) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046704 10.352226 34.263789) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046704 10.352226 34.263789) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046704 10.352226 34.263789) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56519     -795.60581      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4256 ave        4256 max        4256 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4256
Ave neighs/atom = 53.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.046898846843275 y final 0 10.35240848167251 z final 0 34.26439419780485      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.046899 10.352226 34.263789) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046899 10.352408 34.263789) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046899 10.352408 34.264394) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046899 10.352408 34.264394) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046899 10.352408 34.264394) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046899 10.352408 34.264394) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56519     -810.59344      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4256 ave        4256 max        4256 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4256
Ave neighs/atom = 53.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.046897216771217 y final 0 10.352406954078926 z final 0 34.26438914177669      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.046897 10.352408 34.264394) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264394) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56519     -810.46822      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4256 ave        4256 max        4256 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4256
Ave neighs/atom = 53.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.046897216770365 y final 0 10.352406954078127 z final 0 34.264389141774046      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56519     -810.46822      0              0            
Loop time of 8.34e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4256 ave        4256 max        4256 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4256
Ave neighs/atom = 53.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.046897216771217 y final 0 10.352406954078926 z final 0 34.26438914177669      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56519     -810.46822      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4256 ave        4256 max        4256 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4256
Ave neighs/atom = 53.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.04689721677122 y final 0 10.352406954078926 z final 0 34.26438914177669      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046897 10.352407 34.264389) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56519     -810.46822      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4256 ave        4256 max        4256 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4256
Ave neighs/atom = 53.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.046517996391916 y final 0 10.352051574318004 z final 0 34.263212906097834      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.046518 10.352407 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046518 10.352052 34.264389) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046518 10.352052 34.263213) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046518 10.352052 34.263213) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046518 10.352052 34.263213) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.046518 10.352052 34.263213) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56547     -821.34701      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4256 ave        4256 max        4256 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4256
Ave neighs/atom = 53.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.045001245049864 y final 0 10.350630177265694 z final 0 34.25850836714941      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.045001 10.352052 34.263213) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045001 10.35063 34.263213) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045001 10.35063 34.258508) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045001 10.35063 34.258508) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045001 10.35063 34.258508) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045001 10.35063 34.258508) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56621     -864.20667      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4224 ave        4224 max        4224 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4224
Ave neighs/atom = 52.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.045001245049864 y final 0 10.350630177265694 z final 0 34.25850836714941      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.045001 10.35063 34.258508) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045001 10.35063 34.258508) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045001 10.35063 34.258508) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045001 10.35063 34.258508) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045001 10.35063 34.258508) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.045001 10.35063 34.258508) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56621     -864.20667      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4224 ave        4224 max        4224 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4224
Ave neighs/atom = 52.8
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.041252585301748 y final 0 10.347117186198304 z final 0 34.246881071823914      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.041253 10.35063 34.258508) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.041253 10.347117 34.258508) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.041253 10.347117 34.246881) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.041253 10.347117 34.246881) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.041253 10.347117 34.246881) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.041253 10.347117 34.246881) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56672     -825.87479      0              0            
Loop time of 1.459e-06 on 1 procs for 0 steps with 80 atoms

68.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.459e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4256 ave        4256 max        4256 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4256
Ave neighs/atom = 53.2
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.026270664896652 y final 0 10.33307714093095 z final 0 34.20041134002692      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.026271 10.347117 34.246881) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026271 10.333077 34.246881) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026271 10.333077 34.200411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026271 10.333077 34.200411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026271 10.333077 34.200411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026271 10.333077 34.200411) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56529     -673.86227      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4352 ave        4352 max        4352 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4352
Ave neighs/atom = 54.4
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.037158241751191 y final 0 10.343280243587772 z final 0 34.23418156191232      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.037158 10.333077 34.200411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.037158 10.34328 34.200411) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.037158 10.34328 34.234182) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.037158 10.34328 34.234182) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.037158 10.34328 34.234182) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.037158 10.34328 34.234182) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56696     -784.76872      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4288 ave        4288 max        4288 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4288
Ave neighs/atom = 53.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.037158241751191 y final 0 10.343280243587772 z final 0 34.23418156191232      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.037158 10.34328 34.234182) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.037158 10.34328 34.234182) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.037158 10.34328 34.234182) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.037158 10.34328 34.234182) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.037158 10.34328 34.234182) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.037158 10.34328 34.234182) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56696     -784.76872      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4288 ave        4288 max        4288 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4288
Ave neighs/atom = 53.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.03539413616422 y final 0 10.341627042821264 z final 0 34.22870979919622      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.035394 10.34328 34.234182) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.035394 10.341627 34.234182) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.035394 10.341627 34.22871) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.035394 10.341627 34.22871) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.035394 10.341627 34.22871) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.035394 10.341627 34.22871) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56752     -760.71784      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4288 ave        4288 max        4288 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4288
Ave neighs/atom = 53.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.028568097306978 y final 0 10.335230139623208 z final 0 34.20753733356064      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.028568 10.341627 34.22871) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.028568 10.33523 34.22871) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.028568 10.33523 34.207537) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.028568 10.33523 34.207537) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.028568 10.33523 34.207537) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.028568 10.33523 34.207537) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56848     -665.32855      0              0            
Loop time of 1.834e-06 on 1 procs for 0 steps with 80 atoms

109.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.834e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4288 ave        4288 max        4288 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4288
Ave neighs/atom = 53.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.02856809730698 y final 0 10.335230139623208 z final 0 34.20753733356064      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.028568 10.33523 34.207537) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.028568 10.33523 34.207537) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.028568 10.33523 34.207537) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.028568 10.33523 34.207537) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.028568 10.33523 34.207537) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.028568 10.33523 34.207537) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56848     -665.32855      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4288 ave        4288 max        4288 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4288
Ave neighs/atom = 53.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.027905049526431 y final 0 10.334608775966107 z final 0 34.2054807445722      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.027905 10.33523 34.207537) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.027905 10.334609 34.207537) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.027905 10.334609 34.205481) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.027905 10.334609 34.205481) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.027905 10.334609 34.205481) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.027905 10.334609 34.205481) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56893     -711.27063      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4288 ave        4288 max        4288 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4288
Ave neighs/atom = 53.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.026451082704234 y final 0 10.333246216330965 z final 0 34.20097094566507      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.026451 10.334609 34.205481) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026451 10.333246 34.205481) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026451 10.333246 34.200971) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026451 10.333246 34.200971) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026451 10.333246 34.200971) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026451 10.333246 34.200971) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56933     -811.62318      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4288 ave        4288 max        4288 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4288
Ave neighs/atom = 53.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.026451082704234 y final 0 10.333246216330965 z final 0 34.20097094566507      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.026451 10.333246 34.200971) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026451 10.333246 34.200971) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026451 10.333246 34.200971) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026451 10.333246 34.200971) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026451 10.333246 34.200971) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.026451 10.333246 34.200971) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56933     -811.62318      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4288 ave        4288 max        4288 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4288
Ave neighs/atom = 53.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.024802605710601 y final 0 10.33170137488294 z final 0 34.195857830543886      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.024803 10.333246 34.200971) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.024803 10.331701 34.200971) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.024803 10.331701 34.195858) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.024803 10.331701 34.195858) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.024803 10.331701 34.195858) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.024803 10.331701 34.195858) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56961     -775.62697      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4288 ave        4288 max        4288 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4288
Ave neighs/atom = 53.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.021406125130273 y final 0 10.328518422376982 z final 0 34.185322896613265      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.021406 10.331701 34.195858) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021406 10.328518 34.195858) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021406 10.328518 34.185323) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021406 10.328518 34.185323) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021406 10.328518 34.185323) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021406 10.328518 34.185323) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56985     -700.92412      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4320 ave        4320 max        4320 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4320
Ave neighs/atom = 54
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.021406125130275 y final 0 10.328518422376982 z final 0 34.185322896613265      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.021406 10.328518 34.185323) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021406 10.328518 34.185323) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021406 10.328518 34.185323) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021406 10.328518 34.185323) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021406 10.328518 34.185323) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021406 10.328518 34.185323) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56985     -700.92412      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4320 ave        4320 max        4320 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4320
Ave neighs/atom = 54
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.021152180893091 y final 0 10.32828044296653 z final 0 34.18453523253054      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.021152 10.328518 34.185323) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021152 10.32828 34.185323) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021152 10.32828 34.184535) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021152 10.32828 34.184535) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021152 10.32828 34.184535) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.021152 10.32828 34.184535) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56988     -689.57532      0              0            
Loop time of 7.91e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4320 ave        4320 max        4320 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4320
Ave neighs/atom = 54
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.0208263015739 y final 0 10.327975050849272 z final 0 34.183524445918984      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.020826 10.32828 34.184535) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020826 10.327975 34.184535) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020826 10.327975 34.183524) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020826 10.327975 34.183524) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020826 10.327975 34.183524) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020826 10.327975 34.183524) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.5699      -674.96422      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4320 ave        4320 max        4320 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4320
Ave neighs/atom = 54
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.0208263015739 y final 0 10.327975050849272 z final 0 34.183524445918984      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.020826 10.327975 34.183524) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020826 10.327975 34.183524) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020826 10.327975 34.183524) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020826 10.327975 34.183524) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020826 10.327975 34.183524) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020826 10.327975 34.183524) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.5699      -674.96422      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4320 ave        4320 max        4320 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4320
Ave neighs/atom = 54
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.02099130500822 y final 0 10.328129680938371 z final 0 34.184036240477425      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.020991 10.327975 34.183524) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020991 10.32813 34.183524) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020991 10.32813 34.184036) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020991 10.32813 34.184036) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020991 10.32813 34.184036) with tilt (0 0 0)
  triclinic box = (0 0 0) to (11.020991 10.32813 34.184036) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.25 | 3.25 | 3.25 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-117.56991     -686.59855      0              0            
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
Nghost:            495 ave         495 max         495 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4320 ave        4320 max        4320 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4320
Ave neighs/atom = 54
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.405456946931836 y final 0 5.547288438890481 z final 0 69.3861733027372      xy final -3.2027284734659185 xz final 0.0 yz final -0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4054569 10.32813 34.184036) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4054569 5.5472884 34.184036) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4054569 5.5472884 69.386173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.4054569 5.5472884 69.386173) with tilt (-3.2027285 0 0)
  triclinic box = (0 0 0) to (6.4054569 5.5472884 69.386173) with tilt (-3.2027285 0 0)
  triclinic box = (0 0 0) to (6.4054569 5.5472884 69.386173) with tilt (-3.2027285 0 -0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33641     -1139.3405      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 5.832537074891441 y final 0 5.051125275370569 z final 0 63.18010278203562      xy final -2.9162685374457205 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.8325371 5.5472884 69.386173) with tilt (-3.2027285 0 -0)
  triclinic box = (0 0 0) to (5.8325371 5.0511253 69.386173) with tilt (-3.2027285 0 -0)
  triclinic box = (0 0 0) to (5.8325371 5.0511253 63.180103) with tilt (-3.2027285 0 -0)
  triclinic box = (0 0 0) to (5.8325371 5.0511253 63.180103) with tilt (-2.9162685 0 -0)
  triclinic box = (0 0 0) to (5.8325371 5.0511253 63.180103) with tilt (-2.9162685 0 -0)
  triclinic box = (0 0 0) to (5.8325371 5.0511253 63.180103) with tilt (-2.9162685 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-100.93575      154037.87      0              0            
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
Nghost:           1540 ave        1540 max        1540 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           6768 ave        6768 max        6768 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 6768
Ave neighs/atom = 84.6
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400866522279599 y final 0 5.543313014527485 z final 0 69.33644820067312      xy final -3.2004332611397994 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4008665 5.0511253 63.180103) with tilt (-2.9162685 0 0)
  triclinic box = (0 0 0) to (6.4008665 5.543313 63.180103) with tilt (-2.9162685 0 0)
  triclinic box = (0 0 0) to (6.4008665 5.543313 69.336448) with tilt (-2.9162685 0 0)
  triclinic box = (0 0 0) to (6.4008665 5.543313 69.336448) with tilt (-3.2004333 0 0)
  triclinic box = (0 0 0) to (6.4008665 5.543313 69.336448) with tilt (-3.2004333 0 0)
  triclinic box = (0 0 0) to (6.4008665 5.543313 69.336448) with tilt (-3.2004333 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33747     -566.30746      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400866522279599 y final 0 5.543313014527485 z final 0 69.33644820067312      xy final -3.200433261139799 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4008665 5.543313 69.336448) with tilt (-3.2004333 0 0)
  triclinic box = (0 0 0) to (6.4008665 5.543313 69.336448) with tilt (-3.2004333 0 0)
  triclinic box = (0 0 0) to (6.4008665 5.543313 69.336448) with tilt (-3.2004333 0 0)
  triclinic box = (0 0 0) to (6.4008665 5.543313 69.336448) with tilt (-3.2004333 0 0)
  triclinic box = (0 0 0) to (6.4008665 5.543313 69.336448) with tilt (-3.2004333 0 0)
  triclinic box = (0 0 0) to (6.4008665 5.543313 69.336448) with tilt (-3.2004333 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33747     -566.30746      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400773442049901 y final 0 5.543232404683978 z final 0 69.33543992273098      xy final -3.20038672102495 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4007734 5.543313 69.336448) with tilt (-3.2004333 0 0)
  triclinic box = (0 0 0) to (6.4007734 5.5432324 69.336448) with tilt (-3.2004333 0 0)
  triclinic box = (0 0 0) to (6.4007734 5.5432324 69.33544) with tilt (-3.2004333 0 0)
  triclinic box = (0 0 0) to (6.4007734 5.5432324 69.33544) with tilt (-3.2003867 0 0)
  triclinic box = (0 0 0) to (6.4007734 5.5432324 69.33544) with tilt (-3.2003867 0 0)
  triclinic box = (0 0 0) to (6.4007734 5.5432324 69.33544) with tilt (-3.2003867 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33791     -537.71598      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400628602419999 y final 0 5.543106969885009 z final 0 69.33387096867436      xy final -3.200314301209999 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006286 5.5432324 69.33544) with tilt (-3.2003867 0 0)
  triclinic box = (0 0 0) to (6.4006286 5.543107 69.33544) with tilt (-3.2003867 0 0)
  triclinic box = (0 0 0) to (6.4006286 5.543107 69.333871) with tilt (-3.2003867 0 0)
  triclinic box = (0 0 0) to (6.4006286 5.543107 69.333871) with tilt (-3.2003143 0 0)
  triclinic box = (0 0 0) to (6.4006286 5.543107 69.333871) with tilt (-3.2003143 0 0)
  triclinic box = (0 0 0) to (6.4006286 5.543107 69.333871) with tilt (-3.2003143 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33836     -495.05202      0              0            
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
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.401025577282441 y final 0 5.543450760200548 z final 0 69.33817114067308      xy final -3.2005127886412192 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4010256 5.543107 69.333871) with tilt (-3.2003143 0 0)
  triclinic box = (0 0 0) to (6.4010256 5.5434508 69.333871) with tilt (-3.2003143 0 0)
  triclinic box = (0 0 0) to (6.4010256 5.5434508 69.338171) with tilt (-3.2003143 0 0)
  triclinic box = (0 0 0) to (6.4010256 5.5434508 69.338171) with tilt (-3.2005128 0 0)
  triclinic box = (0 0 0) to (6.4010256 5.5434508 69.338171) with tilt (-3.2005128 0 0)
  triclinic box = (0 0 0) to (6.4010256 5.5434508 69.338171) with tilt (-3.2005128 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33823     -540.54898      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.4006575386162 y final 0 5.543132029366008 z final 0 69.3341844157775      xy final -3.200328769308099 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006575 5.5434508 69.338171) with tilt (-3.2005128 0 0)
  triclinic box = (0 0 0) to (6.4006575 5.543132 69.338171) with tilt (-3.2005128 0 0)
  triclinic box = (0 0 0) to (6.4006575 5.543132 69.334184) with tilt (-3.2005128 0 0)
  triclinic box = (0 0 0) to (6.4006575 5.543132 69.334184) with tilt (-3.2003288 0 0)
  triclinic box = (0 0 0) to (6.4006575 5.543132 69.334184) with tilt (-3.2003288 0 0)
  triclinic box = (0 0 0) to (6.4006575 5.543132 69.334184) with tilt (-3.2003288 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33836     -498.5545       0              0            
Loop time of 1.459e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.459e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400900441761573 y final 0 5.543342389660561 z final 0 69.33681562848882      xy final -3.2004502208807852 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4009004 5.543132 69.334184) with tilt (-3.2003288 0 0)
  triclinic box = (0 0 0) to (6.4009004 5.5433424 69.334184) with tilt (-3.2003288 0 0)
  triclinic box = (0 0 0) to (6.4009004 5.5433424 69.336816) with tilt (-3.2003288 0 0)
  triclinic box = (0 0 0) to (6.4009004 5.5433424 69.336816) with tilt (-3.2004502 0 0)
  triclinic box = (0 0 0) to (6.4009004 5.5433424 69.336816) with tilt (-3.2004502 0 0)
  triclinic box = (0 0 0) to (6.4009004 5.5433424 69.336816) with tilt (-3.2004502 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33821     -525.42152      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.40066792871907 y final 0 5.543141027459042 z final 0 69.33429696504359      xy final -3.2003339643595337 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006679 5.5433424 69.336816) with tilt (-3.2004502 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.336816) with tilt (-3.2004502 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.2004502 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200334 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.81204      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400821386379456 y final 0 5.5432739256913415 z final 0 69.33595927265199      xy final -3.2004106931897267 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4008214 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.4008214 5.5432739 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.4008214 5.5432739 69.335959) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.4008214 5.5432739 69.335959) with tilt (-3.2004107 0 0)
  triclinic box = (0 0 0) to (6.4008214 5.5432739 69.335959) with tilt (-3.2004107 0 0)
  triclinic box = (0 0 0) to (6.4008214 5.5432739 69.335959) with tilt (-3.2004107 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.3382      -515.86107      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.4006719900702755 y final 0 5.543144544692359 z final 0 69.3343409590351      xy final -3.2003359950351364 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.400672 5.5432739 69.335959) with tilt (-3.2004107 0 0)
  triclinic box = (0 0 0) to (6.400672 5.5431445 69.335959) with tilt (-3.2004107 0 0)
  triclinic box = (0 0 0) to (6.400672 5.5431445 69.334341) with tilt (-3.2004107 0 0)
  triclinic box = (0 0 0) to (6.400672 5.5431445 69.334341) with tilt (-3.200336 0 0)
  triclinic box = (0 0 0) to (6.400672 5.5431445 69.334341) with tilt (-3.200336 0 0)
  triclinic box = (0 0 0) to (6.400672 5.5431445 69.334341) with tilt (-3.200336 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33818     -497.78654      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400667931499453 y final 0 5.5431410298669235 z final 0 69.33429699516168      xy final -3.200333965749725 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006679 5.5431445 69.334341) with tilt (-3.200336 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334341) with tilt (-3.200336 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200336 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200334 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.81238      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400667949617195 y final 0 5.543141045557348 z final 0 69.33429719141947      xy final -3.200333974808596 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.4006679 5.543141 69.334297) with tilt (-3.200334 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.81457      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669969843417 y final 0 5.543142795124577 z final 0 69.33431907522383      xy final -3.200334984921707 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.40067 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.40067 5.5431428 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.40067 5.5431428 69.334319) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.40067 5.5431428 69.334319) with tilt (-3.200335 0 0)
  triclinic box = (0 0 0) to (6.40067 5.5431428 69.334319) with tilt (-3.200335 0 0)
  triclinic box = (0 0 0) to (6.40067 5.5431428 69.334319) with tilt (-3.200335 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33817     -497.54205      0              0            
Loop time of 5.84e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400667950304903 y final 0 5.543141046152921 z final 0 69.33429719886897      xy final -3.20033397515245 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.400668 5.5431428 69.334319) with tilt (-3.200335 0 0)
  triclinic box = (0 0 0) to (6.400668 5.543141 69.334319) with tilt (-3.200335 0 0)
  triclinic box = (0 0 0) to (6.400668 5.543141 69.334297) with tilt (-3.200335 0 0)
  triclinic box = (0 0 0) to (6.400668 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.400668 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.400668 5.543141 69.334297) with tilt (-3.200334 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.81465      0              0            
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
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400667951673682 y final 0 5.543141047338319 z final 0 69.33429721369606      xy final -3.20033397583684 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.400668 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.400668 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.400668 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.400668 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.400668 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.400668 5.543141 69.334297) with tilt (-3.200334 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.81482      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400668960758471 y final 0 5.5431419212313795 z final 0 69.33430814445909      xy final -3.200334480379234 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.400669 5.543141 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334297) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.200334 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003345 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.93695      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.4006696267545165 y final 0 5.543142498000875 z final 0 69.33431535876362      xy final -3.200334813377257 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006696 5.5431419 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.4006696 5.5431425 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.4006696 5.5431425 69.334315) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.4006696 5.5431425 69.334315) with tilt (-3.2003348 0 0)
  triclinic box = (0 0 0) to (6.4006696 5.5431425 69.334315) with tilt (-3.2003348 0 0)
  triclinic box = (0 0 0) to (6.4006696 5.5431425 69.334315) with tilt (-3.2003348 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33817     -497.50053      0              0            
Loop time of 5.42e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400668960833122 y final 0 5.54314192129603 z final 0 69.33430814526776      xy final -3.2003344804165597 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.400669 5.5431425 69.334315) with tilt (-3.2003348 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334315) with tilt (-3.2003348 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003348 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003345 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.93696      0              0            
Loop time of 1.709e-06 on 1 procs for 0 steps with 80 atoms

175.5% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.709e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.4006689608937695 y final 0 5.5431419213485515 z final 0 69.3343081459247      xy final -3.2003344804468834 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.400669 5.5431419 69.334308) with tilt (-3.2003345 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.93697      0              0            
Loop time of 9.16e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669293824134 y final 0 5.543142209674706 z final 0 69.33431175234406      xy final -3.2003346469120655 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006693 5.5431419 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.4006693 5.5431422 69.334308) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.4006693 5.5431422 69.334312) with tilt (-3.2003345 0 0)
  triclinic box = (0 0 0) to (6.4006693 5.5431422 69.334312) with tilt (-3.2003346 0 0)
  triclinic box = (0 0 0) to (6.4006693 5.5431422 69.334312) with tilt (-3.2003346 0 0)
  triclinic box = (0 0 0) to (6.4006693 5.5431422 69.334312) with tilt (-3.2003346 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.97726      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669513558185 y final 0 5.543142399969976 z final 0 69.33431413258096      xy final -3.200334756779091 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006695 5.5431422 69.334312) with tilt (-3.2003346 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334312) with tilt (-3.2003346 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334314) with tilt (-3.2003346 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334314) with tilt (-3.2003348 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334314) with tilt (-3.2003348 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334314) with tilt (-3.2003348 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33817     -497.48683      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669293832258 y final 0 5.543142209681741 z final 0 69.33431175243206      xy final -3.2003346469161276 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006693 5.5431424 69.334314) with tilt (-3.2003348 0 0)
  triclinic box = (0 0 0) to (6.4006693 5.5431422 69.334314) with tilt (-3.2003348 0 0)
  triclinic box = (0 0 0) to (6.4006693 5.5431422 69.334312) with tilt (-3.2003348 0 0)
  triclinic box = (0 0 0) to (6.4006693 5.5431422 69.334312) with tilt (-3.2003346 0 0)
  triclinic box = (0 0 0) to (6.4006693 5.5431422 69.334312) with tilt (-3.2003346 0 0)
  triclinic box = (0 0 0) to (6.4006693 5.5431422 69.334312) with tilt (-3.2003346 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.97726      0              0            
Loop time of 1.417e-06 on 1 procs for 0 steps with 80 atoms

70.6% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.417e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438851369 y final 0 5.543142335271975 z final 0 69.33431332333032      xy final -3.2003347194256833 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431422 69.334312) with tilt (-3.2003346 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334312) with tilt (-3.2003346 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003346 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669488157867 y final 0 5.543142377972655 z final 0 69.33431385743573      xy final -3.200334744078932 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006695 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334314) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334314) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334314) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334314) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33817     -497.48376      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438851778 y final 0 5.5431423352723295 z final 0 69.33431332333475      xy final -3.2003347194258875 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431424 69.334314) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334314) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.40066943885137 y final 0 5.543142335271977 z final 0 69.33431332333033      xy final -3.2003347194256837 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.4006694388515735 y final 0 5.543142335272153 z final 0 69.33431332333254      xy final -3.2003347194257854 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.40066943885137 y final 0 5.543142335271977 z final 0 69.33431332333033      xy final -3.2003347194256837 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438851371 y final 0 5.543142335271977 z final 0 69.33431332333035      xy final -3.200334719425684 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.40066943885137 y final 0 5.543142335271977 z final 0 69.33431332333033      xy final -3.2003347194256837 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438851371 y final 0 5.543142335271977 z final 0 69.33431332333035      xy final -3.200334719425684 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.40066943885137 y final 0 5.543142335271977 z final 0 69.33431332333033      xy final -3.2003347194256837 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.40066943885137 y final 0 5.543142335271977 z final 0 69.33431332333033      xy final -3.2003347194256837 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438851371 y final 0 5.543142335271977 z final 0 69.33431332333035      xy final -3.200334719425684 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.40066943885137 y final 0 5.543142335271977 z final 0 69.33431332333033      xy final -3.2003347194256837 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
Loop time of 6.25e-07 on 1 procs for 0 steps with 80 atoms

320.0% CPU use with 1 MPI tasks x no OpenMP threads

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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438851371 y final 0 5.543142335271977 z final 0 69.33431332333035      xy final -3.200334719425684 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.40066943885137 y final 0 5.543142335271977 z final 0 69.33431332333033      xy final -3.2003347194256837 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.40066943885137 y final 0 5.543142335271977 z final 0 69.33431332333033      xy final -3.2003347194256833 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.40139649488764 y final 0 5.543771984269383 z final 0 69.34218905125257      xy final -3.200698247443803 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4013965 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4013965 5.543772 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4013965 5.543772 69.342189) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4013965 5.543772 69.342189) with tilt (-3.2006982 0 0)
  triclinic box = (0 0 0) to (6.4013965 5.543772 69.342189) with tilt (-3.2006982 0 0)
  triclinic box = (0 0 0) to (6.4013965 5.543772 69.342189) with tilt (-3.2006982 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33833     -584.20938      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400853750100504 y final 0 5.543301953495936 z final 0 69.33630984791547      xy final -3.2004268750502463 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4008538 5.543772 69.342189) with tilt (-3.2006982 0 0)
  triclinic box = (0 0 0) to (6.4008538 5.543302 69.342189) with tilt (-3.2006982 0 0)
  triclinic box = (0 0 0) to (6.4008538 5.543302 69.33631) with tilt (-3.2006982 0 0)
  triclinic box = (0 0 0) to (6.4008538 5.543302 69.33631) with tilt (-3.2004269 0 0)
  triclinic box = (0 0 0) to (6.4008538 5.543302 69.33631) with tilt (-3.2004269 0 0)
  triclinic box = (0 0 0) to (6.4008538 5.543302 69.33631) with tilt (-3.2004269 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33824     -519.50901      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400681285426194 y final 0 5.543152594706723 z final 0 69.33444164961534      xy final -3.200340642713095 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006813 5.543302 69.33631) with tilt (-3.2004269 0 0)
  triclinic box = (0 0 0) to (6.4006813 5.5431526 69.33631) with tilt (-3.2004269 0 0)
  triclinic box = (0 0 0) to (6.4006813 5.5431526 69.334442) with tilt (-3.2004269 0 0)
  triclinic box = (0 0 0) to (6.4006813 5.5431526 69.334442) with tilt (-3.2003406 0 0)
  triclinic box = (0 0 0) to (6.4006813 5.5431526 69.334442) with tilt (-3.2003406 0 0)
  triclinic box = (0 0 0) to (6.4006813 5.5431526 69.334442) with tilt (-3.2003406 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33818     -498.89476      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669485451981 y final 0 5.54314237562929 z final 0 69.3343138281246      xy final -3.2003347427259885 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006695 5.5431526 69.334442) with tilt (-3.2003406 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334442) with tilt (-3.2003406 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334314) with tilt (-3.2003406 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334314) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334314) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006695 5.5431424 69.334314) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33817     -497.48337      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431424 69.334314) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334314) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438860999 y final 0 5.543142335280316 z final 0 69.33431332343463      xy final -3.2003347194304976 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858722 y final 0 5.543142335278344 z final 0 69.33431332340997      xy final -3.2003347194293594 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858642 y final 0 5.543142335278274 z final 0 69.33431332340909      xy final -3.200334719429319 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278274 z final 0 69.33431332340909      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278274 z final 0 69.33431332340909      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278273 z final 0 69.33431332340908      xy final -3.2003347194293186 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400669438858641 y final 0 5.543142335278274 z final 0 69.33431332340908      xy final -3.200334719429318 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4006694 5.5431423 69.334313) with tilt (-3.2003347 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -499.99481      0              0            
Loop time of 7.08e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.401341100822779 y final 0 5.543724092980852 z final 0 69.3415879852692      xy final -3.2006704917446753 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4013411 5.5431423 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4013411 5.5437241 69.334313) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4013411 5.5437241 69.341588) with tilt (-3.2003347 0 0)
  triclinic box = (0 0 0) to (6.4013411 5.5437241 69.341588) with tilt (-3.2006705 0 0)
  triclinic box = (0 0 0) to (6.4013411 5.5437241 69.341588) with tilt (-3.2006705 0 0)
  triclinic box = (0 0 0) to (6.4013411 5.5437241 69.341588) with tilt (-3.2006705 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -578.13909      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.404028453569878 y final 0 5.546051734415111 z final 0 69.37069426619671      xy final -3.202013933328223 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4040285 5.5437241 69.341588) with tilt (-3.2006705 0 0)
  triclinic box = (0 0 0) to (6.4040285 5.5460517 69.341588) with tilt (-3.2006705 0 0)
  triclinic box = (0 0 0) to (6.4040285 5.5460517 69.370694) with tilt (-3.2006705 0 0)
  triclinic box = (0 0 0) to (6.4040285 5.5460517 69.370694) with tilt (-3.2020139 0 0)
  triclinic box = (0 0 0) to (6.4040285 5.5460517 69.370694) with tilt (-3.2020139 0 0)
  triclinic box = (0 0 0) to (6.4040285 5.5460517 69.370694) with tilt (-3.2020139 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33589     -894.69525      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.401410182613824 y final 0 5.543783927937302 z final 0 69.3423361989441      xy final -3.2007050266058585 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4014102 5.5460517 69.370694) with tilt (-3.2020139 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.370694) with tilt (-3.2020139 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.2020139 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -586.40017      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.4014101888546735 y final 0 5.543783933342793 z final 0 69.34233626653773      xy final -3.2007050297257384 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -586.40091      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.402719187376365 y final 0 5.544917717941216 z final 0 69.356513817015      xy final -3.20135941463241 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4027192 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4027192 5.5449177 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4027192 5.5449177 69.356514) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4027192 5.5449177 69.356514) with tilt (-3.2013594 0 0)
  triclinic box = (0 0 0) to (6.4027192 5.5449177 69.356514) with tilt (-3.2013594 0 0)
  triclinic box = (0 0 0) to (6.4027192 5.5449177 69.356514) with tilt (-3.2013594 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33775     -741.70106      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.401410188856237 y final 0 5.543783933344146 z final 0 69.34233626655465      xy final -3.2007050297265196 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4014102 5.5449177 69.356514) with tilt (-3.2013594 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.356514) with tilt (-3.2013594 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.2013594 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -586.40091      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.4014101888546735 y final 0 5.543783933342793 z final 0 69.34233626653773      xy final -3.2007050297257384 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -586.40091      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.4014101888546735 y final 0 5.543783933342793 z final 0 69.34233626653773      xy final -3.200705029725738 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4014102 5.5437839 69.342336) with tilt (-3.200705 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33837     -586.40091      0              0            
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
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.4013226748894665 y final 0 5.543708151330142 z final 0 69.34138819297527      xy final -3.200661267477329 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4013227 5.5437839 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4013227 5.5437082 69.342336) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4013227 5.5437082 69.341388) with tilt (-3.200705 0 0)
  triclinic box = (0 0 0) to (6.4013227 5.5437082 69.341388) with tilt (-3.2006613 0 0)
  triclinic box = (0 0 0) to (6.4013227 5.5437082 69.341388) with tilt (-3.2006613 0 0)
  triclinic box = (0 0 0) to (6.4013227 5.5437082 69.341388) with tilt (-3.2006613 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33841     -577.62819      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.40097263099255 y final 0 5.54340503363859 z final 0 69.33759602834773      xy final -3.2004862244670895 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4009726 5.5437082 69.341388) with tilt (-3.2006613 0 0)
  triclinic box = (0 0 0) to (6.4009726 5.543405 69.341388) with tilt (-3.2006613 0 0)
  triclinic box = (0 0 0) to (6.4009726 5.543405 69.337596) with tilt (-3.2006613 0 0)
  triclinic box = (0 0 0) to (6.4009726 5.543405 69.337596) with tilt (-3.2004862 0 0)
  triclinic box = (0 0 0) to (6.4009726 5.543405 69.337596) with tilt (-3.2004862 0 0)
  triclinic box = (0 0 0) to (6.4009726 5.543405 69.337596) with tilt (-3.2004862 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33871     -545.0069       0              0            
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
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.399872965760375 y final 0 5.542452787382525 z final 0 69.32568291805839      xy final -3.1999363256927684 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.399873 5.543405 69.337596) with tilt (-3.2004862 0 0)
  triclinic box = (0 0 0) to (6.399873 5.5424528 69.337596) with tilt (-3.2004862 0 0)
  triclinic box = (0 0 0) to (6.399873 5.5424528 69.325683) with tilt (-3.2004862 0 0)
  triclinic box = (0 0 0) to (6.399873 5.5424528 69.325683) with tilt (-3.1999363 0 0)
  triclinic box = (0 0 0) to (6.399873 5.5424528 69.325683) with tilt (-3.1999363 0 0)
  triclinic box = (0 0 0) to (6.399873 5.5424528 69.325683) with tilt (-3.1999363 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33885     -434.16815      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.399872965760375 y final 0 5.542452787382525 z final 0 69.32568291805839      xy final -3.199936325692768 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.399873 5.5424528 69.325683) with tilt (-3.1999363 0 0)
  triclinic box = (0 0 0) to (6.399873 5.5424528 69.325683) with tilt (-3.1999363 0 0)
  triclinic box = (0 0 0) to (6.399873 5.5424528 69.325683) with tilt (-3.1999363 0 0)
  triclinic box = (0 0 0) to (6.399873 5.5424528 69.325683) with tilt (-3.1999363 0 0)
  triclinic box = (0 0 0) to (6.399873 5.5424528 69.325683) with tilt (-3.1999363 0 0)
  triclinic box = (0 0 0) to (6.399873 5.5424528 69.325683) with tilt (-3.1999363 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33885     -434.16815      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.399959792618689 y final 0 5.542528009775637 z final 0 69.3266231054423      xy final -3.199979718844157 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3999598 5.5424528 69.325683) with tilt (-3.1999363 0 0)
  triclinic box = (0 0 0) to (6.3999598 5.542528 69.325683) with tilt (-3.1999363 0 0)
  triclinic box = (0 0 0) to (6.3999598 5.542528 69.326623) with tilt (-3.1999363 0 0)
  triclinic box = (0 0 0) to (6.3999598 5.542528 69.326623) with tilt (-3.1999797 0 0)
  triclinic box = (0 0 0) to (6.3999598 5.542528 69.326623) with tilt (-3.1999797 0 0)
  triclinic box = (0 0 0) to (6.3999598 5.542528 69.326623) with tilt (-3.1999797 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33897     -454.36707      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400307111831876 y final 0 5.542828909557435 z final 0 69.33038398248686      xy final -3.200153297334176 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4003071 5.542528 69.326623) with tilt (-3.1999797 0 0)
  triclinic box = (0 0 0) to (6.4003071 5.5428289 69.326623) with tilt (-3.1999797 0 0)
  triclinic box = (0 0 0) to (6.4003071 5.5428289 69.330384) with tilt (-3.1999797 0 0)
  triclinic box = (0 0 0) to (6.4003071 5.5428289 69.330384) with tilt (-3.2001533 0 0)
  triclinic box = (0 0 0) to (6.4003071 5.5428289 69.330384) with tilt (-3.2001533 0 0)
  triclinic box = (0 0 0) to (6.4003071 5.5428289 69.330384) with tilt (-3.2001533 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33927     -534.88457      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.400307111831876 y final 0 5.542828909557435 z final 0 69.33038398248686      xy final -3.2001532973341758 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.4003071 5.5428289 69.330384) with tilt (-3.2001533 0 0)
  triclinic box = (0 0 0) to (6.4003071 5.5428289 69.330384) with tilt (-3.2001533 0 0)
  triclinic box = (0 0 0) to (6.4003071 5.5428289 69.330384) with tilt (-3.2001533 0 0)
  triclinic box = (0 0 0) to (6.4003071 5.5428289 69.330384) with tilt (-3.2001533 0 0)
  triclinic box = (0 0 0) to (6.4003071 5.5428289 69.330384) with tilt (-3.2001533 0 0)
  triclinic box = (0 0 0) to (6.4003071 5.5428289 69.330384) with tilt (-3.2001533 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33927     -534.88457      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.399786161195404 y final 0 5.542377894672704 z final 0 69.32473908993428      xy final -3.1998927199349074 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3997862 5.5428289 69.330384) with tilt (-3.2001533 0 0)
  triclinic box = (0 0 0) to (6.3997862 5.5423779 69.330384) with tilt (-3.2001533 0 0)
  triclinic box = (0 0 0) to (6.3997862 5.5423779 69.324739) with tilt (-3.2001533 0 0)
  triclinic box = (0 0 0) to (6.3997862 5.5423779 69.324739) with tilt (-3.1998927 0 0)
  triclinic box = (0 0 0) to (6.3997862 5.5423779 69.324739) with tilt (-3.1998927 0 0)
  triclinic box = (0 0 0) to (6.3997862 5.5423779 69.324739) with tilt (-3.1998927 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.3396      -530.12355      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.399100765889193 y final 0 5.5417845112069175 z final 0 69.31731231573676      xy final -3.1995498879902553 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3991008 5.5423779 69.324739) with tilt (-3.1998927 0 0)
  triclinic box = (0 0 0) to (6.3991008 5.5417845 69.324739) with tilt (-3.1998927 0 0)
  triclinic box = (0 0 0) to (6.3991008 5.5417845 69.317312) with tilt (-3.1998927 0 0)
  triclinic box = (0 0 0) to (6.3991008 5.5417845 69.317312) with tilt (-3.1995499 0 0)
  triclinic box = (0 0 0) to (6.3991008 5.5417845 69.317312) with tilt (-3.1995499 0 0)
  triclinic box = (0 0 0) to (6.3991008 5.5417845 69.317312) with tilt (-3.1995499 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33976     -523.31142      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.399100765889193 y final 0 5.541784511206918 z final 0 69.31731231573676      xy final -3.199549887990255 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3991008 5.5417845 69.317312) with tilt (-3.1995499 0 0)
  triclinic box = (0 0 0) to (6.3991008 5.5417845 69.317312) with tilt (-3.1995499 0 0)
  triclinic box = (0 0 0) to (6.3991008 5.5417845 69.317312) with tilt (-3.1995499 0 0)
  triclinic box = (0 0 0) to (6.3991008 5.5417845 69.317312) with tilt (-3.1995499 0 0)
  triclinic box = (0 0 0) to (6.3991008 5.5417845 69.317312) with tilt (-3.1995499 0 0)
  triclinic box = (0 0 0) to (6.3991008 5.5417845 69.317312) with tilt (-3.1995499 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.33976     -523.31142      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.39814358853468 y final 0 5.54095562247056 z final 0 69.30694319234868      xy final -3.1990712624251443 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3981436 5.5417845 69.317312) with tilt (-3.1995499 0 0)
  triclinic box = (0 0 0) to (6.3981436 5.5409556 69.317312) with tilt (-3.1995499 0 0)
  triclinic box = (0 0 0) to (6.3981436 5.5409556 69.306943) with tilt (-3.1995499 0 0)
  triclinic box = (0 0 0) to (6.3981436 5.5409556 69.306943) with tilt (-3.1990713 0 0)
  triclinic box = (0 0 0) to (6.3981436 5.5409556 69.306943) with tilt (-3.1990713 0 0)
  triclinic box = (0 0 0) to (6.3981436 5.5409556 69.306943) with tilt (-3.1990713 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.34004     -428.767        0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.39532470431917 y final 0 5.538514547776334 z final 0 69.27640615804053      xy final -3.1976617117149035 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3953247 5.5409556 69.306943) with tilt (-3.1990713 0 0)
  triclinic box = (0 0 0) to (6.3953247 5.5385145 69.306943) with tilt (-3.1990713 0 0)
  triclinic box = (0 0 0) to (6.3953247 5.5385145 69.276406) with tilt (-3.1990713 0 0)
  triclinic box = (0 0 0) to (6.3953247 5.5385145 69.276406) with tilt (-3.1976617 0 0)
  triclinic box = (0 0 0) to (6.3953247 5.5385145 69.276406) with tilt (-3.1976617 0 0)
  triclinic box = (0 0 0) to (6.3953247 5.5385145 69.276406) with tilt (-3.1976617 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.34123     -159.62913      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.39532470431917 y final 0 5.538514547776334 z final 0 69.27640615804053      xy final -3.197661711714903 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3953247 5.5385145 69.276406) with tilt (-3.1976617 0 0)
  triclinic box = (0 0 0) to (6.3953247 5.5385145 69.276406) with tilt (-3.1976617 0 0)
  triclinic box = (0 0 0) to (6.3953247 5.5385145 69.276406) with tilt (-3.1976617 0 0)
  triclinic box = (0 0 0) to (6.3953247 5.5385145 69.276406) with tilt (-3.1976617 0 0)
  triclinic box = (0 0 0) to (6.3953247 5.5385145 69.276406) with tilt (-3.1976617 0 0)
  triclinic box = (0 0 0) to (6.3953247 5.5385145 69.276406) with tilt (-3.1976617 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.34123     -159.62913      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.395217222067516 y final 0 5.538421541501055 z final 0 69.27524092062532      xy final -3.1976079157386863 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3952172 5.5385145 69.276406) with tilt (-3.1976617 0 0)
  triclinic box = (0 0 0) to (6.3952172 5.5384215 69.276406) with tilt (-3.1976617 0 0)
  triclinic box = (0 0 0) to (6.3952172 5.5384215 69.275241) with tilt (-3.1976617 0 0)
  triclinic box = (0 0 0) to (6.3952172 5.5384215 69.275241) with tilt (-3.1976079 0 0)
  triclinic box = (0 0 0) to (6.3952172 5.5384215 69.275241) with tilt (-3.1976079 0 0)
  triclinic box = (0 0 0) to (6.3952172 5.5384215 69.275241) with tilt (-3.1976079 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.34129     -176.00308      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.394787311124474 y final 0 5.538049532017888 z final 0 69.27058016695554      xy final -3.1973927408840357 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3947873 5.5384215 69.275241) with tilt (-3.1976079 0 0)
  triclinic box = (0 0 0) to (6.3947873 5.5380495 69.275241) with tilt (-3.1976079 0 0)
  triclinic box = (0 0 0) to (6.3947873 5.5380495 69.27058) with tilt (-3.1976079 0 0)
  triclinic box = (0 0 0) to (6.3947873 5.5380495 69.27058) with tilt (-3.1973927 0 0)
  triclinic box = (0 0 0) to (6.3947873 5.5380495 69.27058) with tilt (-3.1973927 0 0)
  triclinic box = (0 0 0) to (6.3947873 5.5380495 69.27058) with tilt (-3.1973927 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.3416      -250.18213      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.394787311124474 y final 0 5.538049532017888 z final 0 69.27058016695554      xy final -3.1973927408840352 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3947873 5.5380495 69.27058) with tilt (-3.1973927 0 0)
  triclinic box = (0 0 0) to (6.3947873 5.5380495 69.27058) with tilt (-3.1973927 0 0)
  triclinic box = (0 0 0) to (6.3947873 5.5380495 69.27058) with tilt (-3.1973927 0 0)
  triclinic box = (0 0 0) to (6.3947873 5.5380495 69.27058) with tilt (-3.1973927 0 0)
  triclinic box = (0 0 0) to (6.3947873 5.5380495 69.27058) with tilt (-3.1973927 0 0)
  triclinic box = (0 0 0) to (6.3947873 5.5380495 69.27058) with tilt (-3.1973927 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.3416      -250.18213      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.394705368102161 y final 0 5.5379785283270415 z final 0 69.26969301839414      xy final -3.197351797453601 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3947054 5.5380495 69.27058) with tilt (-3.1973927 0 0)
  triclinic box = (0 0 0) to (6.3947054 5.5379785 69.27058) with tilt (-3.1973927 0 0)
  triclinic box = (0 0 0) to (6.3947054 5.5379785 69.269693) with tilt (-3.1973927 0 0)
  triclinic box = (0 0 0) to (6.3947054 5.5379785 69.269693) with tilt (-3.1973518 0 0)
  triclinic box = (0 0 0) to (6.3947054 5.5379785 69.269693) with tilt (-3.1973518 0 0)
  triclinic box = (0 0 0) to (6.3947054 5.5379785 69.269693) with tilt (-3.1973518 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.34161     -224.23394      0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.394377606512985 y final 0 5.537694522666964 z final 0 69.2661445377642      xy final -3.197188028974706 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3943776 5.5379785 69.269693) with tilt (-3.1973518 0 0)
  triclinic box = (0 0 0) to (6.3943776 5.5376945 69.269693) with tilt (-3.1973518 0 0)
  triclinic box = (0 0 0) to (6.3943776 5.5376945 69.266145) with tilt (-3.1973518 0 0)
  triclinic box = (0 0 0) to (6.3943776 5.5376945 69.266145) with tilt (-3.197188 0 0)
  triclinic box = (0 0 0) to (6.3943776 5.5376945 69.266145) with tilt (-3.197188 0 0)
  triclinic box = (0 0 0) to (6.3943776 5.5376945 69.266145) with tilt (-3.197188 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.34233     -138.77281      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.394543438170123 y final 0 5.537838215919851 z final 0 69.26793989892519      xy final -3.1972708879776364 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3945434 5.5376945 69.266145) with tilt (-3.197188 0 0)
  triclinic box = (0 0 0) to (6.3945434 5.5378382 69.266145) with tilt (-3.197188 0 0)
  triclinic box = (0 0 0) to (6.3945434 5.5378382 69.26794) with tilt (-3.197188 0 0)
  triclinic box = (0 0 0) to (6.3945434 5.5378382 69.26794) with tilt (-3.1972709 0 0)
  triclinic box = (0 0 0) to (6.3945434 5.5378382 69.26794) with tilt (-3.1972709 0 0)
  triclinic box = (0 0 0) to (6.3945434 5.5378382 69.26794) with tilt (-3.1972709 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.34212     -182.33497      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.394543438170123 y final 0 5.537838215919851 z final 0 69.26793989892519      xy final -3.197270887977636 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3945434 5.5378382 69.26794) with tilt (-3.1972709 0 0)
  triclinic box = (0 0 0) to (6.3945434 5.5378382 69.26794) with tilt (-3.1972709 0 0)
  triclinic box = (0 0 0) to (6.3945434 5.5378382 69.26794) with tilt (-3.1972709 0 0)
  triclinic box = (0 0 0) to (6.3945434 5.5378382 69.26794) with tilt (-3.1972709 0 0)
  triclinic box = (0 0 0) to (6.3945434 5.5378382 69.26794) with tilt (-3.1972709 0 0)
  triclinic box = (0 0 0) to (6.3945434 5.5378382 69.26794) with tilt (-3.1972709 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.34212     -182.33497      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.39385661538317 y final 0 5.537243532867111 z final 0 69.2604984541713      xy final -3.196927387963864 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3938566 5.5378382 69.26794) with tilt (-3.1972709 0 0)
  triclinic box = (0 0 0) to (6.3938566 5.5372435 69.26794) with tilt (-3.1972709 0 0)
  triclinic box = (0 0 0) to (6.3938566 5.5372435 69.260498) with tilt (-3.1972709 0 0)
  triclinic box = (0 0 0) to (6.3938566 5.5372435 69.260498) with tilt (-3.1969274 0 0)
  triclinic box = (0 0 0) to (6.3938566 5.5372435 69.260498) with tilt (-3.1969274 0 0)
  triclinic box = (0 0 0) to (6.3938566 5.5372435 69.260498) with tilt (-3.1969274 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.34252     -149.07227      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 6.392248337586267 y final 0 5.535851011239114 z final 0 69.24307342087666      xy final -3.1961230415876583 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.3922483 5.5372435 69.260498) with tilt (-3.1969274 0 0)
  triclinic box = (0 0 0) to (6.3922483 5.535851 69.260498) with tilt (-3.1969274 0 0)
  triclinic box = (0 0 0) to (6.3922483 5.535851 69.243073) with tilt (-3.1969274 0 0)
  triclinic box = (0 0 0) to (6.3922483 5.535851 69.243073) with tilt (-3.196123 0 0)
  triclinic box = (0 0 0) to (6.3922483 5.535851 69.243073) with tilt (-3.196123 0 0)
  triclinic box = (0 0 0) to (6.3922483 5.535851 69.243073) with tilt (-3.196123 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-122.34328     -71.917823      0              0            
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
Nghost:           1370 ave        1370 max        1370 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           5028 ave        5028 max        5028 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 5028
Ave neighs/atom = 62.85
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.094576877780963 y final 0 10.879197202385848 z final 0 34.53880286290898      xy final -5.54728843889048 xz final 0.0 yz final -0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.094577 5.535851 69.243073) with tilt (-3.196123 0 0)
  triclinic box = (0 0 0) to (11.094577 10.879197 69.243073) with tilt (-3.196123 0 0)
  triclinic box = (0 0 0) to (11.094577 10.879197 34.538803) with tilt (-3.196123 0 0)
  triclinic box = (0 0 0) to (11.094577 10.879197 34.538803) with tilt (-5.5472884 0 0)
  triclinic box = (0 0 0) to (11.094577 10.879197 34.538803) with tilt (-5.5472884 0 0)
  triclinic box = (0 0 0) to (11.094577 10.879197 34.538803) with tilt (-5.5472884 0 -0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.79923     -3228.1036      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4156 ave        4156 max        4156 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4156
Ave neighs/atom = 51.95
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 9.083471282227173 y final 0 8.907133318393239 z final 0 28.277980078361423      xy final -4.541735641113585 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.0834713 10.879197 34.538803) with tilt (-5.5472884 0 -0)
  triclinic box = (0 0 0) to (9.0834713 8.9071333 34.538803) with tilt (-5.5472884 0 -0)
  triclinic box = (0 0 0) to (9.0834713 8.9071333 28.27798) with tilt (-5.5472884 0 -0)
  triclinic box = (0 0 0) to (9.0834713 8.9071333 28.27798) with tilt (-4.5417356 0 -0)
  triclinic box = (0 0 0) to (9.0834713 8.9071333 28.27798) with tilt (-4.5417356 0 -0)
  triclinic box = (0 0 0) to (9.0834713 8.9071333 28.27798) with tilt (-4.5417356 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.267 | 3.267 | 3.267 Mbytes
    PotEng          Pxx           E_mol          E_coul    
 16.916775      436202.24      0              0            
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
Nghost:            720 ave         720 max         720 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           7436 ave        7436 max        7436 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 7436
Ave neighs/atom = 92.95
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.026781884267134 y final 0 10.812718317080316 z final 0 34.32774858460155      xy final -5.513390942133566 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.026782 8.9071333 28.27798) with tilt (-4.5417356 0 0)
  triclinic box = (0 0 0) to (11.026782 10.812718 28.27798) with tilt (-4.5417356 0 0)
  triclinic box = (0 0 0) to (11.026782 10.812718 34.327749) with tilt (-4.5417356 0 0)
  triclinic box = (0 0 0) to (11.026782 10.812718 34.327749) with tilt (-5.5133909 0 0)
  triclinic box = (0 0 0) to (11.026782 10.812718 34.327749) with tilt (-5.5133909 0 0)
  triclinic box = (0 0 0) to (11.026782 10.812718 34.327749) with tilt (-5.5133909 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.85934     -385.17901      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4156 ave        4156 max        4156 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4156
Ave neighs/atom = 51.95
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.026781884267136 y final 0 10.812718317080316 z final 0 34.32774858460155      xy final -5.513390942133567 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.026782 10.812718 34.327749) with tilt (-5.5133909 0 0)
  triclinic box = (0 0 0) to (11.026782 10.812718 34.327749) with tilt (-5.5133909 0 0)
  triclinic box = (0 0 0) to (11.026782 10.812718 34.327749) with tilt (-5.5133909 0 0)
  triclinic box = (0 0 0) to (11.026782 10.812718 34.327749) with tilt (-5.5133909 0 0)
  triclinic box = (0 0 0) to (11.026782 10.812718 34.327749) with tilt (-5.5133909 0 0)
  triclinic box = (0 0 0) to (11.026782 10.812718 34.327749) with tilt (-5.5133909 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.85934     -385.17901      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4156 ave        4156 max        4156 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4156
Ave neighs/atom = 51.95
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.03489828550852 y final 0 10.820677154145619 z final 0 34.35301595491415      xy final -5.517449142754259 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.034898 10.812718 34.327749) with tilt (-5.5133909 0 0)
  triclinic box = (0 0 0) to (11.034898 10.820677 34.327749) with tilt (-5.5133909 0 0)
  triclinic box = (0 0 0) to (11.034898 10.820677 34.353016) with tilt (-5.5133909 0 0)
  triclinic box = (0 0 0) to (11.034898 10.820677 34.353016) with tilt (-5.5174491 0 0)
  triclinic box = (0 0 0) to (11.034898 10.820677 34.353016) with tilt (-5.5174491 0 0)
  triclinic box = (0 0 0) to (11.034898 10.820677 34.353016) with tilt (-5.5174491 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.89697     -599.10943      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4172 ave        4172 max        4172 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4172
Ave neighs/atom = 52.15
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.057161126715048 y final 0 10.842507805502523 z final 0 34.42232295887393      xy final -5.528580563357523 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.057161 10.820677 34.353016) with tilt (-5.5174491 0 0)
  triclinic box = (0 0 0) to (11.057161 10.842508 34.353016) with tilt (-5.5174491 0 0)
  triclinic box = (0 0 0) to (11.057161 10.842508 34.422323) with tilt (-5.5174491 0 0)
  triclinic box = (0 0 0) to (11.057161 10.842508 34.422323) with tilt (-5.5285806 0 0)
  triclinic box = (0 0 0) to (11.057161 10.842508 34.422323) with tilt (-5.5285806 0 0)
  triclinic box = (0 0 0) to (11.057161 10.842508 34.422323) with tilt (-5.5285806 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.93874     -1115.6183      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4180 ave        4180 max        4180 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4180
Ave neighs/atom = 52.25
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.057161126715048 y final 0 10.842507805502523 z final 0 34.42232295887393      xy final -5.528580563357523 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.057161 10.842508 34.422323) with tilt (-5.5285806 0 0)
  triclinic box = (0 0 0) to (11.057161 10.842508 34.422323) with tilt (-5.5285806 0 0)
  triclinic box = (0 0 0) to (11.057161 10.842508 34.422323) with tilt (-5.5285806 0 0)
  triclinic box = (0 0 0) to (11.057161 10.842508 34.422323) with tilt (-5.5285806 0 0)
  triclinic box = (0 0 0) to (11.057161 10.842508 34.422323) with tilt (-5.5285806 0 0)
  triclinic box = (0 0 0) to (11.057161 10.842508 34.422323) with tilt (-5.5285806 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.93874     -1115.6183      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4180 ave        4180 max        4180 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4180
Ave neighs/atom = 52.25
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.046337416022446 y final 0 10.831894216144084 z final 0 34.388627396260304      xy final -5.523168708011222 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.046337 10.842508 34.422323) with tilt (-5.5285806 0 0)
  triclinic box = (0 0 0) to (11.046337 10.831894 34.422323) with tilt (-5.5285806 0 0)
  triclinic box = (0 0 0) to (11.046337 10.831894 34.388627) with tilt (-5.5285806 0 0)
  triclinic box = (0 0 0) to (11.046337 10.831894 34.388627) with tilt (-5.5231687 0 0)
  triclinic box = (0 0 0) to (11.046337 10.831894 34.388627) with tilt (-5.5231687 0 0)
  triclinic box = (0 0 0) to (11.046337 10.831894 34.388627) with tilt (-5.5231687 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.94808     -644.81815      0              0            
Loop time of 1.542e-06 on 1 procs for 0 steps with 80 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.542e-06  |            |       |100.00

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4188 ave        4188 max        4188 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4188
Ave neighs/atom = 52.35
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.034296921081827 y final 0 10.820087464042015 z final 0 34.351143832379755      xy final -5.517148460540913 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.034297 10.831894 34.388627) with tilt (-5.5231687 0 0)
  triclinic box = (0 0 0) to (11.034297 10.820087 34.388627) with tilt (-5.5231687 0 0)
  triclinic box = (0 0 0) to (11.034297 10.820087 34.351144) with tilt (-5.5231687 0 0)
  triclinic box = (0 0 0) to (11.034297 10.820087 34.351144) with tilt (-5.5171485 0 0)
  triclinic box = (0 0 0) to (11.034297 10.820087 34.351144) with tilt (-5.5171485 0 0)
  triclinic box = (0 0 0) to (11.034297 10.820087 34.351144) with tilt (-5.5171485 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.95166     -109.38613      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.034296921081829 y final 0 10.820087464042015 z final 0 34.351143832379755      xy final -5.517148460540914 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.034297 10.820087 34.351144) with tilt (-5.5171485 0 0)
  triclinic box = (0 0 0) to (11.034297 10.820087 34.351144) with tilt (-5.5171485 0 0)
  triclinic box = (0 0 0) to (11.034297 10.820087 34.351144) with tilt (-5.5171485 0 0)
  triclinic box = (0 0 0) to (11.034297 10.820087 34.351144) with tilt (-5.5171485 0 0)
  triclinic box = (0 0 0) to (11.034297 10.820087 34.351144) with tilt (-5.5171485 0 0)
  triclinic box = (0 0 0) to (11.034297 10.820087 34.351144) with tilt (-5.5171485 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.95166     -109.38613      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.03993506825435 y final 0 10.825616157530883 z final 0 34.36869608839177      xy final -5.519967534127174 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.039935 10.820087 34.351144) with tilt (-5.5171485 0 0)
  triclinic box = (0 0 0) to (11.039935 10.825616 34.351144) with tilt (-5.5171485 0 0)
  triclinic box = (0 0 0) to (11.039935 10.825616 34.368696) with tilt (-5.5171485 0 0)
  triclinic box = (0 0 0) to (11.039935 10.825616 34.368696) with tilt (-5.5199675 0 0)
  triclinic box = (0 0 0) to (11.039935 10.825616 34.368696) with tilt (-5.5199675 0 0)
  triclinic box = (0 0 0) to (11.039935 10.825616 34.368696) with tilt (-5.5199675 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.95619     -368.44228      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.049936988894578 y final 0 10.835423910295711 z final 0 34.39983331598043      xy final -5.524968494447288 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.049937 10.825616 34.368696) with tilt (-5.5199675 0 0)
  triclinic box = (0 0 0) to (11.049937 10.835424 34.368696) with tilt (-5.5199675 0 0)
  triclinic box = (0 0 0) to (11.049937 10.835424 34.399833) with tilt (-5.5199675 0 0)
  triclinic box = (0 0 0) to (11.049937 10.835424 34.399833) with tilt (-5.5249685 0 0)
  triclinic box = (0 0 0) to (11.049937 10.835424 34.399833) with tilt (-5.5249685 0 0)
  triclinic box = (0 0 0) to (11.049937 10.835424 34.399833) with tilt (-5.5249685 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.95715     -815.80964      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.049936988894578 y final 0 10.835423910295711 z final 0 34.39983331598043      xy final -5.524968494447288 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.049937 10.835424 34.399833) with tilt (-5.5249685 0 0)
  triclinic box = (0 0 0) to (11.049937 10.835424 34.399833) with tilt (-5.5249685 0 0)
  triclinic box = (0 0 0) to (11.049937 10.835424 34.399833) with tilt (-5.5249685 0 0)
  triclinic box = (0 0 0) to (11.049937 10.835424 34.399833) with tilt (-5.5249685 0 0)
  triclinic box = (0 0 0) to (11.049937 10.835424 34.399833) with tilt (-5.5249685 0 0)
  triclinic box = (0 0 0) to (11.049937 10.835424 34.399833) with tilt (-5.5249685 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.95715     -815.80964      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.045820350782831 y final 0 10.831387188722466 z final 0 34.387017707619165      xy final -5.522910175391415 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.04582 10.835424 34.399833) with tilt (-5.5249685 0 0)
  triclinic box = (0 0 0) to (11.04582 10.831387 34.399833) with tilt (-5.5249685 0 0)
  triclinic box = (0 0 0) to (11.04582 10.831387 34.387018) with tilt (-5.5249685 0 0)
  triclinic box = (0 0 0) to (11.04582 10.831387 34.387018) with tilt (-5.5229102 0 0)
  triclinic box = (0 0 0) to (11.04582 10.831387 34.387018) with tilt (-5.5229102 0 0)
  triclinic box = (0 0 0) to (11.04582 10.831387 34.387018) with tilt (-5.5229102 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.95866     -642.18837      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.045820350782831 y final 0 10.831387188722466 z final 0 34.387017707619165      xy final -5.522910175391415 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.04582 10.831387 34.387018) with tilt (-5.5229102 0 0)
  triclinic box = (0 0 0) to (11.04582 10.831387 34.387018) with tilt (-5.5229102 0 0)
  triclinic box = (0 0 0) to (11.04582 10.831387 34.387018) with tilt (-5.5229102 0 0)
  triclinic box = (0 0 0) to (11.04582 10.831387 34.387018) with tilt (-5.5229102 0 0)
  triclinic box = (0 0 0) to (11.04582 10.831387 34.387018) with tilt (-5.5229102 0 0)
  triclinic box = (0 0 0) to (11.04582 10.831387 34.387018) with tilt (-5.5229102 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.95866     -642.18837      0              0            
Loop time of 1.417e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.043317606570236 y final 0 10.828933030430905 z final 0 34.37922635244381      xy final -5.521658803285117 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.043318 10.831387 34.387018) with tilt (-5.5229102 0 0)
  triclinic box = (0 0 0) to (11.043318 10.828933 34.387018) with tilt (-5.5229102 0 0)
  triclinic box = (0 0 0) to (11.043318 10.828933 34.379226) with tilt (-5.5229102 0 0)
  triclinic box = (0 0 0) to (11.043318 10.828933 34.379226) with tilt (-5.5216588 0 0)
  triclinic box = (0 0 0) to (11.043318 10.828933 34.379226) with tilt (-5.5216588 0 0)
  triclinic box = (0 0 0) to (11.043318 10.828933 34.379226) with tilt (-5.5216588 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.95954     -565.04096      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.040922631601495 y final 0 10.82658454925251 z final 0 34.37177049637843      xy final -5.5204613158007465 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.040923 10.828933 34.379226) with tilt (-5.5216588 0 0)
  triclinic box = (0 0 0) to (11.040923 10.826585 34.379226) with tilt (-5.5216588 0 0)
  triclinic box = (0 0 0) to (11.040923 10.826585 34.37177) with tilt (-5.5216588 0 0)
  triclinic box = (0 0 0) to (11.040923 10.826585 34.37177) with tilt (-5.5204613 0 0)
  triclinic box = (0 0 0) to (11.040923 10.826585 34.37177) with tilt (-5.5204613 0 0)
  triclinic box = (0 0 0) to (11.040923 10.826585 34.37177) with tilt (-5.5204613 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.95985     -490.3291       0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.040922631601497 y final 0 10.82658454925251 z final 0 34.37177049637843      xy final -5.520461315800747 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.040923 10.826585 34.37177) with tilt (-5.5204613 0 0)
  triclinic box = (0 0 0) to (11.040923 10.826585 34.37177) with tilt (-5.5204613 0 0)
  triclinic box = (0 0 0) to (11.040923 10.826585 34.37177) with tilt (-5.5204613 0 0)
  triclinic box = (0 0 0) to (11.040923 10.826585 34.37177) with tilt (-5.5204613 0 0)
  triclinic box = (0 0 0) to (11.040923 10.826585 34.37177) with tilt (-5.5204613 0 0)
  triclinic box = (0 0 0) to (11.040923 10.826585 34.37177) with tilt (-5.5204613 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.95985     -490.3291       0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.041387258387253 y final 0 10.827040156211066 z final 0 34.37321693756631      xy final -5.520693629193626 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.041387 10.826585 34.37177) with tilt (-5.5204613 0 0)
  triclinic box = (0 0 0) to (11.041387 10.82704 34.37177) with tilt (-5.5204613 0 0)
  triclinic box = (0 0 0) to (11.041387 10.82704 34.373217) with tilt (-5.5204613 0 0)
  triclinic box = (0 0 0) to (11.041387 10.82704 34.373217) with tilt (-5.5206936 0 0)
  triclinic box = (0 0 0) to (11.041387 10.82704 34.373217) with tilt (-5.5206936 0 0)
  triclinic box = (0 0 0) to (11.041387 10.82704 34.373217) with tilt (-5.5206936 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96044     -543.77095      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.042449003340762 y final 0 10.828081289447162 z final 0 34.37652228215438      xy final -5.52122450167038 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.042449 10.82704 34.373217) with tilt (-5.5206936 0 0)
  triclinic box = (0 0 0) to (11.042449 10.828081 34.373217) with tilt (-5.5206936 0 0)
  triclinic box = (0 0 0) to (11.042449 10.828081 34.376522) with tilt (-5.5206936 0 0)
  triclinic box = (0 0 0) to (11.042449 10.828081 34.376522) with tilt (-5.5212245 0 0)
  triclinic box = (0 0 0) to (11.042449 10.828081 34.376522) with tilt (-5.5212245 0 0)
  triclinic box = (0 0 0) to (11.042449 10.828081 34.376522) with tilt (-5.5212245 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96092     -664.27286      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.042449003340762 y final 0 10.828081289447162 z final 0 34.37652228215438      xy final -5.52122450167038 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.042449 10.828081 34.376522) with tilt (-5.5212245 0 0)
  triclinic box = (0 0 0) to (11.042449 10.828081 34.376522) with tilt (-5.5212245 0 0)
  triclinic box = (0 0 0) to (11.042449 10.828081 34.376522) with tilt (-5.5212245 0 0)
  triclinic box = (0 0 0) to (11.042449 10.828081 34.376522) with tilt (-5.5212245 0 0)
  triclinic box = (0 0 0) to (11.042449 10.828081 34.376522) with tilt (-5.5212245 0 0)
  triclinic box = (0 0 0) to (11.042449 10.828081 34.376522) with tilt (-5.5212245 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96092     -664.27286      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.041526119821523 y final 0 10.827176321920124 z final 0 34.3736492305466      xy final -5.52076305991076 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.041526 10.828081 34.376522) with tilt (-5.5212245 0 0)
  triclinic box = (0 0 0) to (11.041526 10.827176 34.376522) with tilt (-5.5212245 0 0)
  triclinic box = (0 0 0) to (11.041526 10.827176 34.373649) with tilt (-5.5212245 0 0)
  triclinic box = (0 0 0) to (11.041526 10.827176 34.373649) with tilt (-5.5207631 0 0)
  triclinic box = (0 0 0) to (11.041526 10.827176 34.373649) with tilt (-5.5207631 0 0)
  triclinic box = (0 0 0) to (11.041526 10.827176 34.373649) with tilt (-5.5207631 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96127     -651.32728      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4196 ave        4196 max        4196 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4196
Ave neighs/atom = 52.45
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.03830699971078 y final 0 10.824019694777986 z final 0 34.36362770776824      xy final -5.519153499855389 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.038307 10.827176 34.373649) with tilt (-5.5207631 0 0)
  triclinic box = (0 0 0) to (11.038307 10.82402 34.373649) with tilt (-5.5207631 0 0)
  triclinic box = (0 0 0) to (11.038307 10.82402 34.363628) with tilt (-5.5207631 0 0)
  triclinic box = (0 0 0) to (11.038307 10.82402 34.363628) with tilt (-5.5191535 0 0)
  triclinic box = (0 0 0) to (11.038307 10.82402 34.363628) with tilt (-5.5191535 0 0)
  triclinic box = (0 0 0) to (11.038307 10.82402 34.363628) with tilt (-5.5191535 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96182     -604.92843      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4212 ave        4212 max        4212 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4212
Ave neighs/atom = 52.65
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.03830699971078 y final 0 10.824019694777986 z final 0 34.36362770776824      xy final -5.519153499855389 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.038307 10.82402 34.363628) with tilt (-5.5191535 0 0)
  triclinic box = (0 0 0) to (11.038307 10.82402 34.363628) with tilt (-5.5191535 0 0)
  triclinic box = (0 0 0) to (11.038307 10.82402 34.363628) with tilt (-5.5191535 0 0)
  triclinic box = (0 0 0) to (11.038307 10.82402 34.363628) with tilt (-5.5191535 0 0)
  triclinic box = (0 0 0) to (11.038307 10.82402 34.363628) with tilt (-5.5191535 0 0)
  triclinic box = (0 0 0) to (11.038307 10.82402 34.363628) with tilt (-5.5191535 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96182     -604.92843      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4212 ave        4212 max        4212 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4212
Ave neighs/atom = 52.65
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.037984026752053 y final 0 10.823702991712358 z final 0 34.362622252627915      xy final -5.5189920133760255 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.037984 10.82402 34.363628) with tilt (-5.5191535 0 0)
  triclinic box = (0 0 0) to (11.037984 10.823703 34.363628) with tilt (-5.5191535 0 0)
  triclinic box = (0 0 0) to (11.037984 10.823703 34.362622) with tilt (-5.5191535 0 0)
  triclinic box = (0 0 0) to (11.037984 10.823703 34.362622) with tilt (-5.518992 0 0)
  triclinic box = (0 0 0) to (11.037984 10.823703 34.362622) with tilt (-5.518992 0 0)
  triclinic box = (0 0 0) to (11.037984 10.823703 34.362622) with tilt (-5.518992 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96208     -611.59067      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4212 ave        4212 max        4212 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4212
Ave neighs/atom = 52.65
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.03669222941395 y final 0 10.822436272112183 z final 0 34.358600726246955      xy final -5.518346114706974 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.036692 10.823703 34.362622) with tilt (-5.518992 0 0)
  triclinic box = (0 0 0) to (11.036692 10.822436 34.362622) with tilt (-5.518992 0 0)
  triclinic box = (0 0 0) to (11.036692 10.822436 34.358601) with tilt (-5.518992 0 0)
  triclinic box = (0 0 0) to (11.036692 10.822436 34.358601) with tilt (-5.5183461 0 0)
  triclinic box = (0 0 0) to (11.036692 10.822436 34.358601) with tilt (-5.5183461 0 0)
  triclinic box = (0 0 0) to (11.036692 10.822436 34.358601) with tilt (-5.5183461 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96277     -637.68871      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4212 ave        4212 max        4212 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4212
Ave neighs/atom = 52.65
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.036692229413951 y final 0 10.822436272112183 z final 0 34.358600726246955      xy final -5.518346114706975 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.036692 10.822436 34.358601) with tilt (-5.5183461 0 0)
  triclinic box = (0 0 0) to (11.036692 10.822436 34.358601) with tilt (-5.5183461 0 0)
  triclinic box = (0 0 0) to (11.036692 10.822436 34.358601) with tilt (-5.5183461 0 0)
  triclinic box = (0 0 0) to (11.036692 10.822436 34.358601) with tilt (-5.5183461 0 0)
  triclinic box = (0 0 0) to (11.036692 10.822436 34.358601) with tilt (-5.5183461 0 0)
  triclinic box = (0 0 0) to (11.036692 10.822436 34.358601) with tilt (-5.5183461 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96277     -637.68871      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4212 ave        4212 max        4212 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4212
Ave neighs/atom = 52.65
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.032724261044173 y final 0 10.818545334146439 z final 0 34.34624794535231      xy final -5.516362130522086 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.032724 10.822436 34.358601) with tilt (-5.5183461 0 0)
  triclinic box = (0 0 0) to (11.032724 10.818545 34.358601) with tilt (-5.5183461 0 0)
  triclinic box = (0 0 0) to (11.032724 10.818545 34.346248) with tilt (-5.5183461 0 0)
  triclinic box = (0 0 0) to (11.032724 10.818545 34.346248) with tilt (-5.5163621 0 0)
  triclinic box = (0 0 0) to (11.032724 10.818545 34.346248) with tilt (-5.5163621 0 0)
  triclinic box = (0 0 0) to (11.032724 10.818545 34.346248) with tilt (-5.5163621 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96343     -612.00872      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4252 ave        4252 max        4252 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4252
Ave neighs/atom = 53.15
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.032724261044175 y final 0 10.818545334146439 z final 0 34.34624794535231      xy final -5.5163621305220865 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.032724 10.818545 34.346248) with tilt (-5.5163621 0 0)
  triclinic box = (0 0 0) to (11.032724 10.818545 34.346248) with tilt (-5.5163621 0 0)
  triclinic box = (0 0 0) to (11.032724 10.818545 34.346248) with tilt (-5.5163621 0 0)
  triclinic box = (0 0 0) to (11.032724 10.818545 34.346248) with tilt (-5.5163621 0 0)
  triclinic box = (0 0 0) to (11.032724 10.818545 34.346248) with tilt (-5.5163621 0 0)
  triclinic box = (0 0 0) to (11.032724 10.818545 34.346248) with tilt (-5.5163621 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96343     -612.00872      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4252 ave        4252 max        4252 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4252
Ave neighs/atom = 53.15
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.029933276473344 y final 0 10.815808531123722 z final 0 34.337559261957956      xy final -5.514966638236671 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.029933 10.818545 34.346248) with tilt (-5.5163621 0 0)
  triclinic box = (0 0 0) to (11.029933 10.815809 34.346248) with tilt (-5.5163621 0 0)
  triclinic box = (0 0 0) to (11.029933 10.815809 34.337559) with tilt (-5.5163621 0 0)
  triclinic box = (0 0 0) to (11.029933 10.815809 34.337559) with tilt (-5.5149666 0 0)
  triclinic box = (0 0 0) to (11.029933 10.815809 34.337559) with tilt (-5.5149666 0 0)
  triclinic box = (0 0 0) to (11.029933 10.815809 34.337559) with tilt (-5.5149666 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96385     -588.99095      0              0            
Loop time of 7.09e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4252 ave        4252 max        4252 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4252
Ave neighs/atom = 53.15
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.02657291319405 y final 0 10.812513402774995 z final 0 34.327098031563146      xy final -5.513286456597024 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.026573 10.815809 34.337559) with tilt (-5.5149666 0 0)
  triclinic box = (0 0 0) to (11.026573 10.812513 34.337559) with tilt (-5.5149666 0 0)
  triclinic box = (0 0 0) to (11.026573 10.812513 34.327098) with tilt (-5.5149666 0 0)
  triclinic box = (0 0 0) to (11.026573 10.812513 34.327098) with tilt (-5.5132865 0 0)
  triclinic box = (0 0 0) to (11.026573 10.812513 34.327098) with tilt (-5.5132865 0 0)
  triclinic box = (0 0 0) to (11.026573 10.812513 34.327098) with tilt (-5.5132865 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96403     -560.90697      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4292 ave        4292 max        4292 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4292
Ave neighs/atom = 53.65
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.02657291319405 y final 0 10.812513402774995 z final 0 34.327098031563146      xy final -5.513286456597024 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.026573 10.812513 34.327098) with tilt (-5.5132865 0 0)
  triclinic box = (0 0 0) to (11.026573 10.812513 34.327098) with tilt (-5.5132865 0 0)
  triclinic box = (0 0 0) to (11.026573 10.812513 34.327098) with tilt (-5.5132865 0 0)
  triclinic box = (0 0 0) to (11.026573 10.812513 34.327098) with tilt (-5.5132865 0 0)
  triclinic box = (0 0 0) to (11.026573 10.812513 34.327098) with tilt (-5.5132865 0 0)
  triclinic box = (0 0 0) to (11.026573 10.812513 34.327098) with tilt (-5.5132865 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96403     -560.90697      0              0            
Loop time of 1.333e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4292 ave        4292 max        4292 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4292
Ave neighs/atom = 53.65
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.026358872797994 y final 0 10.812303517557774 z final 0 34.32643169708951      xy final -5.513179436398996 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.026359 10.812513 34.327098) with tilt (-5.5132865 0 0)
  triclinic box = (0 0 0) to (11.026359 10.812304 34.327098) with tilt (-5.5132865 0 0)
  triclinic box = (0 0 0) to (11.026359 10.812304 34.326432) with tilt (-5.5132865 0 0)
  triclinic box = (0 0 0) to (11.026359 10.812304 34.326432) with tilt (-5.5131794 0 0)
  triclinic box = (0 0 0) to (11.026359 10.812304 34.326432) with tilt (-5.5131794 0 0)
  triclinic box = (0 0 0) to (11.026359 10.812304 34.326432) with tilt (-5.5131794 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96438     -555.22005      0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4284 ave        4284 max        4284 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4284
Ave neighs/atom = 53.55
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.025823137571999 y final 0 10.811778182591324 z final 0 34.32476388645002      xy final -5.512911568785999 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.025823 10.812304 34.326432) with tilt (-5.5131794 0 0)
  triclinic box = (0 0 0) to (11.025823 10.811778 34.326432) with tilt (-5.5131794 0 0)
  triclinic box = (0 0 0) to (11.025823 10.811778 34.324764) with tilt (-5.5131794 0 0)
  triclinic box = (0 0 0) to (11.025823 10.811778 34.324764) with tilt (-5.5129116 0 0)
  triclinic box = (0 0 0) to (11.025823 10.811778 34.324764) with tilt (-5.5129116 0 0)
  triclinic box = (0 0 0) to (11.025823 10.811778 34.324764) with tilt (-5.5129116 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.96477     -540.21169      0              0            
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
Nghost:            530 ave         530 max         530 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4284 ave        4284 max        4284 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4284
Ave neighs/atom = 53.55
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.094576877780963 y final 0 12.21185247178527 z final 0 32.270516732811124      xy final -11.09457687778096 xz final 0.0 yz final -0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.094577 10.811778 34.324764) with tilt (-5.5129116 0 0)
  triclinic box = (0 0 0) to (11.094577 12.211852 34.324764) with tilt (-5.5129116 0 0)
  triclinic box = (0 0 0) to (11.094577 12.211852 32.270517) with tilt (-5.5129116 0 0)
  triclinic box = (0 0 0) to (11.094577 12.211852 32.270517) with tilt (-11.094577 0 0)
  triclinic box = (0 0 0) to (11.094577 12.211852 32.270517) with tilt (-11.094577 0 0)
  triclinic box = (0 0 0) to (11.094577 12.211852 32.270517) with tilt (-11.094577 0 -0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.32617     -3014.0332      0              0            
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
Nghost:            564 ave         564 max         564 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4024 ave        4024 max        4024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4024
Ave neighs/atom = 50.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 9.083471282227173 y final 0 9.998219170701969 z final 0 26.420864466870064      xy final -9.08347128222717 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.0834713 12.211852 32.270517) with tilt (-11.094577 0 -0)
  triclinic box = (0 0 0) to (9.0834713 9.9982192 32.270517) with tilt (-11.094577 0 -0)
  triclinic box = (0 0 0) to (9.0834713 9.9982192 26.420864) with tilt (-11.094577 0 -0)
  triclinic box = (0 0 0) to (9.0834713 9.9982192 26.420864) with tilt (-9.0834713 0 -0)
  triclinic box = (0 0 0) to (9.0834713 9.9982192 26.420864) with tilt (-9.0834713 0 -0)
  triclinic box = (0 0 0) to (9.0834713 9.9982192 26.420864) with tilt (-9.0834713 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
 16.436276      412360.69      0              0            
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
Nghost:            758 ave         758 max         758 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           7080 ave        7080 max        7080 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 7080
Ave neighs/atom = 88.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.025294551766265 y final 0 12.13559309267474 z final 0 32.068996973601294      xy final -11.025294551766263 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.025295 9.9982192 26.420864) with tilt (-9.0834713 0 0)
  triclinic box = (0 0 0) to (11.025295 12.135593 26.420864) with tilt (-9.0834713 0 0)
  triclinic box = (0 0 0) to (11.025295 12.135593 32.068997) with tilt (-9.0834713 0 0)
  triclinic box = (0 0 0) to (11.025295 12.135593 32.068997) with tilt (-11.025295 0 0)
  triclinic box = (0 0 0) to (11.025295 12.135593 32.068997) with tilt (-11.025295 0 0)
  triclinic box = (0 0 0) to (11.025295 12.135593 32.068997) with tilt (-11.025295 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.38936     -248.45514      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4024 ave        4024 max        4024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4024
Ave neighs/atom = 50.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.025294551766267 y final 0 12.13559309267474 z final 0 32.068996973601294      xy final -11.025294551766265 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.025295 12.135593 32.068997) with tilt (-11.025295 0 0)
  triclinic box = (0 0 0) to (11.025295 12.135593 32.068997) with tilt (-11.025295 0 0)
  triclinic box = (0 0 0) to (11.025295 12.135593 32.068997) with tilt (-11.025295 0 0)
  triclinic box = (0 0 0) to (11.025295 12.135593 32.068997) with tilt (-11.025295 0 0)
  triclinic box = (0 0 0) to (11.025295 12.135593 32.068997) with tilt (-11.025295 0 0)
  triclinic box = (0 0 0) to (11.025295 12.135593 32.068997) with tilt (-11.025295 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.38936     -248.45514      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4024 ave        4024 max        4024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4024
Ave neighs/atom = 50.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.032556112933227 y final 0 12.143585926891191 z final 0 32.09011849393742      xy final -11.032556112933225 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.032556 12.135593 32.068997) with tilt (-11.025295 0 0)
  triclinic box = (0 0 0) to (11.032556 12.143586 32.068997) with tilt (-11.025295 0 0)
  triclinic box = (0 0 0) to (11.032556 12.143586 32.090118) with tilt (-11.025295 0 0)
  triclinic box = (0 0 0) to (11.032556 12.143586 32.090118) with tilt (-11.032556 0 0)
  triclinic box = (0 0 0) to (11.032556 12.143586 32.090118) with tilt (-11.032556 0 0)
  triclinic box = (0 0 0) to (11.032556 12.143586 32.090118) with tilt (-11.032556 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.41255     -437.08501      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4024 ave        4024 max        4024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4024
Ave neighs/atom = 50.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.052212164553916 y final 0 12.165221434510267 z final 0 32.147291556931535      xy final -11.052212164553914 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.052212 12.143586 32.090118) with tilt (-11.032556 0 0)
  triclinic box = (0 0 0) to (11.052212 12.165221 32.090118) with tilt (-11.032556 0 0)
  triclinic box = (0 0 0) to (11.052212 12.165221 32.147292) with tilt (-11.032556 0 0)
  triclinic box = (0 0 0) to (11.052212 12.165221 32.147292) with tilt (-11.052212 0 0)
  triclinic box = (0 0 0) to (11.052212 12.165221 32.147292) with tilt (-11.052212 0 0)
  triclinic box = (0 0 0) to (11.052212 12.165221 32.147292) with tilt (-11.052212 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.43862     -904.65167      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4024 ave        4024 max        4024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4024
Ave neighs/atom = 50.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.052212164553916 y final 0 12.165221434510267 z final 0 32.147291556931535      xy final -11.052212164553914 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.052212 12.165221 32.147292) with tilt (-11.052212 0 0)
  triclinic box = (0 0 0) to (11.052212 12.165221 32.147292) with tilt (-11.052212 0 0)
  triclinic box = (0 0 0) to (11.052212 12.165221 32.147292) with tilt (-11.052212 0 0)
  triclinic box = (0 0 0) to (11.052212 12.165221 32.147292) with tilt (-11.052212 0 0)
  triclinic box = (0 0 0) to (11.052212 12.165221 32.147292) with tilt (-11.052212 0 0)
  triclinic box = (0 0 0) to (11.052212 12.165221 32.147292) with tilt (-11.052212 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.43862     -904.65167      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4024 ave        4024 max        4024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4024
Ave neighs/atom = 50.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.04162843985998 y final 0 12.153571879417887 z final 0 32.116506943097804      xy final -11.041628439859979 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.041628 12.165221 32.147292) with tilt (-11.052212 0 0)
  triclinic box = (0 0 0) to (11.041628 12.153572 32.147292) with tilt (-11.052212 0 0)
  triclinic box = (0 0 0) to (11.041628 12.153572 32.116507) with tilt (-11.052212 0 0)
  triclinic box = (0 0 0) to (11.041628 12.153572 32.116507) with tilt (-11.041628 0 0)
  triclinic box = (0 0 0) to (11.041628 12.153572 32.116507) with tilt (-11.041628 0 0)
  triclinic box = (0 0 0) to (11.041628 12.153572 32.116507) with tilt (-11.041628 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.44715     -437.44766      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4056 ave        4056 max        4056 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4056
Ave neighs/atom = 50.7
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.031172473420167 y final 0 12.142062948431192 z final 0 32.08609393648468      xy final -11.031172473420165 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.031172 12.153572 32.116507) with tilt (-11.041628 0 0)
  triclinic box = (0 0 0) to (11.031172 12.142063 32.116507) with tilt (-11.041628 0 0)
  triclinic box = (0 0 0) to (11.031172 12.142063 32.086094) with tilt (-11.041628 0 0)
  triclinic box = (0 0 0) to (11.031172 12.142063 32.086094) with tilt (-11.031172 0 0)
  triclinic box = (0 0 0) to (11.031172 12.142063 32.086094) with tilt (-11.031172 0 0)
  triclinic box = (0 0 0) to (11.031172 12.142063 32.086094) with tilt (-11.031172 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.44993      35.473756      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4088 ave        4088 max        4088 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4088
Ave neighs/atom = 51.1
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.031172473420169 y final 0 12.142062948431192 z final 0 32.08609393648468      xy final -11.031172473420167 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.031172 12.142063 32.086094) with tilt (-11.031172 0 0)
  triclinic box = (0 0 0) to (11.031172 12.142063 32.086094) with tilt (-11.031172 0 0)
  triclinic box = (0 0 0) to (11.031172 12.142063 32.086094) with tilt (-11.031172 0 0)
  triclinic box = (0 0 0) to (11.031172 12.142063 32.086094) with tilt (-11.031172 0 0)
  triclinic box = (0 0 0) to (11.031172 12.142063 32.086094) with tilt (-11.031172 0 0)
  triclinic box = (0 0 0) to (11.031172 12.142063 32.086094) with tilt (-11.031172 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.44993      35.473756      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4088 ave        4088 max        4088 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4088
Ave neighs/atom = 51.1
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.034396135855742 y final 0 12.145611248696785 z final 0 32.095470522335056      xy final -11.03439613585574 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.034396 12.142063 32.086094) with tilt (-11.031172 0 0)
  triclinic box = (0 0 0) to (11.034396 12.145611 32.086094) with tilt (-11.031172 0 0)
  triclinic box = (0 0 0) to (11.034396 12.145611 32.095471) with tilt (-11.031172 0 0)
  triclinic box = (0 0 0) to (11.034396 12.145611 32.095471) with tilt (-11.034396 0 0)
  triclinic box = (0 0 0) to (11.034396 12.145611 32.095471) with tilt (-11.034396 0 0)
  triclinic box = (0 0 0) to (11.034396 12.145611 32.095471) with tilt (-11.034396 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.45402     -127.27595      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4072 ave        4072 max        4072 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4072
Ave neighs/atom = 50.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.0473002089271 y final 0 12.159814822061366 z final 0 32.133004275136805      xy final -11.047300208927098 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.0473 12.145611 32.095471) with tilt (-11.034396 0 0)
  triclinic box = (0 0 0) to (11.0473 12.159815 32.095471) with tilt (-11.034396 0 0)
  triclinic box = (0 0 0) to (11.0473 12.159815 32.133004) with tilt (-11.034396 0 0)
  triclinic box = (0 0 0) to (11.0473 12.159815 32.133004) with tilt (-11.0473 0 0)
  triclinic box = (0 0 0) to (11.0473 12.159815 32.133004) with tilt (-11.0473 0 0)
  triclinic box = (0 0 0) to (11.0473 12.159815 32.133004) with tilt (-11.0473 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.44893     -737.8537       0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4024 ave        4024 max        4024 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4024
Ave neighs/atom = 50.3
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.038882067215008 y final 0 12.150548934249048 z final 0 32.108518638057376      xy final -11.038882067215006 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.038882 12.159815 32.133004) with tilt (-11.0473 0 0)
  triclinic box = (0 0 0) to (11.038882 12.150549 32.133004) with tilt (-11.0473 0 0)
  triclinic box = (0 0 0) to (11.038882 12.150549 32.108519) with tilt (-11.0473 0 0)
  triclinic box = (0 0 0) to (11.038882 12.150549 32.108519) with tilt (-11.038882 0 0)
  triclinic box = (0 0 0) to (11.038882 12.150549 32.108519) with tilt (-11.038882 0 0)
  triclinic box = (0 0 0) to (11.038882 12.150549 32.108519) with tilt (-11.038882 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.45615     -346.85994      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4040 ave        4040 max        4040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4040
Ave neighs/atom = 50.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.038882067215008 y final 0 12.150548934249048 z final 0 32.108518638057376      xy final -11.038882067215006 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.038882 12.150549 32.108519) with tilt (-11.038882 0 0)
  triclinic box = (0 0 0) to (11.038882 12.150549 32.108519) with tilt (-11.038882 0 0)
  triclinic box = (0 0 0) to (11.038882 12.150549 32.108519) with tilt (-11.038882 0 0)
  triclinic box = (0 0 0) to (11.038882 12.150549 32.108519) with tilt (-11.038882 0 0)
  triclinic box = (0 0 0) to (11.038882 12.150549 32.108519) with tilt (-11.038882 0 0)
  triclinic box = (0 0 0) to (11.038882 12.150549 32.108519) with tilt (-11.038882 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.45615     -346.85994      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4040 ave        4040 max        4040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4040
Ave neighs/atom = 50.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.038477379133738 y final 0 12.150103492192088 z final 0 32.1073415320134      xy final -11.038477379133736 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.038477 12.150549 32.108519) with tilt (-11.038882 0 0)
  triclinic box = (0 0 0) to (11.038477 12.150103 32.108519) with tilt (-11.038882 0 0)
  triclinic box = (0 0 0) to (11.038477 12.150103 32.107342) with tilt (-11.038882 0 0)
  triclinic box = (0 0 0) to (11.038477 12.150103 32.107342) with tilt (-11.038477 0 0)
  triclinic box = (0 0 0) to (11.038477 12.150103 32.107342) with tilt (-11.038477 0 0)
  triclinic box = (0 0 0) to (11.038477 12.150103 32.107342) with tilt (-11.038477 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.45735     -341.59811      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4040 ave        4040 max        4040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4040
Ave neighs/atom = 50.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.036858775162852 y final 0 12.148321887258389 z final 0 32.1026335393516      xy final -11.03685877516285 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.036859 12.150103 32.107342) with tilt (-11.038477 0 0)
  triclinic box = (0 0 0) to (11.036859 12.148322 32.107342) with tilt (-11.038477 0 0)
  triclinic box = (0 0 0) to (11.036859 12.148322 32.102634) with tilt (-11.038477 0 0)
  triclinic box = (0 0 0) to (11.036859 12.148322 32.102634) with tilt (-11.036859 0 0)
  triclinic box = (0 0 0) to (11.036859 12.148322 32.102634) with tilt (-11.036859 0 0)
  triclinic box = (0 0 0) to (11.036859 12.148322 32.102634) with tilt (-11.036859 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.45925     -313.88696      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4040 ave        4040 max        4040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4040
Ave neighs/atom = 50.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.036858775162852 y final 0 12.148321887258389 z final 0 32.1026335393516      xy final -11.03685877516285 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.036859 12.148322 32.102634) with tilt (-11.036859 0 0)
  triclinic box = (0 0 0) to (11.036859 12.148322 32.102634) with tilt (-11.036859 0 0)
  triclinic box = (0 0 0) to (11.036859 12.148322 32.102634) with tilt (-11.036859 0 0)
  triclinic box = (0 0 0) to (11.036859 12.148322 32.102634) with tilt (-11.036859 0 0)
  triclinic box = (0 0 0) to (11.036859 12.148322 32.102634) with tilt (-11.036859 0 0)
  triclinic box = (0 0 0) to (11.036859 12.148322 32.102634) with tilt (-11.036859 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.45925     -313.88696      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4040 ave        4040 max        4040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4040
Ave neighs/atom = 50.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.034698111266861 y final 0 12.145943634439051 z final 0 32.096348870618606      xy final -11.03469811126686 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.034698 12.148322 32.102634) with tilt (-11.036859 0 0)
  triclinic box = (0 0 0) to (11.034698 12.145944 32.102634) with tilt (-11.036859 0 0)
  triclinic box = (0 0 0) to (11.034698 12.145944 32.096349) with tilt (-11.036859 0 0)
  triclinic box = (0 0 0) to (11.034698 12.145944 32.096349) with tilt (-11.034698 0 0)
  triclinic box = (0 0 0) to (11.034698 12.145944 32.096349) with tilt (-11.034698 0 0)
  triclinic box = (0 0 0) to (11.034698 12.145944 32.096349) with tilt (-11.034698 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.45981     -275.95124      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4040 ave        4040 max        4040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4040
Ave neighs/atom = 50.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.034698111266863 y final 0 12.145943634439051 z final 0 32.096348870618606      xy final -11.034698111266861 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.034698 12.145944 32.096349) with tilt (-11.034698 0 0)
  triclinic box = (0 0 0) to (11.034698 12.145944 32.096349) with tilt (-11.034698 0 0)
  triclinic box = (0 0 0) to (11.034698 12.145944 32.096349) with tilt (-11.034698 0 0)
  triclinic box = (0 0 0) to (11.034698 12.145944 32.096349) with tilt (-11.034698 0 0)
  triclinic box = (0 0 0) to (11.034698 12.145944 32.096349) with tilt (-11.034698 0 0)
  triclinic box = (0 0 0) to (11.034698 12.145944 32.096349) with tilt (-11.034698 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.45981     -275.95124      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4040 ave        4040 max        4040 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4040
Ave neighs/atom = 50.5
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.032416506561908 y final 0 12.143432261507684 z final 0 32.089712424396275      xy final -11.032416506561907 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.032417 12.145944 32.096349) with tilt (-11.034698 0 0)
  triclinic box = (0 0 0) to (11.032417 12.143432 32.096349) with tilt (-11.034698 0 0)
  triclinic box = (0 0 0) to (11.032417 12.143432 32.089712) with tilt (-11.034698 0 0)
  triclinic box = (0 0 0) to (11.032417 12.143432 32.089712) with tilt (-11.032417 0 0)
  triclinic box = (0 0 0) to (11.032417 12.143432 32.089712) with tilt (-11.032417 0 0)
  triclinic box = (0 0 0) to (11.032417 12.143432 32.089712) with tilt (-11.032417 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46031     -247.25972      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4072 ave        4072 max        4072 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4072
Ave neighs/atom = 50.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.02964912352695 y final 0 12.140386190104678 z final 0 32.081663006962906      xy final -11.029649123526948 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.029649 12.143432 32.089712) with tilt (-11.032417 0 0)
  triclinic box = (0 0 0) to (11.029649 12.140386 32.089712) with tilt (-11.032417 0 0)
  triclinic box = (0 0 0) to (11.029649 12.140386 32.081663) with tilt (-11.032417 0 0)
  triclinic box = (0 0 0) to (11.029649 12.140386 32.081663) with tilt (-11.029649 0 0)
  triclinic box = (0 0 0) to (11.029649 12.140386 32.081663) with tilt (-11.029649 0 0)
  triclinic box = (0 0 0) to (11.029649 12.140386 32.081663) with tilt (-11.029649 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46052     -211.83659      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4072 ave        4072 max        4072 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4072
Ave neighs/atom = 50.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.02964912352695 y final 0 12.140386190104678 z final 0 32.081663006962906      xy final -11.029649123526948 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.029649 12.140386 32.081663) with tilt (-11.029649 0 0)
  triclinic box = (0 0 0) to (11.029649 12.140386 32.081663) with tilt (-11.029649 0 0)
  triclinic box = (0 0 0) to (11.029649 12.140386 32.081663) with tilt (-11.029649 0 0)
  triclinic box = (0 0 0) to (11.029649 12.140386 32.081663) with tilt (-11.029649 0 0)
  triclinic box = (0 0 0) to (11.029649 12.140386 32.081663) with tilt (-11.029649 0 0)
  triclinic box = (0 0 0) to (11.029649 12.140386 32.081663) with tilt (-11.029649 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46052     -211.83659      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4072 ave        4072 max        4072 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4072
Ave neighs/atom = 50.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.028882429181547 y final 0 12.13954228606578 z final 0 32.07943294240299      xy final -11.028882429181545 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.028882 12.140386 32.081663) with tilt (-11.029649 0 0)
  triclinic box = (0 0 0) to (11.028882 12.139542 32.081663) with tilt (-11.029649 0 0)
  triclinic box = (0 0 0) to (11.028882 12.139542 32.079433) with tilt (-11.029649 0 0)
  triclinic box = (0 0 0) to (11.028882 12.139542 32.079433) with tilt (-11.028882 0 0)
  triclinic box = (0 0 0) to (11.028882 12.139542 32.079433) with tilt (-11.028882 0 0)
  triclinic box = (0 0 0) to (11.028882 12.139542 32.079433) with tilt (-11.028882 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46111     -218.10162      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4088 ave        4088 max        4088 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4088
Ave neighs/atom = 51.1
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.025816184708416 y final 0 12.136167256485036 z final 0 32.070514234220845      xy final -11.025816184708415 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.025816 12.139542 32.079433) with tilt (-11.028882 0 0)
  triclinic box = (0 0 0) to (11.025816 12.136167 32.079433) with tilt (-11.028882 0 0)
  triclinic box = (0 0 0) to (11.025816 12.136167 32.070514) with tilt (-11.028882 0 0)
  triclinic box = (0 0 0) to (11.025816 12.136167 32.070514) with tilt (-11.025816 0 0)
  triclinic box = (0 0 0) to (11.025816 12.136167 32.070514) with tilt (-11.025816 0 0)
  triclinic box = (0 0 0) to (11.025816 12.136167 32.070514) with tilt (-11.025816 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46254     -242.22827      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4152 ave        4152 max        4152 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4152
Ave neighs/atom = 51.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.025816184708416 y final 0 12.136167256485036 z final 0 32.070514234220845      xy final -11.025816184708415 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.025816 12.136167 32.070514) with tilt (-11.025816 0 0)
  triclinic box = (0 0 0) to (11.025816 12.136167 32.070514) with tilt (-11.025816 0 0)
  triclinic box = (0 0 0) to (11.025816 12.136167 32.070514) with tilt (-11.025816 0 0)
  triclinic box = (0 0 0) to (11.025816 12.136167 32.070514) with tilt (-11.025816 0 0)
  triclinic box = (0 0 0) to (11.025816 12.136167 32.070514) with tilt (-11.025816 0 0)
  triclinic box = (0 0 0) to (11.025816 12.136167 32.070514) with tilt (-11.025816 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46254     -242.22827      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4152 ave        4152 max        4152 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4152
Ave neighs/atom = 51.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.022798357790672 y final 0 12.132845520333138 z final 0 32.06173635696433      xy final -11.02279835779067 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.022798 12.136167 32.070514) with tilt (-11.025816 0 0)
  triclinic box = (0 0 0) to (11.022798 12.132846 32.070514) with tilt (-11.025816 0 0)
  triclinic box = (0 0 0) to (11.022798 12.132846 32.061736) with tilt (-11.025816 0 0)
  triclinic box = (0 0 0) to (11.022798 12.132846 32.061736) with tilt (-11.022798 0 0)
  triclinic box = (0 0 0) to (11.022798 12.132846 32.061736) with tilt (-11.022798 0 0)
  triclinic box = (0 0 0) to (11.022798 12.132846 32.061736) with tilt (-11.022798 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46332     -235.94941      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4168 ave        4168 max        4168 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4168
Ave neighs/atom = 52.1
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.018952256275082 y final 0 12.128612098471507 z final 0 32.050549298216396      xy final -11.01895225627508 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.018952 12.132846 32.061736) with tilt (-11.022798 0 0)
  triclinic box = (0 0 0) to (11.018952 12.128612 32.061736) with tilt (-11.022798 0 0)
  triclinic box = (0 0 0) to (11.018952 12.128612 32.050549) with tilt (-11.022798 0 0)
  triclinic box = (0 0 0) to (11.018952 12.128612 32.050549) with tilt (-11.018952 0 0)
  triclinic box = (0 0 0) to (11.018952 12.128612 32.050549) with tilt (-11.018952 0 0)
  triclinic box = (0 0 0) to (11.018952 12.128612 32.050549) with tilt (-11.018952 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46365     -227.36417      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4152 ave        4152 max        4152 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4152
Ave neighs/atom = 51.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.018952256275082 y final 0 12.128612098471507 z final 0 32.050549298216396      xy final -11.018952256275082 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.018952 12.128612 32.050549) with tilt (-11.018952 0 0)
  triclinic box = (0 0 0) to (11.018952 12.128612 32.050549) with tilt (-11.018952 0 0)
  triclinic box = (0 0 0) to (11.018952 12.128612 32.050549) with tilt (-11.018952 0 0)
  triclinic box = (0 0 0) to (11.018952 12.128612 32.050549) with tilt (-11.018952 0 0)
  triclinic box = (0 0 0) to (11.018952 12.128612 32.050549) with tilt (-11.018952 0 0)
  triclinic box = (0 0 0) to (11.018952 12.128612 32.050549) with tilt (-11.018952 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46365     -227.36417      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4152 ave        4152 max        4152 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4152
Ave neighs/atom = 51.9
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.017748232344344 y final 0 12.12728682372001 z final 0 32.04704718409057      xy final -11.017748232344342 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.017748 12.128612 32.050549) with tilt (-11.018952 0 0)
  triclinic box = (0 0 0) to (11.017748 12.127287 32.050549) with tilt (-11.018952 0 0)
  triclinic box = (0 0 0) to (11.017748 12.127287 32.047047) with tilt (-11.018952 0 0)
  triclinic box = (0 0 0) to (11.017748 12.127287 32.047047) with tilt (-11.017748 0 0)
  triclinic box = (0 0 0) to (11.017748 12.127287 32.047047) with tilt (-11.017748 0 0)
  triclinic box = (0 0 0) to (11.017748 12.127287 32.047047) with tilt (-11.017748 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46426     -207.38883      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4168 ave        4168 max        4168 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4168
Ave neighs/atom = 52.1
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.015522717887503 y final 0 12.124837189586128 z final 0 32.040573886162235      xy final -11.015522717887501 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.015523 12.127287 32.047047) with tilt (-11.017748 0 0)
  triclinic box = (0 0 0) to (11.015523 12.124837 32.047047) with tilt (-11.017748 0 0)
  triclinic box = (0 0 0) to (11.015523 12.124837 32.040574) with tilt (-11.017748 0 0)
  triclinic box = (0 0 0) to (11.015523 12.124837 32.040574) with tilt (-11.015523 0 0)
  triclinic box = (0 0 0) to (11.015523 12.124837 32.040574) with tilt (-11.015523 0 0)
  triclinic box = (0 0 0) to (11.015523 12.124837 32.040574) with tilt (-11.015523 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.4647      -168.74156      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4168 ave        4168 max        4168 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4168
Ave neighs/atom = 52.1
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.015522717887503 y final 0 12.124837189586128 z final 0 32.040573886162235      xy final -11.015522717887501 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.015523 12.124837 32.040574) with tilt (-11.015523 0 0)
  triclinic box = (0 0 0) to (11.015523 12.124837 32.040574) with tilt (-11.015523 0 0)
  triclinic box = (0 0 0) to (11.015523 12.124837 32.040574) with tilt (-11.015523 0 0)
  triclinic box = (0 0 0) to (11.015523 12.124837 32.040574) with tilt (-11.015523 0 0)
  triclinic box = (0 0 0) to (11.015523 12.124837 32.040574) with tilt (-11.015523 0 0)
  triclinic box = (0 0 0) to (11.015523 12.124837 32.040574) with tilt (-11.015523 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.4647      -168.74156      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4168 ave        4168 max        4168 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4168
Ave neighs/atom = 52.1
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  80 settings made for x
  80 settings made for y
  80 settings made for z
change_box all     x final 0 11.014960221682934 y final 0 12.12421804739244 z final 0 32.03893776759893      xy final -11.014960221682932 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.01496 12.124837 32.040574) with tilt (-11.015523 0 0)
  triclinic box = (0 0 0) to (11.01496 12.124218 32.040574) with tilt (-11.015523 0 0)
  triclinic box = (0 0 0) to (11.01496 12.124218 32.038938) with tilt (-11.015523 0 0)
  triclinic box = (0 0 0) to (11.01496 12.124218 32.038938) with tilt (-11.01496 0 0)
  triclinic box = (0 0 0) to (11.01496 12.124218 32.038938) with tilt (-11.01496 0 0)
  triclinic box = (0 0 0) to (11.01496 12.124218 32.038938) with tilt (-11.01496 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46487     -158.22023      0              0            
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
Nghost:            576 ave         576 max         576 min
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
change_box all     x final 0 11.013054281265312 y final 0 12.12212016989279 z final 0 32.033394006641096      xy final -11.01305428126531 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.013054 12.124218 32.038938) with tilt (-11.01496 0 0)
  triclinic box = (0 0 0) to (11.013054 12.12212 32.038938) with tilt (-11.01496 0 0)
  triclinic box = (0 0 0) to (11.013054 12.12212 32.033394) with tilt (-11.01496 0 0)
  triclinic box = (0 0 0) to (11.013054 12.12212 32.033394) with tilt (-11.013054 0 0)
  triclinic box = (0 0 0) to (11.013054 12.12212 32.033394) with tilt (-11.013054 0 0)
  triclinic box = (0 0 0) to (11.013054 12.12212 32.033394) with tilt (-11.013054 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46516     -121.93351      0              0            
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
Nghost:            576 ave         576 max         576 min
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
change_box all     x final 0 11.013054281265314 y final 0 12.12212016989279 z final 0 32.033394006641096      xy final -11.013054281265312 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.013054 12.12212 32.033394) with tilt (-11.013054 0 0)
  triclinic box = (0 0 0) to (11.013054 12.12212 32.033394) with tilt (-11.013054 0 0)
  triclinic box = (0 0 0) to (11.013054 12.12212 32.033394) with tilt (-11.013054 0 0)
  triclinic box = (0 0 0) to (11.013054 12.12212 32.033394) with tilt (-11.013054 0 0)
  triclinic box = (0 0 0) to (11.013054 12.12212 32.033394) with tilt (-11.013054 0 0)
  triclinic box = (0 0 0) to (11.013054 12.12212 32.033394) with tilt (-11.013054 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46516     -121.93351      0              0            
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
Nghost:            576 ave         576 max         576 min
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
change_box all     x final 0 11.012923010395005 y final 0 12.1219756794341 z final 0 32.03301218236213      xy final -11.012923010395003 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.012923 12.12212 32.033394) with tilt (-11.013054 0 0)
  triclinic box = (0 0 0) to (11.012923 12.121976 32.033394) with tilt (-11.013054 0 0)
  triclinic box = (0 0 0) to (11.012923 12.121976 32.033012) with tilt (-11.013054 0 0)
  triclinic box = (0 0 0) to (11.012923 12.121976 32.033012) with tilt (-11.012923 0 0)
  triclinic box = (0 0 0) to (11.012923 12.121976 32.033012) with tilt (-11.012923 0 0)
  triclinic box = (0 0 0) to (11.012923 12.121976 32.033012) with tilt (-11.012923 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46521     -124.24254      0              0            
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
Nghost:            576 ave         576 max         576 min
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
change_box all     x final 0 11.012588403257535 y final 0 12.121607375798565 z final 0 32.032038918996825      xy final -11.012588403257533 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.012588 12.121976 32.033012) with tilt (-11.012923 0 0)
  triclinic box = (0 0 0) to (11.012588 12.121607 32.033012) with tilt (-11.012923 0 0)
  triclinic box = (0 0 0) to (11.012588 12.121607 32.032039) with tilt (-11.012923 0 0)
  triclinic box = (0 0 0) to (11.012588 12.121607 32.032039) with tilt (-11.012588 0 0)
  triclinic box = (0 0 0) to (11.012588 12.121607 32.032039) with tilt (-11.012588 0 0)
  triclinic box = (0 0 0) to (11.012588 12.121607 32.032039) with tilt (-11.012588 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46527     -130.02329      0              0            
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
Nghost:            576 ave         576 max         576 min
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
change_box all     x final 0 11.012588403257535 y final 0 12.121607375798565 z final 0 32.032038918996825      xy final -11.012588403257533 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.012588 12.121607 32.032039) with tilt (-11.012588 0 0)
  triclinic box = (0 0 0) to (11.012588 12.121607 32.032039) with tilt (-11.012588 0 0)
  triclinic box = (0 0 0) to (11.012588 12.121607 32.032039) with tilt (-11.012588 0 0)
  triclinic box = (0 0 0) to (11.012588 12.121607 32.032039) with tilt (-11.012588 0 0)
  triclinic box = (0 0 0) to (11.012588 12.121607 32.032039) with tilt (-11.012588 0 0)
  triclinic box = (0 0 0) to (11.012588 12.121607 32.032039) with tilt (-11.012588 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46527     -130.02329      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            576 ave         576 max         576 min
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
change_box all     x final 0 11.012264896295953 y final 0 12.12125129017836 z final 0 32.031097942433966      xy final -11.012264896295951 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.012265 12.121607 32.032039) with tilt (-11.012588 0 0)
  triclinic box = (0 0 0) to (11.012265 12.121251 32.032039) with tilt (-11.012588 0 0)
  triclinic box = (0 0 0) to (11.012265 12.121251 32.031098) with tilt (-11.012588 0 0)
  triclinic box = (0 0 0) to (11.012265 12.121251 32.031098) with tilt (-11.012265 0 0)
  triclinic box = (0 0 0) to (11.012265 12.121251 32.031098) with tilt (-11.012265 0 0)
  triclinic box = (0 0 0) to (11.012265 12.121251 32.031098) with tilt (-11.012265 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46529     -108.57952      0              0            
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
Nghost:            576 ave         576 max         576 min
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
change_box all     x final 0 11.011931456097372 y final 0 12.1208842709979 z final 0 32.03012807331446      xy final -11.01193145609737 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.011931 12.121251 32.031098) with tilt (-11.012265 0 0)
  triclinic box = (0 0 0) to (11.011931 12.120884 32.031098) with tilt (-11.012265 0 0)
  triclinic box = (0 0 0) to (11.011931 12.120884 32.030128) with tilt (-11.012265 0 0)
  triclinic box = (0 0 0) to (11.011931 12.120884 32.030128) with tilt (-11.011931 0 0)
  triclinic box = (0 0 0) to (11.011931 12.120884 32.030128) with tilt (-11.011931 0 0)
  triclinic box = (0 0 0) to (11.011931 12.120884 32.030128) with tilt (-11.011931 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.4653      -86.426931      0              0            
Loop time of 1.542e-06 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            576 ave         576 max         576 min
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
change_box all     x final 0 11.011931456097372 y final 0 12.1208842709979 z final 0 32.03012807331446      xy final -11.01193145609737 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.011931 12.120884 32.030128) with tilt (-11.011931 0 0)
  triclinic box = (0 0 0) to (11.011931 12.120884 32.030128) with tilt (-11.011931 0 0)
  triclinic box = (0 0 0) to (11.011931 12.120884 32.030128) with tilt (-11.011931 0 0)
  triclinic box = (0 0 0) to (11.011931 12.120884 32.030128) with tilt (-11.011931 0 0)
  triclinic box = (0 0 0) to (11.011931 12.120884 32.030128) with tilt (-11.011931 0 0)
  triclinic box = (0 0 0) to (11.011931 12.120884 32.030128) with tilt (-11.011931 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.4653      -86.426931      0              0            
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
Nghost:            576 ave         576 max         576 min
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
change_box all     x final 0 11.011892918118752 y final 0 12.12084185206524 z final 0 32.03001597886528      xy final -11.01189291811875 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.011893 12.120884 32.030128) with tilt (-11.011931 0 0)
  triclinic box = (0 0 0) to (11.011893 12.120842 32.030128) with tilt (-11.011931 0 0)
  triclinic box = (0 0 0) to (11.011893 12.120842 32.030016) with tilt (-11.011931 0 0)
  triclinic box = (0 0 0) to (11.011893 12.120842 32.030016) with tilt (-11.011893 0 0)
  triclinic box = (0 0 0) to (11.011893 12.120842 32.030016) with tilt (-11.011893 0 0)
  triclinic box = (0 0 0) to (11.011893 12.120842 32.030016) with tilt (-11.011893 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46532     -84.69985       0              0            
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
Nghost:            576 ave         576 max         576 min
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
change_box all     x final 0 11.01173876755296 y final 0 12.120672177819106 z final 0 32.029567604991485      xy final -11.011738767552957 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (11.011739 12.120842 32.030016) with tilt (-11.011893 0 0)
  triclinic box = (0 0 0) to (11.011739 12.120672 32.030016) with tilt (-11.011893 0 0)
  triclinic box = (0 0 0) to (11.011739 12.120672 32.029568) with tilt (-11.011893 0 0)
  triclinic box = (0 0 0) to (11.011739 12.120672 32.029568) with tilt (-11.011739 0 0)
  triclinic box = (0 0 0) to (11.011739 12.120672 32.029568) with tilt (-11.011739 0 0)
  triclinic box = (0 0 0) to (11.011739 12.120672 32.029568) with tilt (-11.011739 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.259 | 3.259 | 3.259 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-116.46537     -77.767153      0              0            
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
Nghost:            576 ave         576 max         576 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4176 ave        4176 max        4176 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4176
Ave neighs/atom = 52.2
Neighbor list builds = 0
Dangerous builds = 0
Total wall time: 0:00:06
