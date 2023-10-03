LAMMPS (23 Jun 2022 - Update 4)
units metal
atom_style atomic
atom_modify map array sort 0 0
boundary p p p
box tilt large
region cell prism    0 3.164 0 3.164 0 3.164     0.0 0.0 0.0     units box
create_box 1 cell
Created triclinic box = (0 0 0) to (3.164 3.164 3.164) with tilt (0 0 0)
  1 by 1 by 1 MPI processor grid
echo none
Created 2 atoms
  using lattice units in triclinic box = (0 0 0) to (3.164 3.164 3.164) with tilt (0 0 0)
  create_atoms CPU = 0.000 seconds
Setting atom values ...
  1 settings made for type
Setting atom values ...
  1 settings made for type
pair_style eam/alloy
pair_coeff * * potentials/Mo.eam.alloy Mo
Reading eam/alloy potential file potentials/Mo.eam.alloy with DATE: 2018-03-30
mass 1 95.949999975317410871866741217673
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
  master list distance cutoff = 8.100217
  ghost atom cutoff = 8.100217
  binsize = 4.0501085, bins = 1 1 1
  1 neighbor lists, perpetual/occasional/extra = 1 0 0
  (1) pair eam/alloy, perpetual
      attributes: half, newton on
      pair build: half/bin/newton/tri
      stencil: half/bin/3d/tri
      bin: standard
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.615608     -33565.043      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.96086106587977 y final 0 2.96086106587977 z final 0 2.96086106587977      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9608611 3.164 3.164) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9608611 2.9608611 3.164) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9608611 2.9608611 2.9608611) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9608611 2.9608611 2.9608611) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9608611 2.9608611 2.9608611) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9608611 2.9608611 2.9608611) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-12.617219      685444.34      0              0            
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
change_box all     x final 0 3.151328583480051 y final 0 3.151328583480051 z final 0 3.151328583480051      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1513286 2.9608611 2.9608611) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1513286 3.1513286 2.9608611) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1513286 3.1513286 3.1513286) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1513286 3.1513286 3.1513286) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1513286 3.1513286 3.1513286) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1513286 3.1513286 3.1513286) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.619974     -2975.1297      0              0            
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
change_box all     x final 0 3.1501150158299804 y final 0 3.1501150158299804 z final 0 3.1501150158299804      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.150115 3.1513286 3.1513286) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.150115 3.150115 3.1513286) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.150115 3.150115 3.150115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.150115 3.150115 3.150115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.150115 3.150115 3.150115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.150115 3.150115 3.150115) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.620007      58.337813      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.8351035142469825 y final 0 2.8351035142469825 z final 0 2.8351035142469825      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8351035 3.150115 3.150115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8351035 2.8351035 3.150115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8351035 2.8351035 2.8351035) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8351035 2.8351035 2.8351035) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8351035 2.8351035 2.8351035) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8351035 2.8351035 2.8351035) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-10.599956      1416214.7      0              0            
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
change_box all     x final 0 2.8666046644052825 y final 0 2.8666046644052825 z final 0 2.8666046644052825      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8666047 2.8351035 2.8351035) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8666047 2.8666047 2.8351035) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8666047 2.8666047 2.8666047) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8666047 2.8666047 2.8666047) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8666047 2.8666047 2.8666047) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8666047 2.8666047 2.8666047) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-11.22657       1202765.5      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 2.898105814563582 y final 0 2.898105814563582 z final 0 2.898105814563582      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.8981058 2.8666047 2.8666047) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8981058 2.8981058 2.8666047) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8981058 2.8981058 2.8981058) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8981058 2.8981058 2.8981058) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8981058 2.8981058 2.8981058) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.8981058 2.8981058 2.8981058) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-11.768138      1011659.7      0              0            
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
change_box all     x final 0 2.929606964721882 y final 0 2.929606964721882 z final 0 2.929606964721882      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.929607 2.8981058 2.8981058) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.929607 2.929607 2.8981058) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.929607 2.929607 2.929607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.929607 2.929607 2.929607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.929607 2.929607 2.929607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.929607 2.929607 2.929607) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-12.230854      839766.39      0              0            
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
change_box all     x final 0 2.9611081148801817 y final 0 2.9611081148801817 z final 0 2.9611081148801817      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9611081 2.929607 2.929607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9611081 2.9611081 2.929607) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9611081 2.9611081 2.9611081) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9611081 2.9611081 2.9611081) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9611081 2.9611081 2.9611081) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9611081 2.9611081 2.9611081) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-12.619997      684282.37      0              0            
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
change_box all     x final 0 2.9926092650384817 y final 0 2.9926092650384817 z final 0 2.9926092650384817      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (2.9926093 2.9611081 2.9611081) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9926093 2.9926093 2.9611081) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9926093 2.9926093 2.9926093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9926093 2.9926093 2.9926093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9926093 2.9926093 2.9926093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (2.9926093 2.9926093 2.9926093) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-12.940013      542850.77      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.0241104151967813 y final 0 3.0241104151967813 z final 0 3.0241104151967813      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0241104 2.9926093 2.9926093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0241104 3.0241104 2.9926093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0241104 3.0241104 3.0241104) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0241104 3.0241104 3.0241104) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0241104 3.0241104 3.0241104) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0241104 3.0241104 3.0241104) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.194684      413645.35      0              0            
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
change_box all     x final 0 3.0556115653550813 y final 0 3.0556115653550813 z final 0 3.0556115653550813      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0556116 3.0241104 3.0241104) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0556116 3.0556116 3.0241104) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0556116 3.0556116 3.0556116) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0556116 3.0556116 3.0556116) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0556116 3.0556116 3.0556116) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0556116 3.0556116 3.0556116) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.387332      295390.11      0              0            
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
change_box all     x final 0 3.0871127155133813 y final 0 3.0871127155133813 z final 0 3.0871127155133813      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.0871127 3.0556116 3.0556116) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0871127 3.0871127 3.0556116) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0871127 3.0871127 3.0871127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0871127 3.0871127 3.0871127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0871127 3.0871127 3.0871127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.0871127 3.0871127 3.0871127) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.520234      183958.48      0              0            
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
change_box all     x final 0 3.118613865671681 y final 0 3.118613865671681 z final 0 3.118613865671681      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1186139 3.0871127 3.0871127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1186139 3.1186139 3.0871127) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1186139 3.1186139 3.1186139) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1186139 3.1186139 3.1186139) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1186139 3.1186139 3.1186139) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1186139 3.1186139 3.1186139) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.595958      85362.607      0              0            
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
change_box all     x final 0 3.1501150158299804 y final 0 3.1501150158299804 z final 0 3.1501150158299804      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.150115 3.1186139 3.1186139) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.150115 3.150115 3.1186139) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.150115 3.150115 3.150115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.150115 3.150115 3.150115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.150115 3.150115 3.150115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.150115 3.150115 3.150115) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.620007      58.337813      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.181616165988281 y final 0 3.181616165988281 z final 0 3.181616165988281      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.1816162 3.150115 3.150115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1816162 3.1816162 3.150115) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1816162 3.1816162 3.1816162) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1816162 3.1816162 3.1816162) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1816162 3.1816162 3.1816162) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.1816162 3.1816162 3.1816162) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.597822     -72876.054      0              0            
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
change_box all     x final 0 3.21311731614658 y final 0 3.21311731614658 z final 0 3.21311731614658      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2131173 3.1816162 3.1816162) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2131173 3.2131173 3.1816162) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2131173 3.2131173 3.2131173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2131173 3.2131173 3.2131173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2131173 3.2131173 3.2131173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2131173 3.2131173 3.2131173) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.534752     -134239.17      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.2446184663048805 y final 0 3.2446184663048805 z final 0 3.2446184663048805      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2446185 3.2131173 3.2131173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2446185 3.2446185 3.2131173) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2446185 3.2446185 3.2446185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2446185 3.2446185 3.2446185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2446185 3.2446185 3.2446185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2446185 3.2446185 3.2446185) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.436075     -184747.9       0              0            
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
change_box all     x final 0 3.2761196164631796 y final 0 3.2761196164631796 z final 0 3.2761196164631796      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.2761196 3.2446185 3.2446185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2761196 3.2761196 3.2446185) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2761196 3.2761196 3.2761196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2761196 3.2761196 3.2761196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2761196 3.2761196 3.2761196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.2761196 3.2761196 3.2761196) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.305881     -230557         0              0            
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
change_box all     x final 0 3.30762076662148 y final 0 3.30762076662148 z final 0 3.30762076662148      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3076208 3.2761196 3.2761196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3076208 3.3076208 3.2761196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3076208 3.3076208 3.3076208) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3076208 3.3076208 3.3076208) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3076208 3.3076208 3.3076208) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3076208 3.3076208 3.3076208) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-13.143694     -276707         0              0            
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
change_box all     x final 0 3.339121916779779 y final 0 3.339121916779779 z final 0 3.339121916779779      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3391219 3.3076208 3.3076208) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3391219 3.3391219 3.3076208) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3391219 3.3391219 3.3391219) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3391219 3.3391219 3.3391219) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3391219 3.3391219 3.3391219) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3391219 3.3391219 3.3391219) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-12.94862      -321707.1       0              0            
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
change_box all     x final 0 3.37062306693808 y final 0 3.37062306693808 z final 0 3.37062306693808      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.3706231 3.3391219 3.3391219) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3706231 3.3706231 3.3391219) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3706231 3.3706231 3.3706231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3706231 3.3706231 3.3706231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3706231 3.3706231 3.3706231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.3706231 3.3706231 3.3706231) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-12.720724     -364188.9       0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 3.4021242170963792 y final 0 3.4021242170963792 z final 0 3.4021242170963792      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.4021242 3.3706231 3.3706231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4021242 3.4021242 3.3706231) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4021242 3.4021242 3.4021242) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4021242 3.4021242 3.4021242) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4021242 3.4021242 3.4021242) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4021242 3.4021242 3.4021242) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-12.460992     -402957.83      0              0            
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
change_box all     x final 0 3.4336253672546797 y final 0 3.4336253672546797 z final 0 3.4336253672546797      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.4336254 3.4021242 3.4021242) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4336254 3.4336254 3.4021242) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4336254 3.4336254 3.4336254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4336254 3.4336254 3.4336254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4336254 3.4336254 3.4336254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4336254 3.4336254 3.4336254) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-12.171248     -437099.73      0              0            
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
change_box all     x final 0 3.465126517412979 y final 0 3.465126517412979 z final 0 3.465126517412979      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (3.4651265 3.4336254 3.4336254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4651265 3.4651265 3.4336254) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4651265 3.4651265 3.4651265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4651265 3.4651265 3.4651265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4651265 3.4651265 3.4651265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (3.4651265 3.4651265 3.4651265) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.244 | 3.244 | 3.244 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-11.853966     -466122.96      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 2 atoms

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
change_box all     x final 0 9.450345047489941 y final 0 9.450345047489941 z final 0 9.450345047489941      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.450345 3.4651265 3.4651265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.450345 3.4651265) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
echo none
Created 52 atoms
  using lattice units in triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
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
-367.74019      58.337813      0              0            
Loop time of 1.584e-06 on 1 procs for 0 steps with 54 atoms

0.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.584e-06  |            |       |100.00

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
change_box all     x final 0 9.261338146540142 y final 0 9.450345047489941 z final 0 9.450345047489941      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.2613381 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.2613381 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.2613381 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.2613381 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.2613381 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.2613381 9.450345 9.450345) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.24812      93970.482      0              0            
Loop time of 9.58e-07 on 1 procs for 0 steps with 54 atoms

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
change_box all     x final 0 9.355841597015042 y final 0 9.450345047489941 z final 0 9.450345047489941      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.3558416 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3558416 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3558416 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3558416 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3558416 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.3558416 9.450345 9.450345) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.61807      46549.859      0              0            
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
change_box all     x final 0 9.450345047489941 y final 0 9.450345047489941 z final 0 9.450345047489941      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.74019      58.337813      0              0            
Loop time of 7.92e-07 on 1 procs for 0 steps with 54 atoms

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
change_box all     x final 0 9.544848497964841 y final 0 9.450345047489941 z final 0 9.450345047489941      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.5448485 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.5448485 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.5448485 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.5448485 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.5448485 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.5448485 9.450345 9.450345) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.62224     -44411.376      0              0            
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
change_box all     x final 0 9.63935194843974 y final 0 9.450345047489941 z final 0 9.450345047489941      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.6393519 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.6393519 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.6393519 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.6393519 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.6393519 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.6393519 9.450345 9.450345) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.27771     -85793.365      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 54 atoms

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
change_box all     x final 0 9.450345047489941 y final 0 9.450363948161135 z final 0 9.450326146856549      xy final 0.0 xz final 0.0 yz final 0.0189006522937131 units box
Changing box ...
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.4503639 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.4503639 9.4503261) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.4503639 9.4503261) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.4503639 9.4503261) with tilt (0 0 0)
  triclinic box = (0 0 0) to (9.450345 9.4503639 9.4503261) with tilt (0 0 0.018900652)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.739        67.757601      0              0            
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
change_box all     x final 0 9.450345047489941 y final 0 9.450544683920432 z final 0 9.450145415276637      xy final 0.0 xz final 0.0 yz final 0.06142594519929814 units box
Changing box ...
  triclinic box = (0 0 0) to (9.450345 9.4503639 9.4503261) with tilt (0 0 0.018900652)
  triclinic box = (0 0 0) to (9.450345 9.4505447 9.4503261) with tilt (0 0 0.018900652)
  triclinic box = (0 0 0) to (9.450345 9.4505447 9.4501454) with tilt (0 0 0.018900652)
  triclinic box = (0 0 0) to (9.450345 9.4505447 9.4501454) with tilt (0 0 0.018900652)
  triclinic box = (0 0 0) to (9.450345 9.4505447 9.4501454) with tilt (0 0 0.018900652)
  triclinic box = (0 0 0) to (9.450345 9.4505447 9.4501454) with tilt (0 0 0.061425945)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.7276       157.80901      0              0            
Loop time of 5.83e-07 on 1 procs for 0 steps with 54 atoms

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
change_box all     x final 0 9.450345047489941 y final 0 9.450916776071049 z final 0 9.449773353495276      xy final 0.0 xz final 0.0 yz final 0.10394750688844805 units box
Changing box ...
  triclinic box = (0 0 0) to (9.450345 9.4505447 9.4501454) with tilt (0 0 0.061425945)
  triclinic box = (0 0 0) to (9.450345 9.4509168 9.4501454) with tilt (0 0 0.061425945)
  triclinic box = (0 0 0) to (9.450345 9.4509168 9.4497734) with tilt (0 0 0.061425945)
  triclinic box = (0 0 0) to (9.450345 9.4509168 9.4497734) with tilt (0 0 0.061425945)
  triclinic box = (0 0 0) to (9.450345 9.4509168 9.4497734) with tilt (0 0 0.061425945)
  triclinic box = (0 0 0) to (9.450345 9.4509168 9.4497734) with tilt (0 0 0.10394751)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.70413      343.16413      0              0            
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
change_box all     x final 0 9.450345047489941 y final 0 9.451480202012647 z final 0 9.449210029303108      xy final 0.0 xz final 0.0 yz final 0.14646275545419815 units box
Changing box ...
  triclinic box = (0 0 0) to (9.450345 9.4509168 9.4497734) with tilt (0 0 0.10394751)
  triclinic box = (0 0 0) to (9.450345 9.4514802 9.4497734) with tilt (0 0 0.10394751)
  triclinic box = (0 0 0) to (9.450345 9.4514802 9.44921) with tilt (0 0 0.10394751)
  triclinic box = (0 0 0) to (9.450345 9.4514802 9.44921) with tilt (0 0 0.10394751)
  triclinic box = (0 0 0) to (9.450345 9.4514802 9.44921) with tilt (0 0 0.10394751)
  triclinic box = (0 0 0) to (9.450345 9.4514802 9.44921) with tilt (0 0 0.14646276)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.66859      623.71892      0              0            
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
change_box all     x final 0 9.450345047489941 y final 0 9.45223492753033 z final 0 9.448455545312207      xy final 0.0 xz final 0.0 yz final 0.18896911090624413 units box
Changing box ...
  triclinic box = (0 0 0) to (9.450345 9.4514802 9.44921) with tilt (0 0 0.14646276)
  triclinic box = (0 0 0) to (9.450345 9.4522349 9.44921) with tilt (0 0 0.14646276)
  triclinic box = (0 0 0) to (9.450345 9.4522349 9.4484555) with tilt (0 0 0.14646276)
  triclinic box = (0 0 0) to (9.450345 9.4522349 9.4484555) with tilt (0 0 0.14646276)
  triclinic box = (0 0 0) to (9.450345 9.4522349 9.4484555) with tilt (0 0 0.14646276)
  triclinic box = (0 0 0) to (9.450345 9.4522349 9.4484555) with tilt (0 0 0.18896911)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.62096      999.32458      0              0            
Loop time of 9.17e-07 on 1 procs for 0 steps with 54 atoms

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
change_box all     x final 0 9.450345047489941 y final 0 9.450345047489941 z final 0 9.450345047489941      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (9.450345 9.4522349 9.4484555) with tilt (0 0 0.18896911)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.4484555) with tilt (0 0 0.18896911)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0.18896911)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0.18896911)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0.18896911)
  triclinic box = (0 0 0) to (9.450345 9.450345 9.450345) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.251 | 3.251 | 3.251 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-367.73947      60.231711      0              0            
Loop time of 8.33e-07 on 1 procs for 0 steps with 54 atoms

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
change_box all     x final 0 12.600460063319922 y final 0 15.750575079149902 z final 0 18.900690094979883      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.60046 9.450345 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 9.450345) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
echo none
Created 185 atoms
  using lattice units in triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
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
-1624.4263     -1009.7978      0              0            
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
change_box all     x final 0 10.316384156770955 y final 0 12.895480195963692 z final 0 15.47457623515643      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.316384 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-161.17029      4380032.6      0              0            
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
change_box all     x final 0 12.59862998968875 y final 0 15.748287487110938 z final 0 18.897944984533126      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59863 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59863 15.748287 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59863 15.748287 18.897945) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59863 15.748287 18.897945) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59863 15.748287 18.897945) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59863 15.748287 18.897945) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1624.427       125.07353      0              0            
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
change_box all     x final 0 12.59862998968875 y final 0 15.748287487110938 z final 0 18.897944984533126      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59863 15.748287 18.897945) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59863 15.748287 18.897945) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59863 15.748287 18.897945) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59863 15.748287 18.897945) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59863 15.748287 18.897945) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59863 15.748287 18.897945) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1624.427       125.07353      0              0            
Loop time of 1.125e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.59874998241753 y final 0 15.748437478021913 z final 0 18.898124973627084      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59875 15.748287 18.897945) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59875 15.748437 18.897945) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59875 15.748437 18.898125) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59875 15.748437 18.898125) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59875 15.748437 18.898125) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59875 15.748437 18.898125) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1624.607       32.126497      0              0            
Loop time of 1.041e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.59874998241753 y final 0 15.748437478021913 z final 0 18.898124973627084      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59875 15.748437 18.898125) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59875 15.748437 18.898125) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59875 15.748437 18.898125) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59875 15.748437 18.898125) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59875 15.748437 18.898125) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59875 15.748437 18.898125) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1624.607       32.126497      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.598887742844205 y final 0 15.748609678555255 z final 0 18.898331614267036      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598888 15.748437 18.898125) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598888 15.74861 18.898125) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598888 15.74861 18.898332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598888 15.74861 18.898332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598888 15.74861 18.898332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598888 15.74861 18.898332) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1624.6271     -9.8569047      0              0            
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
change_box all     x final 0 12.598842789534322 y final 0 15.7485534869179 z final 0 18.898264184302228      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598843 15.74861 18.898332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598843 15.748553 18.898332) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598843 15.748553 18.898264) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598843 15.748553 18.898264) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598843 15.748553 18.898264) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598843 15.748553 18.898264) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1624.6332     -8.9836376      0              0            
Loop time of 1.042e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.598842789534322 y final 0 15.748553486917897 z final 0 18.898264184302228      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598843 15.748553 18.898264) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598843 15.748553 18.898264) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598843 15.748553 18.898264) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598843 15.748553 18.898264) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598843 15.748553 18.898264) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598843 15.748553 18.898264) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1624.6332     -8.9836376      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.59891929900363 y final 0 15.74864912375453 z final 0 18.898378948506117      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598919 15.748553 18.898264) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598919 15.748649 18.898264) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598919 15.748649 18.898379) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598919 15.748649 18.898379) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598919 15.748649 18.898379) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598919 15.748649 18.898379) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1624.6353     -15.160803      0              0            
Loop time of 9.59e-07 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.598892493796606 y final 0 15.74861561724575 z final 0 18.898338740695607      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598892 15.748649 18.898379) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598892 15.748616 18.898379) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598892 15.748616 18.898339) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598892 15.748616 18.898339) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598892 15.748616 18.898339) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598892 15.748616 18.898339) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1624.6362     -13.773667      0              0            
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
change_box all     x final 0 12.598892493796608 y final 0 15.74861561724575 z final 0 18.898338740695607      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598892 15.748616 18.898339) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598892 15.748616 18.898339) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598892 15.748616 18.898339) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598892 15.748616 18.898339) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598892 15.748616 18.898339) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598892 15.748616 18.898339) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1624.6362     -13.773667      0              0            
Loop time of 5.83e-07 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.598868970076781 y final 0 15.748586212595965 z final 0 18.898303455115855      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598869 15.748616 18.898339) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598869 15.748586 18.898339) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598869 15.748586 18.898303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598869 15.748586 18.898303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598869 15.748586 18.898303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598869 15.748586 18.898303) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1624.6364     -8.8126464      0              0            
Loop time of 1.208e-06 on 1 procs for 0 steps with 239 atoms

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
change_box all     x final 0 12.59887844432065 y final 0 15.748598055400803 z final 0 18.898317666481663      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598878 15.748586 18.898303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598878 15.748598 18.898303) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598878 15.748598 18.898318) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598878 15.748598 18.898318) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598878 15.748598 18.898318) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598878 15.748598 18.898318) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1624.6365     -10.970134      0              0            
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
change_box all     x final 0 12.600460063319922 y final 0 15.750575079149902 z final 0 18.900690094979883      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.60046 15.748598 18.898318) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.898318) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
echo none
1 atoms in group delatoms
Deleted 1 atoms, new total = 238
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8758     -1557.0392      0              0            
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
change_box all     x final 0 10.316384156770955 y final 0 12.895480195963692 z final 0 15.47457623515643      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.316384 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-164.26602      4344673.2      0              0            
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
change_box all     x final 0 12.5978350477295 y final 0 15.747293809661876 z final 0 18.896752571594252      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597835 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597835 15.747294 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597835 15.747294 18.896753) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597835 15.747294 18.896753) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597835 15.747294 18.896753) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597835 15.747294 18.896753) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8772      54.928941      0              0            
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
change_box all     x final 0 12.5978350477295 y final 0 15.747293809661876 z final 0 18.896752571594252      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597835 15.747294 18.896753) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597835 15.747294 18.896753) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597835 15.747294 18.896753) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597835 15.747294 18.896753) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597835 15.747294 18.896753) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597835 15.747294 18.896753) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8772      54.928941      0              0            
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
change_box all     x final 0 12.598024641003075 y final 0 15.747530801253845 z final 0 18.89703696150511      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598025 15.747294 18.896753) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598025 15.747531 18.896753) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598025 15.747531 18.897037) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598025 15.747531 18.897037) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598025 15.747531 18.897037) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598025 15.747531 18.897037) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.1287      74.275723      0              0            
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
Neighs:          16052 ave       16052 max       16052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16052
Ave neighs/atom = 67.445378
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.598024641003075 y final 0 15.747530801253845 z final 0 18.89703696150511      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598025 15.747531 18.897037) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598025 15.747531 18.897037) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598025 15.747531 18.897037) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598025 15.747531 18.897037) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598025 15.747531 18.897037) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598025 15.747531 18.897037) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.1287      74.275723      0              0            
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
Neighs:          16052 ave       16052 max       16052 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 16052
Ave neighs/atom = 67.445378
Neighbor list builds = 0
Dangerous builds = 0
set atom * x 0.0 y 0.0 z 0.0
Setting atom values ...
  238 settings made for x
  238 settings made for y
  238 settings made for z
change_box all     x final 0 12.598201519799447 y final 0 15.74775189974931 z final 0 18.897302279699673      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598202 15.747531 18.897037) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598202 15.747752 18.897037) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598202 15.747752 18.897302) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598202 15.747752 18.897302) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598202 15.747752 18.897302) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598202 15.747752 18.897302) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.1523     -4.2895039      0              0            
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
change_box all     x final 0 12.598136750107725 y final 0 15.747670937634657 z final 0 18.89720512516209      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598137 15.747752 18.897302) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598137 15.747671 18.897302) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598137 15.747671 18.897205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598137 15.747671 18.897205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598137 15.747671 18.897205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598137 15.747671 18.897205) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.1641      4.1584951      0              0            
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
change_box all     x final 0 12.598136750107725 y final 0 15.747670937634657 z final 0 18.89720512516209      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598137 15.747671 18.897205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598137 15.747671 18.897205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598137 15.747671 18.897205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598137 15.747671 18.897205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598137 15.747671 18.897205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598137 15.747671 18.897205) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.1641      4.1584951      0              0            
Loop time of 1.583e-06 on 1 procs for 0 steps with 238 atoms

315.9% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.598200100392916 y final 0 15.747750125491145 z final 0 18.897300150589842      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.5982 15.747671 18.897205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.5982 15.74775 18.897205) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.5982 15.74775 18.8973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.5982 15.74775 18.8973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.5982 15.74775 18.8973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.5982 15.74775 18.8973) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.168      -32.623929      0              0            
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
change_box all     x final 0 12.598200100392916 y final 0 15.747750125491143 z final 0 18.897300150589842      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.5982 15.74775 18.8973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.5982 15.74775 18.8973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.5982 15.74775 18.8973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.5982 15.74775 18.8973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.5982 15.74775 18.8973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.5982 15.74775 18.8973) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.168      -32.623929      0              0            
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
change_box all     x final 0 12.598139876245249 y final 0 15.747674845306559 z final 0 18.897209814368345      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59814 15.74775 18.8973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59814 15.747675 18.8973) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59814 15.747675 18.89721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59814 15.747675 18.89721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59814 15.747675 18.89721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59814 15.747675 18.89721) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.1692     -27.777682      0              0            
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
change_box all     x final 0 12.598139876245249 y final 0 15.747674845306557 z final 0 18.897209814368345      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59814 15.747675 18.89721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59814 15.747675 18.89721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59814 15.747675 18.89721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59814 15.747675 18.89721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59814 15.747675 18.89721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59814 15.747675 18.89721) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.1692     -27.777682      0              0            
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
change_box all     x final 0 12.598122070326612 y final 0 15.747652587908261 z final 0 18.8971831054904      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598122 15.747675 18.89721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598122 15.747653 18.89721) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598122 15.747653 18.897183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598122 15.747653 18.897183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598122 15.747653 18.897183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598122 15.747653 18.897183) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.1696     -29.664667      0              0            
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
change_box all     x final 0 12.598122070326612 y final 0 15.74765258790826 z final 0 18.8971831054904      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598122 15.747653 18.897183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598122 15.747653 18.897183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598122 15.747653 18.897183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598122 15.747653 18.897183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598122 15.747653 18.897183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598122 15.747653 18.897183) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.1696     -29.664667      0              0            
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
change_box all     x final 0 12.598122772426418 y final 0 15.747653465533016 z final 0 18.89718415864011      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.598123 15.747653 18.897183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598123 15.747653 18.897183) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598123 15.747653 18.897184) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598123 15.747653 18.897184) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598123 15.747653 18.897184) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.598123 15.747653 18.897184) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.1697     -35.290675      0              0            
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
change_box all     x final 0 12.600460063319922 y final 0 15.750575079149902 z final 0 18.900690094979883      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.60046 15.747653 18.897184) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.897184) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8487     -2654.3506      0              0            
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
change_box all     x final 0 10.316384156770955 y final 0 12.895480195963692 z final 0 15.47457623515643      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.316384 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-166.27024      4341317.3      0              0            
Loop time of 2e-06 on 1 procs for 0 steps with 238 atoms

350.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 2e-06      |            |       |100.00

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
change_box all     x final 0 12.596986035534758 y final 0 15.746232544418447 z final 0 18.895479053302136      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596986 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596986 15.746233 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596986 15.746233 18.895479) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596986 15.746233 18.895479) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596986 15.746233 18.895479) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596986 15.746233 18.895479) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8511     -514.97582      0              0            
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
change_box all     x final 0 12.59698603553476 y final 0 15.746232544418445 z final 0 18.895479053302136      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596986 15.746233 18.895479) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596986 15.746233 18.895479) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596986 15.746233 18.895479) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596986 15.746233 18.895479) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596986 15.746233 18.895479) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596986 15.746233 18.895479) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8511     -514.97582      0              0            
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
change_box all     x final 0 12.596067482347326 y final 0 15.745084352933521 z final 0 18.894101223520607      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596067 15.746233 18.895479) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596067 15.745084 18.895479) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596067 15.745084 18.894101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596067 15.745084 18.894101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596067 15.745084 18.894101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596067 15.745084 18.894101) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.1889     -727.1903       0              0            
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
change_box all     x final 0 12.596067482347326 y final 0 15.745084352933521 z final 0 18.894101223520607      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596067 15.745084 18.894101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596067 15.745084 18.894101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596067 15.745084 18.894101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596067 15.745084 18.894101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596067 15.745084 18.894101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596067 15.745084 18.894101) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.1889     -727.1903       0              0            
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
change_box all     x final 0 12.596162819572156 y final 0 15.745203524464566 z final 0 18.894244229357856      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596163 15.745084 18.894101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596163 15.745204 18.894101) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596163 15.745204 18.894244) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596163 15.745204 18.894244) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596163 15.745204 18.894244) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596163 15.745204 18.894244) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.2196     -884.22697      0              0            
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
change_box all     x final 0 12.596130509137494 y final 0 15.745163136421235 z final 0 18.894195763705863      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596131 15.745204 18.894244) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596131 15.745163 18.894244) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596131 15.745163 18.894196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596131 15.745163 18.894196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596131 15.745163 18.894196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596131 15.745163 18.894196) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.2306     -852.75371      0              0            
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
change_box all     x final 0 12.596130509137495 y final 0 15.745163136421233 z final 0 18.894195763705863      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596131 15.745163 18.894196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596131 15.745163 18.894196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596131 15.745163 18.894196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596131 15.745163 18.894196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596131 15.745163 18.894196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596131 15.745163 18.894196) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.2306     -852.75371      0              0            
Loop time of 1.667e-06 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.596331105860672 y final 0 15.745413882325245 z final 0 18.894496658790654      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596331 15.745163 18.894196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596331 15.745414 18.894196) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596331 15.745414 18.894497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596331 15.745414 18.894497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596331 15.745414 18.894497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596331 15.745414 18.894497) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.2361     -887.92285      0              0            
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
change_box all     x final 0 12.596331105860672 y final 0 15.745413882325243 z final 0 18.894496658790654      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596331 15.745414 18.894497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596331 15.745414 18.894497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596331 15.745414 18.894497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596331 15.745414 18.894497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596331 15.745414 18.894497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596331 15.745414 18.894497) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.2361     -887.92285      0              0            
Loop time of 1.542e-06 on 1 procs for 0 steps with 238 atoms

389.1% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.596262307366864 y final 0 15.745327884207981 z final 0 18.89439346104994      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596262 15.745414 18.894497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596262 15.745328 18.894497) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596262 15.745328 18.894393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596262 15.745328 18.894393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596262 15.745328 18.894393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596262 15.745328 18.894393) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.2377     -888.5961       0              0            
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
change_box all     x final 0 12.596262307366864 y final 0 15.745327884207981 z final 0 18.89439346104994      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596262 15.745328 18.894393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596262 15.745328 18.894393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596262 15.745328 18.894393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596262 15.745328 18.894393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596262 15.745328 18.894393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596262 15.745328 18.894393) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.2377     -888.5961       0              0            
Loop time of 1.875e-06 on 1 procs for 0 steps with 238 atoms

106.7% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 1.875e-06  |            |       |100.00

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
change_box all     x final 0 12.59622824658339 y final 0 15.74528530822863 z final 0 18.894342369874725      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596228 15.745328 18.894393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596228 15.745285 18.894393) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596228 15.745285 18.894342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596228 15.745285 18.894342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596228 15.745285 18.894342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596228 15.745285 18.894342) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.2381     -880.91707      0              0            
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
change_box all     x final 0 12.59622824658339 y final 0 15.74528530822863 z final 0 18.894342369874725      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596228 15.745285 18.894342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596228 15.745285 18.894342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596228 15.745285 18.894342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596228 15.745285 18.894342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596228 15.745285 18.894342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596228 15.745285 18.894342) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.2381     -880.91707      0              0            
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
change_box all     x final 0 12.596239146335172 y final 0 15.745298932918354 z final 0 18.894358719502396      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596239 15.745285 18.894342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596239 15.745299 18.894342) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596239 15.745299 18.894359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596239 15.745299 18.894359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596239 15.745299 18.894359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596239 15.745299 18.894359) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1615.2382     -886.64102      0              0            
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
change_box all     x final 0 12.600460063319922 y final 0 15.750575079149902 z final 0 18.900690094979883      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.60046 15.745299 18.894359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.894359) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.4229     -1993.6322      0              0            
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
change_box all     x final 0 10.316384156770955 y final 0 12.895480195963692 z final 0 15.47457623515643      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.316384 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-166.01339      4338662.3      0              0            
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
change_box all     x final 0 12.597153801753912 y final 0 15.74644225219239 z final 0 18.895730702630868      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597154 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597154 15.746442 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597154 15.746442 18.895731) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597154 15.746442 18.895731) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597154 15.746442 18.895731) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597154 15.746442 18.895731) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.4252      36.485207      0              0            
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
change_box all     x final 0 12.597153801753912 y final 0 15.746442252192388 z final 0 18.895730702630868      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597154 15.746442 18.895731) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597154 15.746442 18.895731) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597154 15.746442 18.895731) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597154 15.746442 18.895731) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597154 15.746442 18.895731) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597154 15.746442 18.895731) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.4252      36.485207      0              0            
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
change_box all     x final 0 12.597069854899944 y final 0 15.74633731862458 z final 0 18.895604782349917      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59707 15.746442 18.895731) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59707 15.746337 18.895731) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59707 15.746337 18.895605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59707 15.746337 18.895605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59707 15.746337 18.895605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59707 15.746337 18.895605) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.7792      156.31457      0              0            
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
change_box all     x final 0 12.597069854899944 y final 0 15.74633731862458 z final 0 18.895604782349917      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59707 15.746337 18.895605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59707 15.746337 18.895605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59707 15.746337 18.895605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59707 15.746337 18.895605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59707 15.746337 18.895605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59707 15.746337 18.895605) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.7792      156.31457      0              0            
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
change_box all     x final 0 12.597357161342465 y final 0 15.746696451677732 z final 0 18.8960357420137      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597357 15.746337 18.895605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597357 15.746696 18.895605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597357 15.746696 18.896036) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597357 15.746696 18.896036) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597357 15.746696 18.896036) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597357 15.746696 18.896036) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8018      42.121394      0              0            
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
change_box all     x final 0 12.597249075607941 y final 0 15.746561344509576 z final 0 18.895873613411915      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597249 15.746696 18.896036) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597249 15.746561 18.896036) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597249 15.746561 18.895874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597249 15.746561 18.895874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597249 15.746561 18.895874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597249 15.746561 18.895874) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8147      62.969508      0              0            
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
change_box all     x final 0 12.597249075607941 y final 0 15.746561344509576 z final 0 18.895873613411915      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597249 15.746561 18.895874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597249 15.746561 18.895874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597249 15.746561 18.895874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597249 15.746561 18.895874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597249 15.746561 18.895874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597249 15.746561 18.895874) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8147      62.969508      0              0            
Loop time of 3.125e-06 on 1 procs for 0 steps with 238 atoms

128.0% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 3.125e-06  |            |       |100.00

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
change_box all     x final 0 12.597422188747984 y final 0 15.746777735934646 z final 0 18.89613328312198      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597422 15.746561 18.895874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597422 15.746778 18.895874) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597422 15.746778 18.896133) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597422 15.746778 18.896133) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597422 15.746778 18.896133) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597422 15.746778 18.896133) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8178      9.6215473      0              0            
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
change_box all     x final 0 12.597366577476988 y final 0 15.746708221845896 z final 0 18.896049866215485      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597367 15.746778 18.896133) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597367 15.746708 18.896133) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597367 15.746708 18.89605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597367 15.746708 18.89605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597367 15.746708 18.89605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597367 15.746708 18.89605) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8186      25.279426      0              0            
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
change_box all     x final 0 12.597366577476988 y final 0 15.746708221845896 z final 0 18.896049866215485      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597367 15.746708 18.89605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597367 15.746708 18.89605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597367 15.746708 18.89605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597367 15.746708 18.89605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597367 15.746708 18.89605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597367 15.746708 18.89605) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8186      25.279426      0              0            
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
change_box all     x final 0 12.597340400641796 y final 0 15.746675500801912 z final 0 18.896010600962697      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59734 15.746708 18.89605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59734 15.746676 18.89605) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59734 15.746676 18.896011) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59734 15.746676 18.896011) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59734 15.746676 18.896011) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59734 15.746676 18.896011) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.819       26.488233      0              0            
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
change_box all     x final 0 12.597340400641796 y final 0 15.746675500801912 z final 0 18.896010600962697      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59734 15.746676 18.896011) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59734 15.746676 18.896011) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59734 15.746676 18.896011) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59734 15.746676 18.896011) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59734 15.746676 18.896011) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59734 15.746676 18.896011) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.819       26.488233      0              0            
Loop time of 1.875e-06 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.597322263850838 y final 0 15.746652829813211 z final 0 18.895983395776263      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597322 15.746676 18.896011) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597322 15.746653 18.896011) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597322 15.746653 18.895983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597322 15.746653 18.895983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597322 15.746653 18.895983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597322 15.746653 18.895983) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8192      30.577119      0              0            
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
change_box all     x final 0 12.600460063319922 y final 0 15.750575079149902 z final 0 18.900690094979883      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.60046 15.746653 18.895983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.895983) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.4403     -2012.1181      0              0            
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
change_box all     x final 0 10.316384156770955 y final 0 12.895480195963692 z final 0 15.47457623515643      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.316384 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
echo none
run 0
WARNING: No fixes with time integration, atoms won't move (src/src/verlet.cpp:60)
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-166.2931       4339025.2      0              0            
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
change_box all     x final 0 12.597062761868722 y final 0 15.746328452335902 z final 0 18.895594142803084      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597063 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597063 15.746328 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597063 15.746328 18.895594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597063 15.746328 18.895594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597063 15.746328 18.895594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597063 15.746328 18.895594) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
WARNING: Too many warnings: 101 vs 100. All future warnings will be suppressed (src/src/thermo.cpp:465)
-1614.4426      72.164094      0              0            
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
change_box all     x final 0 12.597062761868724 y final 0 15.7463284523359 z final 0 18.895594142803084      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597063 15.746328 18.895594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597063 15.746328 18.895594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597063 15.746328 18.895594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597063 15.746328 18.895594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597063 15.746328 18.895594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597063 15.746328 18.895594) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.4426      72.164094      0              0            
Loop time of 1.25e-06 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.596870584086359 y final 0 15.746088230107183 z final 0 18.895305876129537      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596871 15.746328 18.895594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596871 15.746088 18.895594) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596871 15.746088 18.895306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596871 15.746088 18.895306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596871 15.746088 18.895306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596871 15.746088 18.895306) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8856      267.94101      0              0            
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
change_box all     x final 0 12.596870584086359 y final 0 15.746088230107183 z final 0 18.895305876129537      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.596871 15.746088 18.895306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596871 15.746088 18.895306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596871 15.746088 18.895306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596871 15.746088 18.895306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596871 15.746088 18.895306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.596871 15.746088 18.895306) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8856      267.94101      0              0            
Loop time of 1.167e-06 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.597264760262934 y final 0 15.746580950327845 z final 0 18.8958971403944      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597265 15.746088 18.895306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597265 15.746581 18.895306) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597265 15.746581 18.895897) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597265 15.746581 18.895897) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597265 15.746581 18.895897) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597265 15.746581 18.895897) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.9079      229.60874      0              0            
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
change_box all     x final 0 12.597108497087047 y final 0 15.746385621358009 z final 0 18.89566274563057      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597108 15.746581 18.895897) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597108 15.746386 18.895897) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597108 15.746386 18.895663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597108 15.746386 18.895663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597108 15.746386 18.895663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597108 15.746386 18.895663) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.9247      222.55006      0              0            
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
change_box all     x final 0 12.597108497087047 y final 0 15.746385621358007 z final 0 18.89566274563057      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597108 15.746386 18.895663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597108 15.746386 18.895663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597108 15.746386 18.895663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597108 15.746386 18.895663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597108 15.746386 18.895663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597108 15.746386 18.895663) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.9247      222.55006      0              0            
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
change_box all     x final 0 12.597280111886054 y final 0 15.746600139856776 z final 0 18.89592016782908      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59728 15.746386 18.895663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59728 15.7466 18.895663) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59728 15.7466 18.89592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59728 15.7466 18.89592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59728 15.7466 18.89592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59728 15.7466 18.89592) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.9287      213.35239      0              0            
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
change_box all     x final 0 12.597280111886054 y final 0 15.746600139856774 z final 0 18.89592016782908      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59728 15.7466 18.89592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59728 15.7466 18.89592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59728 15.7466 18.89592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59728 15.7466 18.89592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59728 15.7466 18.89592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59728 15.7466 18.89592) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.9287      213.35239      0              0            
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
change_box all     x final 0 12.597181086565769 y final 0 15.746476358206419 z final 0 18.895771629848653      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597181 15.7466 18.89592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597181 15.746476 18.89592) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597181 15.746476 18.895772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597181 15.746476 18.895772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597181 15.746476 18.895772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597181 15.746476 18.895772) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.9303      229.10531      0              0            
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
change_box all     x final 0 12.597181086565769 y final 0 15.746476358206417 z final 0 18.895771629848653      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597181 15.746476 18.895772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597181 15.746476 18.895772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597181 15.746476 18.895772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597181 15.746476 18.895772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597181 15.746476 18.895772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597181 15.746476 18.895772) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.9303      229.10531      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 238 atoms

211.9% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.597142674392217 y final 0 15.746428342989473 z final 0 18.895714011588325      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597143 15.746476 18.895772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597143 15.746428 18.895772) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597143 15.746428 18.895714) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597143 15.746428 18.895714) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597143 15.746428 18.895714) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597143 15.746428 18.895714) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.9307      245.45362      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 238 atoms

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
change_box all     x final 0 12.600460063319922 y final 0 15.750575079149902 z final 0 18.900690094979883      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.60046 15.746428 18.895714) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.895714) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.60046 15.750575 18.90069) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.4401     -2049.7694      0              0            
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
change_box all     x final 0 10.316384156770955 y final 0 12.895480195963692 z final 0 15.47457623515643      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (10.316384 15.750575 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 18.90069) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (10.316384 12.89548 15.474576) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-165.93507      4339354.9      0              0            
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
change_box all     x final 0 12.59709376506038 y final 0 15.746367206325475 z final 0 18.89564064759057      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597094 12.89548 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597094 15.746367 15.474576) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597094 15.746367 18.895641) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597094 15.746367 18.895641) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597094 15.746367 18.895641) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597094 15.746367 18.895641) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.4425      16.106571      0              0            
Loop time of 1.416e-06 on 1 procs for 0 steps with 238 atoms

211.9% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.59709376506038 y final 0 15.746367206325473 z final 0 18.89564064759057      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597094 15.746367 18.895641) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597094 15.746367 18.895641) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597094 15.746367 18.895641) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597094 15.746367 18.895641) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597094 15.746367 18.895641) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597094 15.746367 18.895641) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.4425      16.106571      0              0            
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
change_box all     x final 0 12.597099031957534 y final 0 15.746373789946588 z final 0 18.8956485479367      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597099 15.746367 18.895641) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597099 15.746374 18.895641) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597099 15.746374 18.895649) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597099 15.746374 18.895649) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597099 15.746374 18.895649) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597099 15.746374 18.895649) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8173      105.34559      0              0            
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
change_box all     x final 0 12.597099031957534 y final 0 15.746373789946588 z final 0 18.8956485479367      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597099 15.746374 18.895649) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597099 15.746374 18.895649) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597099 15.746374 18.895649) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597099 15.746374 18.895649) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597099 15.746374 18.895649) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597099 15.746374 18.895649) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8173      105.34559      0              0            
Loop time of 1.5e-06 on 1 procs for 0 steps with 238 atoms

266.7% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 12.59735237726562 y final 0 15.746690471581681 z final 0 18.896028565898852      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597352 15.746374 18.895649) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597352 15.74669 18.895649) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597352 15.74669 18.896029) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597352 15.74669 18.896029) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597352 15.74669 18.896029) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597352 15.74669 18.896029) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8297     -3.408576       0              0            
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
change_box all     x final 0 12.597246027970971 y final 0 15.746557534963378 z final 0 18.89586904195687      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597246 15.74669 18.896029) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597246 15.746558 18.896029) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597246 15.746558 18.895869) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597246 15.746558 18.895869) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597246 15.746558 18.895869) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597246 15.746558 18.895869) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8432      23.714636      0              0            
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
change_box all     x final 0 12.597246027970971 y final 0 15.746557534963376 z final 0 18.89586904195687      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597246 15.746558 18.895869) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597246 15.746558 18.895869) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597246 15.746558 18.895869) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597246 15.746558 18.895869) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597246 15.746558 18.895869) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597246 15.746558 18.895869) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8432      23.714636      0              0            
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
change_box all     x final 0 12.597388667667976 y final 0 15.746735834584639 z final 0 18.896083001502372      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597389 15.746558 18.895869) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597389 15.746736 18.895869) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597389 15.746736 18.896083) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597389 15.746736 18.896083) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597389 15.746736 18.896083) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597389 15.746736 18.896083) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8454     -30.566695      0              0            
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
change_box all     x final 0 12.597343436447213 y final 0 15.746679295558682 z final 0 18.896015154671225      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597343 15.746736 18.896083) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597343 15.746679 18.896083) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597343 15.746679 18.896015) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597343 15.746679 18.896015) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597343 15.746679 18.896015) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597343 15.746679 18.896015) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8461     -14.479075      0              0            
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
change_box all     x final 0 12.597343436447213 y final 0 15.746679295558682 z final 0 18.896015154671225      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597343 15.746679 18.896015) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597343 15.746679 18.896015) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597343 15.746679 18.896015) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597343 15.746679 18.896015) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597343 15.746679 18.896015) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597343 15.746679 18.896015) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8461     -14.479075      0              0            
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
change_box all     x final 0 12.597326924237901 y final 0 15.746658655297043 z final 0 18.89599038635725      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597327 15.746679 18.896015) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597327 15.746659 18.896015) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597327 15.746659 18.89599) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597327 15.746659 18.89599) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597327 15.746659 18.89599) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597327 15.746659 18.89599) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8465     -14.953128      0              0            
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
change_box all     x final 0 12.597326924237901 y final 0 15.746658655297043 z final 0 18.89599038635725      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.597327 15.746659 18.89599) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597327 15.746659 18.89599) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597327 15.746659 18.89599) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597327 15.746659 18.89599) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597327 15.746659 18.89599) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.597327 15.746659 18.89599) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8465     -14.953128      0              0            
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
change_box all     x final 0 12.597309813093661 y final 0 15.746637266366742 z final 0 18.89596471964089      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (12.59731 15.746659 18.89599) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59731 15.746637 18.89599) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59731 15.746637 18.895965) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59731 15.746637 18.895965) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59731 15.746637 18.895965) with tilt (0 0 0)
  triclinic box = (0 0 0) to (12.59731 15.746637 18.895965) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-1614.8467     -11.553647      0              0            
Loop time of 7.291e-06 on 1 procs for 0 steps with 238 atoms

41.1% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 7.291e-06  |            |       |100.00

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
change_box all     x final 0 8.909870756843791 y final 0 6.300230031659961 z final 0 40.04720920289853      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9098708 15.746637 18.895965) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 6.30023 18.895965) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 6.30023 40.047209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 6.30023 40.047209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 6.30023 40.047209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 6.30023 40.047209) with tilt (0 0 0)
echo none
158 atoms in group delatoms
Deleted 158 atoms, new total = 80
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.68239      1258.5574      0              0            
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
change_box all     x final 0 8.909870756843791 y final 0 6.300230031659961 z final 0 40.04720920289852      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9098708 6.30023 40.047209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 6.30023 40.047209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 6.30023 40.047209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 6.30023 40.047209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 6.30023 40.047209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 6.30023 40.047209) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.68239      1258.5574      0              0            
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
change_box all     x final 0 7.294785194578205 y final 0 5.158192078385477 z final 0 32.787881749360594      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2947852 6.30023 40.047209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2947852 5.1581921 40.047209) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2947852 5.1581921 32.787882) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2947852 5.1581921 32.787882) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2947852 5.1581921 32.787882) with tilt (0 0 0)
  triclinic box = (0 0 0) to (7.2947852 5.1581921 32.787882) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.335 | 3.335 | 3.335 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-70.382873      2389633.4      0              0            
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
change_box all     x final 0 8.897484858607875 y final 0 6.291471879026258 z final 0 39.99153828786998      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8974849 5.1581921 32.787882) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8974849 6.2914719 32.787882) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8974849 6.2914719 39.991538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8974849 6.2914719 39.991538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8974849 6.2914719 39.991538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8974849 6.2914719 39.991538) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.70169      7300.3306      0              0            
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
change_box all     x final 0 8.902296627027965 y final 0 6.294874313105601 z final 0 40.013165750468914      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9022966 6.2914719 39.991538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9022966 6.2948743 39.991538) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9022966 6.2948743 40.013166) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9022966 6.2948743 40.013166) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9022966 6.2948743 40.013166) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9022966 6.2948743 40.013166) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.69991      7191.3378      0              0            
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
change_box all     x final 0 8.899857446627541 y final 0 6.293149552103925 z final 0 40.00220236273355      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8998574 6.2948743 40.013166) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8998574 6.2931496 40.013166) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8998574 6.2931496 40.002202) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8998574 6.2931496 40.002202) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8998574 6.2931496 40.002202) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8998574 6.2931496 40.002202) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73014      7230.4076      0              0            
Loop time of 1.292e-06 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.899857446627541 y final 0 6.293149552103925 z final 0 40.00220236273355      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8998574 6.2931496 40.002202) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8998574 6.2931496 40.002202) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8998574 6.2931496 40.002202) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8998574 6.2931496 40.002202) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8998574 6.2931496 40.002202) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8998574 6.2931496 40.002202) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73014      7230.4076      0              0            
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
change_box all     x final 0 8.90060180369021 y final 0 6.293675892030563 z final 0 40.00554802551855      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9006018 6.2931496 40.002202) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9006018 6.2936759 40.002202) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9006018 6.2936759 40.005548) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9006018 6.2936759 40.005548) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9006018 6.2936759 40.005548) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9006018 6.2936759 40.005548) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73299      7284.3867      0              0            
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
change_box all     x final 0 8.90060180369021 y final 0 6.293675892030563 z final 0 40.00554802551854      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9006018 6.2936759 40.005548) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9006018 6.2936759 40.005548) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9006018 6.2936759 40.005548) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9006018 6.2936759 40.005548) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9006018 6.2936759 40.005548) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9006018 6.2936759 40.005548) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73299      7284.3867      0              0            
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
change_box all     x final 0 8.900537396300106 y final 0 6.293630349128262 z final 0 40.00525853352746      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9005374 6.2936759 40.005548) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9005374 6.2936303 40.005548) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9005374 6.2936303 40.005259) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9005374 6.2936303 40.005259) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9005374 6.2936303 40.005259) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9005374 6.2936303 40.005259) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73466      7288.5496      0              0            
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
change_box all     x final 0 8.900537396300106 y final 0 6.293630349128262 z final 0 40.00525853352746      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9005374 6.2936303 40.005259) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9005374 6.2936303 40.005259) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9005374 6.2936303 40.005259) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9005374 6.2936303 40.005259) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9005374 6.2936303 40.005259) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9005374 6.2936303 40.005259) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73466      7288.5496      0              0            
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
change_box all     x final 0 8.900410133094281 y final 0 6.293540360452428 z final 0 40.00468652340896      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9004101 6.2936303 40.005259) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9004101 6.2935404 40.005259) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9004101 6.2935404 40.004687) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9004101 6.2935404 40.004687) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9004101 6.2935404 40.004687) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9004101 6.2935404 40.004687) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73469      7274.2526      0              0            
Loop time of 1.209e-06 on 1 procs for 0 steps with 80 atoms

248.1% CPU use with 1 MPI tasks x no OpenMP threads

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
change_box all     x final 0 8.900471691844892 y final 0 6.293583889062426 z final 0 40.004963211615916      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9004717 6.2935404 40.004687) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9004717 6.2935839 40.004687) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9004717 6.2935839 40.004963) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9004717 6.2935839 40.004963) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9004717 6.2935839 40.004963) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9004717 6.2935839 40.004963) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.7349       7280.9306      0              0            
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
change_box all     x final 0 8.900364988836412 y final 0 6.293508438641557 z final 0 40.004483613446666      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.900365 6.2935839 40.004963) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.900365 6.2935084 40.004963) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.900365 6.2935084 40.004484) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.900365 6.2935084 40.004484) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.900365 6.2935084 40.004484) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.900365 6.2935084 40.004484) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73502      7280.4014      0              0            
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
change_box all     x final 0 8.899938189594394 y final 0 6.293206646003321 z final 0 40.00256527826547      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8999382 6.2935084 40.004484) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8999382 6.2932066 40.004484) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8999382 6.2932066 40.002565) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8999382 6.2932066 40.002565) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8999382 6.2932066 40.002565) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8999382 6.2932066 40.002565) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73526      7278.5314      0              0            
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
change_box all     x final 0 8.899938189594394 y final 0 6.293206646003321 z final 0 40.00256527826547      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8999382 6.2932066 40.002565) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8999382 6.2932066 40.002565) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8999382 6.2932066 40.002565) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8999382 6.2932066 40.002565) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8999382 6.2932066 40.002565) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8999382 6.2932066 40.002565) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73526      7278.5314      0              0            
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
change_box all     x final 0 8.89903090451894 y final 0 6.292565098573997 z final 0 39.99848730270162      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8990309 6.2932066 40.002565) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8990309 6.2925651 40.002565) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8990309 6.2925651 39.998487) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8990309 6.2925651 39.998487) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8990309 6.2925651 39.998487) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8990309 6.2925651 39.998487) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73568      7393.4598      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.896563113040322 y final 0 6.290820106484911 z final 0 39.98739531671095      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8965631 6.2925651 39.998487) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8965631 6.2908201 39.998487) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8965631 6.2908201 39.987395) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8965631 6.2908201 39.987395) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8965631 6.2908201 39.987395) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8965631 6.2908201 39.987395) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73616      7707.8632      0              0            
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
change_box all     x final 0 8.896563113040322 y final 0 6.290820106484911 z final 0 39.987395316710945      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8965631 6.2908201 39.987395) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8965631 6.2908201 39.987395) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8965631 6.2908201 39.987395) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8965631 6.2908201 39.987395) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8965631 6.2908201 39.987395) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8965631 6.2908201 39.987395) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73616      7707.8632      0              0            
Loop time of 1.375e-06 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.89648926217701 y final 0 6.290767886038667 z final 0 39.98706337912694      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8964893 6.2908201 39.987395) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8964893 6.2907679 39.987395) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8964893 6.2907679 39.987063) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8964893 6.2907679 39.987063) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8964893 6.2907679 39.987063) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.8964893 6.2907679 39.987063) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.328 | 3.328 | 3.328 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-529.73622      7755.9836      0              0            
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
change_box all     x final 0 6.300230031659961 y final 0 6.300230031659961 z final 0 49.92609265038482      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.30023 6.2907679 39.987063) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.30023 6.30023 39.987063) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.30023 6.30023 49.926093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.30023 6.30023 49.926093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.30023 6.30023 49.926093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.30023 6.30023 49.926093) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.52517     -18345.254      0              0            
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
change_box all     x final 0 5.158192078385477 y final 0 5.158192078385477 z final 0 40.876027433890656      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (5.1581921 6.30023 49.926093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.1581921 5.1581921 49.926093) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.1581921 5.1581921 40.876027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.1581921 5.1581921 40.876027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.1581921 5.1581921 40.876027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (5.1581921 5.1581921 40.876027) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-70.666976      2651823.6      0              0            
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
change_box all     x final 0 6.284255925592295 y final 0 6.284255925592295 z final 0 49.7995060502871      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2842559 5.1581921 40.876027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2842559 6.2842559 40.876027) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2842559 6.2842559 49.799506) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2842559 6.2842559 49.799506) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2842559 6.2842559 49.799506) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2842559 6.2842559 49.799506) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.59233     -5646.0637      0              0            
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
change_box all     x final 0 6.284255925592295 y final 0 6.284255925592295 z final 0 49.7995060502871      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2842559 6.2842559 49.799506) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2842559 6.2842559 49.799506) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2842559 6.2842559 49.799506) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2842559 6.2842559 49.799506) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2842559 6.2842559 49.799506) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2842559 6.2842559 49.799506) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.59233     -5646.0637      0              0            
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
change_box all     x final 0 6.283585910867185 y final 0 6.283585910867185 z final 0 49.79419652713082      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2835859 6.2842559 49.799506) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2835859 6.2835859 49.799506) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2835859 6.2835859 49.794197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2835859 6.2835859 49.794197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2835859 6.2835859 49.794197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2835859 6.2835859 49.794197) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.59814     -4801.0632      0              0            
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
change_box all     x final 0 6.283819247908268 y final 0 6.283819247908268 z final 0 49.79604560354126      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2838192 6.2835859 49.794197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2838192 6.2838192 49.794197) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2838192 6.2838192 49.796046) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2838192 6.2838192 49.796046) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2838192 6.2838192 49.796046) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2838192 6.2838192 49.796046) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.60045     -5105.6975      0              0            
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
change_box all     x final 0 6.283819247908268 y final 0 6.283819247908268 z final 0 49.79604560354126      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2838192 6.2838192 49.796046) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2838192 6.2838192 49.796046) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2838192 6.2838192 49.796046) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2838192 6.2838192 49.796046) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2838192 6.2838192 49.796046) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2838192 6.2838192 49.796046) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.60045     -5105.6975      0              0            
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
change_box all     x final 0 6.283062634367089 y final 0 6.283062634367089 z final 0 49.790049829169895      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2830626 6.2838192 49.796046) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2830626 6.2830626 49.796046) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2830626 6.2830626 49.79005) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2830626 6.2830626 49.79005) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2830626 6.2830626 49.79005) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2830626 6.2830626 49.79005) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.60315     -4605.5996      0              0            
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
change_box all     x final 0 6.281966339230239 y final 0 6.281966339230239 z final 0 49.78136225231961      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2819663 6.2830626 49.79005) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2819663 6.2819663 49.79005) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2819663 6.2819663 49.781362) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2819663 6.2819663 49.781362) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2819663 6.2819663 49.781362) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2819663 6.2819663 49.781362) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.6046      -3875.8899      0              0            
Loop time of 2.125e-06 on 1 procs for 0 steps with 80 atoms

235.3% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0          | 0          | 0          |   0.0 |  0.00
Neigh   | 0          | 0          | 0          |   0.0 |  0.00
Comm    | 0          | 0          | 0          |   0.0 |  0.00
Output  | 0          | 0          | 0          |   0.0 |  0.00
Modify  | 0          | 0          | 0          |   0.0 |  0.00
Other   |            | 2.125e-06  |            |       |100.00

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
change_box all     x final 0 6.281966339230239 y final 0 6.281966339230239 z final 0 49.78136225231961      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2819663 6.2819663 49.781362) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2819663 6.2819663 49.781362) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2819663 6.2819663 49.781362) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2819663 6.2819663 49.781362) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2819663 6.2819663 49.781362) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2819663 6.2819663 49.781362) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.6046      -3875.8899      0              0            
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
change_box all     x final 0 6.280563415218728 y final 0 6.280563415218729 z final 0 49.770244798857085      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2805634 6.2819663 49.781362) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2805634 6.2805634 49.781362) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2805634 6.2805634 49.770245) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2805634 6.2805634 49.770245) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2805634 6.2805634 49.770245) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2805634 6.2805634 49.770245) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.61125     -3327.3304      0              0            
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
change_box all     x final 0 6.276582214487619 y final 0 6.27658221448762 z final 0 49.73869582436531      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2765822 6.2805634 49.770245) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2765822 6.2765822 49.770245) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2765822 6.2765822 49.738696) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2765822 6.2765822 49.738696) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2765822 6.2765822 49.738696) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2765822 6.2765822 49.738696) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.61927     -1749.3056      0              0            
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
change_box all     x final 0 6.272988989220159 y final 0 6.272988989220162 z final 0 49.71022135649423      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.272989 6.2765822 49.738696) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.272989 6.272989 49.738696) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.272989 6.272989 49.710221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.272989 6.272989 49.710221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.272989 6.272989 49.710221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.272989 6.272989 49.710221) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.63891     -1351.5494      0              0            
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
change_box all     x final 0 6.2685274228764065 y final 0 6.26852742287641 z final 0 49.67486573082267      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2685274 6.272989 49.710221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2685274 6.2685274 49.710221) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2685274 6.2685274 49.674866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2685274 6.2685274 49.674866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2685274 6.2685274 49.674866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2685274 6.2685274 49.674866) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.64758     -839.36355      0              0            
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
change_box all     x final 0 6.2685274228764065 y final 0 6.26852742287641 z final 0 49.67486573082267      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2685274 6.2685274 49.674866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2685274 6.2685274 49.674866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2685274 6.2685274 49.674866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2685274 6.2685274 49.674866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2685274 6.2685274 49.674866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2685274 6.2685274 49.674866) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.64758     -839.36355      0              0            
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
change_box all     x final 0 6.266255306471297 y final 0 6.266255306471301 z final 0 49.65686037330652      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2662553 6.2685274 49.674866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2662553 6.2662553 49.674866) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2662553 6.2662553 49.65686) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2662553 6.2662553 49.65686) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2662553 6.2662553 49.65686) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2662553 6.2662553 49.65686) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.6509      -219.03595      0              0            
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
change_box all     x final 0 6.267004438824124 y final 0 6.267004438824128 z final 0 49.662796863096055      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2670044 6.2662553 49.65686) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2670044 6.2670044 49.65686) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2670044 6.2670044 49.662797) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2670044 6.2670044 49.662797) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2670044 6.2670044 49.662797) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2670044 6.2670044 49.662797) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.65195     -427.71897      0              0            
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
change_box all     x final 0 6.2664818305562475 y final 0 6.266481830556251 z final 0 49.65865546053285      xy final 0.0 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (6.2664818 6.2670044 49.662797) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2664818 6.2664818 49.662797) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2664818 6.2664818 49.658655) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2664818 6.2664818 49.658655) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2664818 6.2664818 49.658655) with tilt (0 0 0)
  triclinic box = (0 0 0) to (6.2664818 6.2664818 49.658655) with tilt (0 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.336 | 3.336 | 3.336 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-532.65245      27.30979       0              0            
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
change_box all     x final 0 8.909870756843791 y final 0 7.716174419862806 z final 0 37.27783764749335      xy final 4.454935378421895 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9098708 6.2664818 49.658655) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 7.7161744 49.658655) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 7.7161744 37.277838) with tilt (0 0 0)
  triclinic box = (0 0 0) to (8.9098708 7.7161744 37.277838) with tilt (4.4549354 0 0)
  triclinic box = (0 0 0) to (8.9098708 7.7161744 37.277838) with tilt (4.4549354 0 0)
  triclinic box = (0 0 0) to (8.9098708 7.7161744 37.277838) with tilt (4.4549354 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.21637     -9852.4182      0              0            
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
change_box all     x final 0 7.294785194578205 y final 0 6.317469293655334 z final 0 30.52051209025097      xy final 3.6473925972891017 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2947852 7.7161744 37.277838) with tilt (4.4549354 0 0)
  triclinic box = (0 0 0) to (7.2947852 6.3174693 37.277838) with tilt (4.4549354 0 0)
  triclinic box = (0 0 0) to (7.2947852 6.3174693 30.520512) with tilt (4.4549354 0 0)
  triclinic box = (0 0 0) to (7.2947852 6.3174693 30.520512) with tilt (3.6473926 0 0)
  triclinic box = (0 0 0) to (7.2947852 6.3174693 30.520512) with tilt (3.6473926 0 0)
  triclinic box = (0 0 0) to (7.2947852 6.3174693 30.520512) with tilt (3.6473926 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-78.924026      1993860.8      0              0            
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
change_box all     x final 0 8.882490130649181 y final 0 7.692462102006749 z final 0 37.16328036985996      xy final 4.44124506532459 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8824901 6.3174693 30.520512) with tilt (3.6473926 0 0)
  triclinic box = (0 0 0) to (8.8824901 7.6924621 30.520512) with tilt (3.6473926 0 0)
  triclinic box = (0 0 0) to (8.8824901 7.6924621 37.16328) with tilt (3.6473926 0 0)
  triclinic box = (0 0 0) to (8.8824901 7.6924621 37.16328) with tilt (4.4412451 0 0)
  triclinic box = (0 0 0) to (8.8824901 7.6924621 37.16328) with tilt (4.4412451 0 0)
  triclinic box = (0 0 0) to (8.8824901 7.6924621 37.16328) with tilt (4.4412451 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.31131      1465.2723      0              0            
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
change_box all     x final 0 8.882490130649181 y final 0 7.692462102006749 z final 0 37.16328036985996      xy final 4.44124506532459 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8824901 7.6924621 37.16328) with tilt (4.4412451 0 0)
  triclinic box = (0 0 0) to (8.8824901 7.6924621 37.16328) with tilt (4.4412451 0 0)
  triclinic box = (0 0 0) to (8.8824901 7.6924621 37.16328) with tilt (4.4412451 0 0)
  triclinic box = (0 0 0) to (8.8824901 7.6924621 37.16328) with tilt (4.4412451 0 0)
  triclinic box = (0 0 0) to (8.8824901 7.6924621 37.16328) with tilt (4.4412451 0 0)
  triclinic box = (0 0 0) to (8.8824901 7.6924621 37.16328) with tilt (4.4412451 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.31131      1465.2723      0              0            
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
change_box all     x final 0 8.903483942767588 y final 0 7.710643276623577 z final 0 37.25111597837889      xy final 4.451741971383793 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9034839 7.6924621 37.16328) with tilt (4.4412451 0 0)
  triclinic box = (0 0 0) to (8.9034839 7.7106433 37.16328) with tilt (4.4412451 0 0)
  triclinic box = (0 0 0) to (8.9034839 7.7106433 37.251116) with tilt (4.4412451 0 0)
  triclinic box = (0 0 0) to (8.9034839 7.7106433 37.251116) with tilt (4.451742 0 0)
  triclinic box = (0 0 0) to (8.9034839 7.7106433 37.251116) with tilt (4.451742 0 0)
  triclinic box = (0 0 0) to (8.9034839 7.7106433 37.251116) with tilt (4.451742 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.68818      1727.5713      0              0            
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
change_box all     x final 0 8.895538791969223 y final 0 7.70376257419529 z final 0 37.21787441409256      xy final 4.4477693959846105 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8955388 7.7106433 37.251116) with tilt (4.451742 0 0)
  triclinic box = (0 0 0) to (8.8955388 7.7037626 37.251116) with tilt (4.451742 0 0)
  triclinic box = (0 0 0) to (8.8955388 7.7037626 37.217874) with tilt (4.451742 0 0)
  triclinic box = (0 0 0) to (8.8955388 7.7037626 37.217874) with tilt (4.4477694 0 0)
  triclinic box = (0 0 0) to (8.8955388 7.7037626 37.217874) with tilt (4.4477694 0 0)
  triclinic box = (0 0 0) to (8.8955388 7.7037626 37.217874) with tilt (4.4477694 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.90221      1616.553       0              0            
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
change_box all     x final 0 8.895538791969223 y final 0 7.703762574195289 z final 0 37.21787441409256      xy final 4.4477693959846105 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8955388 7.7037626 37.217874) with tilt (4.4477694 0 0)
  triclinic box = (0 0 0) to (8.8955388 7.7037626 37.217874) with tilt (4.4477694 0 0)
  triclinic box = (0 0 0) to (8.8955388 7.7037626 37.217874) with tilt (4.4477694 0 0)
  triclinic box = (0 0 0) to (8.8955388 7.7037626 37.217874) with tilt (4.4477694 0 0)
  triclinic box = (0 0 0) to (8.8955388 7.7037626 37.217874) with tilt (4.4477694 0 0)
  triclinic box = (0 0 0) to (8.8955388 7.7037626 37.217874) with tilt (4.4477694 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.90221      1616.553       0              0            
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
change_box all     x final 0 8.899752646574296 y final 0 7.707411879331138 z final 0 37.23550467968507      xy final 4.449876323287147 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8997526 7.7037626 37.217874) with tilt (4.4477694 0 0)
  triclinic box = (0 0 0) to (8.8997526 7.7074119 37.217874) with tilt (4.4477694 0 0)
  triclinic box = (0 0 0) to (8.8997526 7.7074119 37.235505) with tilt (4.4477694 0 0)
  triclinic box = (0 0 0) to (8.8997526 7.7074119 37.235505) with tilt (4.4498763 0 0)
  triclinic box = (0 0 0) to (8.8997526 7.7074119 37.235505) with tilt (4.4498763 0 0)
  triclinic box = (0 0 0) to (8.8997526 7.7074119 37.235505) with tilt (4.4498763 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.96191      1479.5093      0              0            
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
change_box all     x final 0 8.900374190813313 y final 0 7.707950152431701 z final 0 37.238105146724955      xy final 4.450187095406656 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9003742 7.7074119 37.235505) with tilt (4.4498763 0 0)
  triclinic box = (0 0 0) to (8.9003742 7.7079502 37.235505) with tilt (4.4498763 0 0)
  triclinic box = (0 0 0) to (8.9003742 7.7079502 37.238105) with tilt (4.4498763 0 0)
  triclinic box = (0 0 0) to (8.9003742 7.7079502 37.238105) with tilt (4.4501871 0 0)
  triclinic box = (0 0 0) to (8.9003742 7.7079502 37.238105) with tilt (4.4501871 0 0)
  triclinic box = (0 0 0) to (8.9003742 7.7079502 37.238105) with tilt (4.4501871 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.97612      1412.272       0              0            
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
change_box all     x final 0 8.900374190813313 y final 0 7.7079501524317005 z final 0 37.238105146724955      xy final 4.450187095406656 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9003742 7.7079502 37.238105) with tilt (4.4501871 0 0)
  triclinic box = (0 0 0) to (8.9003742 7.7079502 37.238105) with tilt (4.4501871 0 0)
  triclinic box = (0 0 0) to (8.9003742 7.7079502 37.238105) with tilt (4.4501871 0 0)
  triclinic box = (0 0 0) to (8.9003742 7.7079502 37.238105) with tilt (4.4501871 0 0)
  triclinic box = (0 0 0) to (8.9003742 7.7079502 37.238105) with tilt (4.4501871 0 0)
  triclinic box = (0 0 0) to (8.9003742 7.7079502 37.238105) with tilt (4.4501871 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.97612      1412.272       0              0            
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
change_box all     x final 0 8.90069812238997 y final 0 7.708230685406174 z final 0 37.23946043784568      xy final 4.450349061194984 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9006981 7.7079502 37.238105) with tilt (4.4501871 0 0)
  triclinic box = (0 0 0) to (8.9006981 7.7082307 37.238105) with tilt (4.4501871 0 0)
  triclinic box = (0 0 0) to (8.9006981 7.7082307 37.23946) with tilt (4.4501871 0 0)
  triclinic box = (0 0 0) to (8.9006981 7.7082307 37.23946) with tilt (4.4503491 0 0)
  triclinic box = (0 0 0) to (8.9006981 7.7082307 37.23946) with tilt (4.4503491 0 0)
  triclinic box = (0 0 0) to (8.9006981 7.7082307 37.23946) with tilt (4.4503491 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.97846      1351.0888      0              0            
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
change_box all     x final 0 8.90069812238997 y final 0 7.708230685406173 z final 0 37.23946043784568      xy final 4.450349061194984 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9006981 7.7082307 37.23946) with tilt (4.4503491 0 0)
  triclinic box = (0 0 0) to (8.9006981 7.7082307 37.23946) with tilt (4.4503491 0 0)
  triclinic box = (0 0 0) to (8.9006981 7.7082307 37.23946) with tilt (4.4503491 0 0)
  triclinic box = (0 0 0) to (8.9006981 7.7082307 37.23946) with tilt (4.4503491 0 0)
  triclinic box = (0 0 0) to (8.9006981 7.7082307 37.23946) with tilt (4.4503491 0 0)
  triclinic box = (0 0 0) to (8.9006981 7.7082307 37.23946) with tilt (4.4503491 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.97846      1351.0888      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.90127107842528 y final 0 7.708726879888002 z final 0 37.241857617630465      xy final 4.450635539212639 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9012711 7.7082307 37.23946) with tilt (4.4503491 0 0)
  triclinic box = (0 0 0) to (8.9012711 7.7087269 37.23946) with tilt (4.4503491 0 0)
  triclinic box = (0 0 0) to (8.9012711 7.7087269 37.241858) with tilt (4.4503491 0 0)
  triclinic box = (0 0 0) to (8.9012711 7.7087269 37.241858) with tilt (4.4506355 0 0)
  triclinic box = (0 0 0) to (8.9012711 7.7087269 37.241858) with tilt (4.4506355 0 0)
  triclinic box = (0 0 0) to (8.9012711 7.7087269 37.241858) with tilt (4.4506355 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.97975      1276.4861      0              0            
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
change_box all     x final 0 8.90127107842528 y final 0 7.708726879888001 z final 0 37.241857617630465      xy final 4.450635539212639 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9012711 7.7087269 37.241858) with tilt (4.4506355 0 0)
  triclinic box = (0 0 0) to (8.9012711 7.7087269 37.241858) with tilt (4.4506355 0 0)
  triclinic box = (0 0 0) to (8.9012711 7.7087269 37.241858) with tilt (4.4506355 0 0)
  triclinic box = (0 0 0) to (8.9012711 7.7087269 37.241858) with tilt (4.4506355 0 0)
  triclinic box = (0 0 0) to (8.9012711 7.7087269 37.241858) with tilt (4.4506355 0 0)
  triclinic box = (0 0 0) to (8.9012711 7.7087269 37.241858) with tilt (4.4506355 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.97975      1276.4861      0              0            
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
change_box all     x final 0 8.902837982766808 y final 0 7.710083858853062 z final 0 37.248413358700866      xy final 4.451418991383403 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.902838 7.7087269 37.241858) with tilt (4.4506355 0 0)
  triclinic box = (0 0 0) to (8.902838 7.7100839 37.241858) with tilt (4.4506355 0 0)
  triclinic box = (0 0 0) to (8.902838 7.7100839 37.248413) with tilt (4.4506355 0 0)
  triclinic box = (0 0 0) to (8.902838 7.7100839 37.248413) with tilt (4.451419 0 0)
  triclinic box = (0 0 0) to (8.902838 7.7100839 37.248413) with tilt (4.451419 0 0)
  triclinic box = (0 0 0) to (8.902838 7.7100839 37.248413) with tilt (4.451419 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.98178      1095.555       0              0            
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
change_box all     x final 0 8.905217502735477 y final 0 7.7121445835947355 z final 0 37.25836898673315      xy final 4.4526087513677375 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9052175 7.7100839 37.248413) with tilt (4.451419 0 0)
  triclinic box = (0 0 0) to (8.9052175 7.7121446 37.248413) with tilt (4.451419 0 0)
  triclinic box = (0 0 0) to (8.9052175 7.7121446 37.258369) with tilt (4.451419 0 0)
  triclinic box = (0 0 0) to (8.9052175 7.7121446 37.258369) with tilt (4.4526088 0 0)
  triclinic box = (0 0 0) to (8.9052175 7.7121446 37.258369) with tilt (4.4526088 0 0)
  triclinic box = (0 0 0) to (8.9052175 7.7121446 37.258369) with tilt (4.4526088 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.98294      822.30648      0              0            
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
change_box all     x final 0 8.905217502735477 y final 0 7.712144583594735 z final 0 37.25836898673315      xy final 4.4526087513677375 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9052175 7.7121446 37.258369) with tilt (4.4526088 0 0)
  triclinic box = (0 0 0) to (8.9052175 7.7121446 37.258369) with tilt (4.4526088 0 0)
  triclinic box = (0 0 0) to (8.9052175 7.7121446 37.258369) with tilt (4.4526088 0 0)
  triclinic box = (0 0 0) to (8.9052175 7.7121446 37.258369) with tilt (4.4526088 0 0)
  triclinic box = (0 0 0) to (8.9052175 7.7121446 37.258369) with tilt (4.4526088 0 0)
  triclinic box = (0 0 0) to (8.9052175 7.7121446 37.258369) with tilt (4.4526088 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.98294      822.30648      0              0            
Loop time of 1.166e-06 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.908506117593479 y final 0 7.714992607605021 z final 0 37.27212815946497      xy final 4.454253058796739 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9085061 7.7121446 37.258369) with tilt (4.4526088 0 0)
  triclinic box = (0 0 0) to (8.9085061 7.7149926 37.258369) with tilt (4.4526088 0 0)
  triclinic box = (0 0 0) to (8.9085061 7.7149926 37.272128) with tilt (4.4526088 0 0)
  triclinic box = (0 0 0) to (8.9085061 7.7149926 37.272128) with tilt (4.4542531 0 0)
  triclinic box = (0 0 0) to (8.9085061 7.7149926 37.272128) with tilt (4.4542531 0 0)
  triclinic box = (0 0 0) to (8.9085061 7.7149926 37.272128) with tilt (4.4542531 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.98605      456.87589      0              0            
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
change_box all     x final 0 8.908506117593479 y final 0 7.71499260760502 z final 0 37.27212815946497      xy final 4.454253058796739 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9085061 7.7149926 37.272128) with tilt (4.4542531 0 0)
  triclinic box = (0 0 0) to (8.9085061 7.7149926 37.272128) with tilt (4.4542531 0 0)
  triclinic box = (0 0 0) to (8.9085061 7.7149926 37.272128) with tilt (4.4542531 0 0)
  triclinic box = (0 0 0) to (8.9085061 7.7149926 37.272128) with tilt (4.4542531 0 0)
  triclinic box = (0 0 0) to (8.9085061 7.7149926 37.272128) with tilt (4.4542531 0 0)
  triclinic box = (0 0 0) to (8.9085061 7.7149926 37.272128) with tilt (4.4542531 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.98605      456.87589      0              0            
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
change_box all     x final 0 8.912547515652934 y final 0 7.718492560991304 z final 0 37.28903688741856      xy final 4.456273757826466 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9125475 7.7149926 37.272128) with tilt (4.4542531 0 0)
  triclinic box = (0 0 0) to (8.9125475 7.7184926 37.272128) with tilt (4.4542531 0 0)
  triclinic box = (0 0 0) to (8.9125475 7.7184926 37.289037) with tilt (4.4542531 0 0)
  triclinic box = (0 0 0) to (8.9125475 7.7184926 37.289037) with tilt (4.4562738 0 0)
  triclinic box = (0 0 0) to (8.9125475 7.7184926 37.289037) with tilt (4.4562738 0 0)
  triclinic box = (0 0 0) to (8.9125475 7.7184926 37.289037) with tilt (4.4562738 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.337 | 3.337 | 3.337 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-520.98801     -12.028476      0              0            
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
change_box all     x final 0 8.909870756843791 y final 0 10.912318514206328 z final 0 31.5742616297942      xy final -8.90987075684379 xz final 0.0 yz final -0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9098708 7.7184926 37.289037) with tilt (4.4562738 0 0)
  triclinic box = (0 0 0) to (8.9098708 10.912319 37.289037) with tilt (4.4562738 0 0)
  triclinic box = (0 0 0) to (8.9098708 10.912319 31.574262) with tilt (4.4562738 0 0)
  triclinic box = (0 0 0) to (8.9098708 10.912319 31.574262) with tilt (-8.9098708 0 0)
  triclinic box = (0 0 0) to (8.9098708 10.912319 31.574262) with tilt (-8.9098708 0 0)
  triclinic box = (0 0 0) to (8.9098708 10.912319 31.574262) with tilt (-8.9098708 0 -0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-513.15844     -13560.982      0              0            
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
change_box all     x final 0 8.909870756843791 y final 0 10.912318514206328 z final 0 31.574261629794197      xy final -8.90987075684379 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.9098708 10.912319 31.574262) with tilt (-8.9098708 0 -0)
  triclinic box = (0 0 0) to (8.9098708 10.912319 31.574262) with tilt (-8.9098708 0 -0)
  triclinic box = (0 0 0) to (8.9098708 10.912319 31.574262) with tilt (-8.9098708 0 -0)
  triclinic box = (0 0 0) to (8.9098708 10.912319 31.574262) with tilt (-8.9098708 0 -0)
  triclinic box = (0 0 0) to (8.9098708 10.912319 31.574262) with tilt (-8.9098708 0 -0)
  triclinic box = (0 0 0) to (8.9098708 10.912319 31.574262) with tilt (-8.9098708 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-513.15844     -13560.982      0              0            
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
change_box all     x final 0 7.294785194578205 y final 0 8.934250754962951 z final 0 25.850819002042627      xy final -7.294785194578203 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (7.2947852 10.912319 31.574262) with tilt (-8.9098708 0 0)
  triclinic box = (0 0 0) to (7.2947852 8.9342508 31.574262) with tilt (-8.9098708 0 0)
  triclinic box = (0 0 0) to (7.2947852 8.9342508 25.850819) with tilt (-8.9098708 0 0)
  triclinic box = (0 0 0) to (7.2947852 8.9342508 25.850819) with tilt (-7.2947852 0 0)
  triclinic box = (0 0 0) to (7.2947852 8.9342508 25.850819) with tilt (-7.2947852 0 0)
  triclinic box = (0 0 0) to (7.2947852 8.9342508 25.850819) with tilt (-7.2947852 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-88.988578      1584393.8      0              0            
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
change_box all     x final 0 8.87451020955439 y final 0 10.869010865264036 z final 0 31.44895305888921      xy final -8.874510209554389 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8745102 8.9342508 25.850819) with tilt (-7.2947852 0 0)
  triclinic box = (0 0 0) to (8.8745102 10.869011 25.850819) with tilt (-7.2947852 0 0)
  triclinic box = (0 0 0) to (8.8745102 10.869011 31.448953) with tilt (-7.2947852 0 0)
  triclinic box = (0 0 0) to (8.8745102 10.869011 31.448953) with tilt (-8.8745102 0 0)
  triclinic box = (0 0 0) to (8.8745102 10.869011 31.448953) with tilt (-8.8745102 0 0)
  triclinic box = (0 0 0) to (8.8745102 10.869011 31.448953) with tilt (-8.8745102 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-513.31043     -1721.9983      0              0            
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
change_box all     x final 0 8.87451020955439 y final 0 10.869010865264036 z final 0 31.448953058889206      xy final -8.874510209554389 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8745102 10.869011 31.448953) with tilt (-8.8745102 0 0)
  triclinic box = (0 0 0) to (8.8745102 10.869011 31.448953) with tilt (-8.8745102 0 0)
  triclinic box = (0 0 0) to (8.8745102 10.869011 31.448953) with tilt (-8.8745102 0 0)
  triclinic box = (0 0 0) to (8.8745102 10.869011 31.448953) with tilt (-8.8745102 0 0)
  triclinic box = (0 0 0) to (8.8745102 10.869011 31.448953) with tilt (-8.8745102 0 0)
  triclinic box = (0 0 0) to (8.8745102 10.869011 31.448953) with tilt (-8.8745102 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-513.31043     -1721.9983      0              0            
Loop time of 7.5e-07 on 1 procs for 0 steps with 80 atoms

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
change_box all     x final 0 8.891647753527575 y final 0 10.889999984353448 z final 0 31.509684051724943      xy final -8.891647753527574 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8916478 10.869011 31.448953) with tilt (-8.8745102 0 0)
  triclinic box = (0 0 0) to (8.8916478 10.89 31.448953) with tilt (-8.8745102 0 0)
  triclinic box = (0 0 0) to (8.8916478 10.89 31.509684) with tilt (-8.8745102 0 0)
  triclinic box = (0 0 0) to (8.8916478 10.89 31.509684) with tilt (-8.8916478 0 0)
  triclinic box = (0 0 0) to (8.8916478 10.89 31.509684) with tilt (-8.8916478 0 0)
  triclinic box = (0 0 0) to (8.8916478 10.89 31.509684) with tilt (-8.8916478 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-513.94076     -1202.8917      0              0            
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
change_box all     x final 0 8.891647753527575 y final 0 10.889999984353448 z final 0 31.50968405172494      xy final -8.891647753527574 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8916478 10.89 31.509684) with tilt (-8.8916478 0 0)
  triclinic box = (0 0 0) to (8.8916478 10.89 31.509684) with tilt (-8.8916478 0 0)
  triclinic box = (0 0 0) to (8.8916478 10.89 31.509684) with tilt (-8.8916478 0 0)
  triclinic box = (0 0 0) to (8.8916478 10.89 31.509684) with tilt (-8.8916478 0 0)
  triclinic box = (0 0 0) to (8.8916478 10.89 31.509684) with tilt (-8.8916478 0 0)
  triclinic box = (0 0 0) to (8.8916478 10.89 31.509684) with tilt (-8.8916478 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-513.94076     -1202.8917      0              0            
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
change_box all     x final 0 8.888300095835707 y final 0 10.885899957764158 z final 0 31.49782082467096      xy final -8.888300095835705 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8883001 10.89 31.509684) with tilt (-8.8916478 0 0)
  triclinic box = (0 0 0) to (8.8883001 10.8859 31.509684) with tilt (-8.8916478 0 0)
  triclinic box = (0 0 0) to (8.8883001 10.8859 31.497821) with tilt (-8.8916478 0 0)
  triclinic box = (0 0 0) to (8.8883001 10.8859 31.497821) with tilt (-8.8883001 0 0)
  triclinic box = (0 0 0) to (8.8883001 10.8859 31.497821) with tilt (-8.8883001 0 0)
  triclinic box = (0 0 0) to (8.8883001 10.8859 31.497821) with tilt (-8.8883001 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.05107     -1283.6874      0              0            
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
change_box all     x final 0 8.888300095835707 y final 0 10.885899957764158 z final 0 31.49782082467096      xy final -8.888300095835705 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8883001 10.8859 31.497821) with tilt (-8.8883001 0 0)
  triclinic box = (0 0 0) to (8.8883001 10.8859 31.497821) with tilt (-8.8883001 0 0)
  triclinic box = (0 0 0) to (8.8883001 10.8859 31.497821) with tilt (-8.8883001 0 0)
  triclinic box = (0 0 0) to (8.8883001 10.8859 31.497821) with tilt (-8.8883001 0 0)
  triclinic box = (0 0 0) to (8.8883001 10.8859 31.497821) with tilt (-8.8883001 0 0)
  triclinic box = (0 0 0) to (8.8883001 10.8859 31.497821) with tilt (-8.8883001 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.05107     -1283.6874      0              0            
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
change_box all     x final 0 8.886780092551124 y final 0 10.884038341536867 z final 0 31.492434328873735      xy final -8.886780092551122 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8867801 10.8859 31.497821) with tilt (-8.8883001 0 0)
  triclinic box = (0 0 0) to (8.8867801 10.884038 31.497821) with tilt (-8.8883001 0 0)
  triclinic box = (0 0 0) to (8.8867801 10.884038 31.492434) with tilt (-8.8883001 0 0)
  triclinic box = (0 0 0) to (8.8867801 10.884038 31.492434) with tilt (-8.8867801 0 0)
  triclinic box = (0 0 0) to (8.8867801 10.884038 31.492434) with tilt (-8.8867801 0 0)
  triclinic box = (0 0 0) to (8.8867801 10.884038 31.492434) with tilt (-8.8867801 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.10563     -1098.707       0              0            
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
change_box all     x final 0 8.886780092551124 y final 0 10.884038341536867 z final 0 31.49243432887373      xy final -8.886780092551122 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8867801 10.884038 31.492434) with tilt (-8.8867801 0 0)
  triclinic box = (0 0 0) to (8.8867801 10.884038 31.492434) with tilt (-8.8867801 0 0)
  triclinic box = (0 0 0) to (8.8867801 10.884038 31.492434) with tilt (-8.8867801 0 0)
  triclinic box = (0 0 0) to (8.8867801 10.884038 31.492434) with tilt (-8.8867801 0 0)
  triclinic box = (0 0 0) to (8.8867801 10.884038 31.492434) with tilt (-8.8867801 0 0)
  triclinic box = (0 0 0) to (8.8867801 10.884038 31.492434) with tilt (-8.8867801 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.10563     -1098.707       0              0            
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
change_box all     x final 0 8.887890153787351 y final 0 10.885397883342867 z final 0 31.496368096809793      xy final -8.88789015378735 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8878902 10.884038 31.492434) with tilt (-8.8867801 0 0)
  triclinic box = (0 0 0) to (8.8878902 10.885398 31.492434) with tilt (-8.8867801 0 0)
  triclinic box = (0 0 0) to (8.8878902 10.885398 31.496368) with tilt (-8.8867801 0 0)
  triclinic box = (0 0 0) to (8.8878902 10.885398 31.496368) with tilt (-8.8878902 0 0)
  triclinic box = (0 0 0) to (8.8878902 10.885398 31.496368) with tilt (-8.8878902 0 0)
  triclinic box = (0 0 0) to (8.8878902 10.885398 31.496368) with tilt (-8.8878902 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.11198     -1083.166       0              0            
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
change_box all     x final 0 8.887890153787351 y final 0 10.885397883342868 z final 0 31.49636809680979      xy final -8.88789015378735 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8878902 10.885398 31.496368) with tilt (-8.8878902 0 0)
  triclinic box = (0 0 0) to (8.8878902 10.885398 31.496368) with tilt (-8.8878902 0 0)
  triclinic box = (0 0 0) to (8.8878902 10.885398 31.496368) with tilt (-8.8878902 0 0)
  triclinic box = (0 0 0) to (8.8878902 10.885398 31.496368) with tilt (-8.8878902 0 0)
  triclinic box = (0 0 0) to (8.8878902 10.885398 31.496368) with tilt (-8.8878902 0 0)
  triclinic box = (0 0 0) to (8.8878902 10.885398 31.496368) with tilt (-8.8878902 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.11198     -1083.166       0              0            
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
change_box all     x final 0 8.888646618352386 y final 0 10.886324359439284 z final 0 31.499048810227738      xy final -8.888646618352384 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8886466 10.885398 31.496368) with tilt (-8.8878902 0 0)
  triclinic box = (0 0 0) to (8.8886466 10.886324 31.496368) with tilt (-8.8878902 0 0)
  triclinic box = (0 0 0) to (8.8886466 10.886324 31.499049) with tilt (-8.8878902 0 0)
  triclinic box = (0 0 0) to (8.8886466 10.886324 31.499049) with tilt (-8.8886466 0 0)
  triclinic box = (0 0 0) to (8.8886466 10.886324 31.499049) with tilt (-8.8886466 0 0)
  triclinic box = (0 0 0) to (8.8886466 10.886324 31.499049) with tilt (-8.8886466 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.11437     -1156.9704      0              0            
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
change_box all     x final 0 8.889538999320484 y final 0 10.88741729845329 z final 0 31.502211175982428      xy final -8.889538999320482 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.889539 10.886324 31.499049) with tilt (-8.8886466 0 0)
  triclinic box = (0 0 0) to (8.889539 10.887417 31.499049) with tilt (-8.8886466 0 0)
  triclinic box = (0 0 0) to (8.889539 10.887417 31.502211) with tilt (-8.8886466 0 0)
  triclinic box = (0 0 0) to (8.889539 10.887417 31.502211) with tilt (-8.889539 0 0)
  triclinic box = (0 0 0) to (8.889539 10.887417 31.502211) with tilt (-8.889539 0 0)
  triclinic box = (0 0 0) to (8.889539 10.887417 31.502211) with tilt (-8.889539 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.11536     -1243.5583      0              0            
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
change_box all     x final 0 8.889538999320484 y final 0 10.887417298453292 z final 0 31.502211175982428      xy final -8.889538999320482 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.889539 10.887417 31.502211) with tilt (-8.889539 0 0)
  triclinic box = (0 0 0) to (8.889539 10.887417 31.502211) with tilt (-8.889539 0 0)
  triclinic box = (0 0 0) to (8.889539 10.887417 31.502211) with tilt (-8.889539 0 0)
  triclinic box = (0 0 0) to (8.889539 10.887417 31.502211) with tilt (-8.889539 0 0)
  triclinic box = (0 0 0) to (8.889539 10.887417 31.502211) with tilt (-8.889539 0 0)
  triclinic box = (0 0 0) to (8.889539 10.887417 31.502211) with tilt (-8.889539 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.11536     -1243.5583      0              0            
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
change_box all     x final 0 8.890155480001324 y final 0 10.888172330005462 z final 0 31.50439582296977      xy final -8.890155480001322 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8901555 10.887417 31.502211) with tilt (-8.889539 0 0)
  triclinic box = (0 0 0) to (8.8901555 10.888172 31.502211) with tilt (-8.889539 0 0)
  triclinic box = (0 0 0) to (8.8901555 10.888172 31.504396) with tilt (-8.889539 0 0)
  triclinic box = (0 0 0) to (8.8901555 10.888172 31.504396) with tilt (-8.8901555 0 0)
  triclinic box = (0 0 0) to (8.8901555 10.888172 31.504396) with tilt (-8.8901555 0 0)
  triclinic box = (0 0 0) to (8.8901555 10.888172 31.504396) with tilt (-8.8901555 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.11742     -1280.2683      0              0            
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
change_box all     x final 0 8.89262183027755 y final 0 10.891192979857324 z final 0 31.513135926055142      xy final -8.892621830277548 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8926218 10.888172 31.504396) with tilt (-8.8901555 0 0)
  triclinic box = (0 0 0) to (8.8926218 10.891193 31.504396) with tilt (-8.8901555 0 0)
  triclinic box = (0 0 0) to (8.8926218 10.891193 31.513136) with tilt (-8.8901555 0 0)
  triclinic box = (0 0 0) to (8.8926218 10.891193 31.513136) with tilt (-8.8926218 0 0)
  triclinic box = (0 0 0) to (8.8926218 10.891193 31.513136) with tilt (-8.8926218 0 0)
  triclinic box = (0 0 0) to (8.8926218 10.891193 31.513136) with tilt (-8.8926218 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.12147     -1424.9807      0              0            
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
Nghost:            587 ave         587 max         587 min
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
change_box all     x final 0 8.89262183027755 y final 0 10.891192979857326 z final 0 31.51313592605514      xy final -8.892621830277548 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8926218 10.891193 31.513136) with tilt (-8.8926218 0 0)
  triclinic box = (0 0 0) to (8.8926218 10.891193 31.513136) with tilt (-8.8926218 0 0)
  triclinic box = (0 0 0) to (8.8926218 10.891193 31.513136) with tilt (-8.8926218 0 0)
  triclinic box = (0 0 0) to (8.8926218 10.891193 31.513136) with tilt (-8.8926218 0 0)
  triclinic box = (0 0 0) to (8.8926218 10.891193 31.513136) with tilt (-8.8926218 0 0)
  triclinic box = (0 0 0) to (8.8926218 10.891193 31.513136) with tilt (-8.8926218 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.12147     -1424.9807      0              0            
Loop time of 6.67e-07 on 1 procs for 0 steps with 80 atoms

299.9% CPU use with 1 MPI tasks x no OpenMP threads

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
Nghost:            587 ave         587 max         587 min
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
change_box all     x final 0 8.893588767911313 y final 0 10.892377231765234 z final 0 31.5165625012167      xy final -8.893588767911313 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8935888 10.891193 31.513136) with tilt (-8.8926218 0 0)
  triclinic box = (0 0 0) to (8.8935888 10.892377 31.513136) with tilt (-8.8926218 0 0)
  triclinic box = (0 0 0) to (8.8935888 10.892377 31.516563) with tilt (-8.8926218 0 0)
  triclinic box = (0 0 0) to (8.8935888 10.892377 31.516563) with tilt (-8.8935888 0 0)
  triclinic box = (0 0 0) to (8.8935888 10.892377 31.516563) with tilt (-8.8935888 0 0)
  triclinic box = (0 0 0) to (8.8935888 10.892377 31.516563) with tilt (-8.8935888 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.1222      -1495.5527      0              0            
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
Nghost:            587 ave         587 max         587 min
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
change_box all     x final 0 8.893588767911313 y final 0 10.892377231765234 z final 0 31.5165625012167      xy final -8.893588767911313 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8935888 10.892377 31.516563) with tilt (-8.8935888 0 0)
  triclinic box = (0 0 0) to (8.8935888 10.892377 31.516563) with tilt (-8.8935888 0 0)
  triclinic box = (0 0 0) to (8.8935888 10.892377 31.516563) with tilt (-8.8935888 0 0)
  triclinic box = (0 0 0) to (8.8935888 10.892377 31.516563) with tilt (-8.8935888 0 0)
  triclinic box = (0 0 0) to (8.8935888 10.892377 31.516563) with tilt (-8.8935888 0 0)
  triclinic box = (0 0 0) to (8.8935888 10.892377 31.516563) with tilt (-8.8935888 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.1222      -1495.5527      0              0            
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
change_box all     x final 0 8.893735940143563 y final 0 10.892557480201894 z final 0 31.517084041280807      xy final -8.893735940143563 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8937359 10.892377 31.516563) with tilt (-8.8935888 0 0)
  triclinic box = (0 0 0) to (8.8937359 10.892557 31.516563) with tilt (-8.8935888 0 0)
  triclinic box = (0 0 0) to (8.8937359 10.892557 31.517084) with tilt (-8.8935888 0 0)
  triclinic box = (0 0 0) to (8.8937359 10.892557 31.517084) with tilt (-8.8937359 0 0)
  triclinic box = (0 0 0) to (8.8937359 10.892557 31.517084) with tilt (-8.8937359 0 0)
  triclinic box = (0 0 0) to (8.8937359 10.892557 31.517084) with tilt (-8.8937359 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.12224     -1500.5489      0              0            
Loop time of 7.09e-07 on 1 procs for 0 steps with 80 atoms

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

Nlocal:             80 ave          80 max          80 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Nghost:            587 ave         587 max         587 min
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
change_box all     x final 0 8.89368913603383 y final 0 10.892500157108538 z final 0 31.51691817970476      xy final -8.89368913603383 xz final 0.0 yz final 0.0 units box
Changing box ...
  triclinic box = (0 0 0) to (8.8936891 10.892557 31.517084) with tilt (-8.8937359 0 0)
  triclinic box = (0 0 0) to (8.8936891 10.8925 31.517084) with tilt (-8.8937359 0 0)
  triclinic box = (0 0 0) to (8.8936891 10.8925 31.516918) with tilt (-8.8937359 0 0)
  triclinic box = (0 0 0) to (8.8936891 10.8925 31.516918) with tilt (-8.8936891 0 0)
  triclinic box = (0 0 0) to (8.8936891 10.8925 31.516918) with tilt (-8.8936891 0 0)
  triclinic box = (0 0 0) to (8.8936891 10.8925 31.516918) with tilt (-8.8936891 0 0)
echo none
run 0
Per MPI rank memory allocation (min/avg/max) = 3.33 | 3.33 | 3.33 Mbytes
    PotEng          Pxx           E_mol          E_coul    
-514.12224     -1498.9748      0              0            
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
Nghost:            587 ave         587 max         587 min
Histogram: 1 0 0 0 0 0 0 0 0 0
Neighs:           4288 ave        4288 max        4288 min
Histogram: 1 0 0 0 0 0 0 0 0 0

Total # of neighbors = 4288
Ave neighs/atom = 53.6
Neighbor list builds = 0
Dangerous builds = 0
Total wall time: 0:00:05
