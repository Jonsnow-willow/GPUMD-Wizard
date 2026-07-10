---
title: "GPUMD-Wizard: A Python package for atomistic simulation, potential training, and property evaluation"
tags:
  - Python
  - atomistic simulation
  - machine-learned interatomic potentials
  - GPUMD
  - materials workflows
authors:
  - name: Jiahui Liu
    affiliation: 1
    corresponding: true
affiliations:
  - index: 1
    name: Zhongguancun Academy, Beijing, China
date: 10 July 2026
bibliography: paper.bib
---

# Summary

Atomistic materials simulations require more than a molecular-dynamics engine or an interatomic potential. Researchers must also construct physically meaningful structures, introduce defects and deformations, evaluate whether a potential reproduces relevant material properties, prepare simulation directories, and retain the structures associated with each numerical result. These steps are commonly implemented as project-specific scripts, which makes them difficult to reuse and compare across materials.

GPUMD-Wizard is an open-source Python package that organizes these steps as scriptable materials workflows. It uses Atomic Simulation Environment (ASE) `Atoms` objects and calculators as its common interface [@Larsen2017jpcm] and connects them to Graphics Processing Units Molecular Dynamics (GPUMD) [@Fan2022jcp; @Xu2025mge] and neuroevolution potential (NEP) workflows [@Fan2021prb; @Fan2022jpcm]. The package supports alloy and defect structure preparation, ASE-calculator-based property evaluation, selected extended-XYZ dataset operations, GPUMD input staging, and PyTorch-based NEP4 training and artifact operations. Its purpose is not to replace ASE, GPUMD, calorine, or phonopy, but to express the physical choices that connect these tools in reusable research scripts.

# Statement of need

NEP models are commonly assessed first through energy, force, and virial errors on training and test data [@Fan2021prb; @Fan2022jpcm]. Such aggregate errors do not establish whether a model is suitable for a specific materials problem; independent benchmarks have exposed systematic errors in surfaces, defects, migration barriers, phonons, and high-energy states [@deng2025npjcm]. A potential used for radiation damage, for example, must also be inspected at short interatomic distances and for defects, migration paths, and highly distorted configurations [@Liu2023prb]. Alloy studies introduce further choices about lattice prototypes, finite-cell compositions, chemical reference states, and chemical disorder. Each choice affects the physical meaning of the result, yet it is often hidden inside a sequence of one-off scripts.

GPUMD-Wizard targets researchers who develop or apply NEP and other ASE-compatible interatomic potentials, particularly for metals, multicomponent alloys, defects, and irradiation-related simulations. It addresses the workflow between a structure or training set and a production calculation. The package provides reusable operations for constructing and perturbing atomic configurations, evaluating physically interpretable properties, selecting candidate training frames, and preparing GPUMD runs. For methods that produce both structures and scalar summaries, it writes them side by side so that they can be inspected together.

Energies and forces are supplied by the selected calculator, while the reliability of a result also depends on reference states, supercell sizes, boundary conditions, workflow definitions, and convergence parameters. GPUMD-Wizard makes these decisions visible at the Python-workflow level, where they can be inspected, changed, and reused.

# Software design

The design follows four related choices. First, ASE `Atoms` is the common data plane. Structures generated for bulk crystals, random alloys, interstitials, vacancies, predefined body-centered-cubic screw-dislocation configurations, or primary knock-on atoms remain ordinary ASE objects. Structures can therefore enter ASE-compatible workflows without a package-specific structure type; GPUMD-Wizard-specific forces, groups, magnetic moments, and labels use explicit `atoms.info` conventions.

Second, the public-facing code is organized around physical tasks. Structure objects describe composition and lattice prototypes, while morphology operations make explicit changes to an existing configuration. Property workflows accept an ASE calculator and produce quantities such as equations of state, elastic constants, phonons, surfaces, defect formation energies, migration barriers, and generalized stacking faults. The implementation delegates relaxation, force evaluation, elastic analysis, phonons, and nudged-elastic-band calculations to established libraries. This keeps the workflow close to the quantity being calculated and leaves the calculator replaceable where it supplies the required ASE properties.

Third, GPUMD-Wizard uses files as transparent workflow boundaries. Many property methods append generated configurations to `MaterialProperties.xyz` and scalar summaries to `MaterialProperties.out`, while curve and phonon workflows use task-specific outputs. Extended-XYZ helpers handle cells, periodicity, and selected metadata---including energy, stress and derived virial, forces, velocities, groups, magnetic moments, configuration labels, and weights---under the package's `atoms.info` conventions. GPUMD preparation writes `model.xyz` and `run.in` and copies a specified NEP file plus an optional electron-stopping file into a run directory. These outputs are deliberately simple: they aid inspection and downstream processing without claiming to be a complete provenance database.

Fourth, the TorchNEP subsystem is separated from the direct materials workflows because training has different engineering requirements. It organizes checkpoints, epoch metrics, GPUMD-format `nep.txt` exports, and evaluations around a run directory containing `nep.in` and extended-XYZ data. Within its supported NEP4 subset, command-line operations train or resume a model, inspect checkpoints, evaluate an artifact, export a text model, and report prediction differences between checkpoint and text artifacts on the same structures. Checkpoints retain optimizer, scheduler, and random-number states. This lifecycle makes a stopped or remote training run inspectable and allows an export to be checked against its source within TorchNEP. The current parser does not support other NEP versions or typewise ZBL cutoffs.

# Research impact statement

Version 1.0 of GPUMD-Wizard was archived in 2024 [@wizard]. The UNEP-v1 study used GPUMD-Wizard to calculate energetics, elastic properties, and phonon dispersion relations for a general-purpose potential covering 16 metals and their alloys, and cited the archived release [@Song2024nc; @wizard]. The GPUMD 4.0 ecosystem article subsequently described GPUMD-Wizard as a package for automated material-property and GPUMD workflows [@Xu2025mge]. More recently, the NEP89 study used GPUMD-Wizard in its static-property benchmarks and cited the archived software release [@liang2026ncs; @wizard].

Published applications span tungsten radiation damage [@Liu2023prb], a 16-metal general-purpose potential [@Song2024nc], and an 89-element foundation potential [@liang2026ncs]. The repository complements these publications with worked scripts and selected outputs for property evaluation, structure generation, GPUMD preparation, and TorchNEP model handling. Together, this record documents use across element-specific, multicomponent, and foundation-potential projects.

# Acknowledgements

The author acknowledges the developers and maintainers of ASE, GPUMD, calorine, and phonopy, on which GPUMD-Wizard builds.

# References
