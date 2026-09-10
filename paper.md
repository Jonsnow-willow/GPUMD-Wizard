---
title: "GPUMD-Wizard: A Python package for atomistic modeling and material-property evaluation"
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

GPUMD-Wizard is an open-source Python package that organizes these steps as scriptable materials workflows. It uses Atomic Simulation Environment (ASE) structures and calculators as its common interface [@Larsen2017jpcm] and connects them to Graphics Processing Units Molecular Dynamics (GPUMD) [@Fan2022jcp; @Xu2025mge] and neuroevolution potential (NEP) models [@Fan2021prb; @Fan2022jpcm]. The package supports alloy and defect structure preparation, property evaluation with ASE-compatible calculators, selected extended-XYZ dataset operations, molecular dynamics and Monte Carlo sampling, and GPUMD input preparation. It brings the physical choices linking these operations into reusable research scripts.

# Statement of need

NEP models are commonly assessed first through energy, force, and virial errors on training and test data [@Fan2021prb; @Fan2022jpcm]. Such aggregate errors do not establish whether a model is suitable for a specific materials problem; independent benchmarks have exposed systematic errors in surfaces, defects, migration barriers, phonons, and high-energy states [@deng2025npjcm]. A potential used for radiation damage, for example, must also be inspected at short interatomic distances and for defects, migration paths, and highly distorted configurations [@Liu2023prb]. Alloy studies introduce further choices about lattice prototypes, finite-cell compositions, chemical reference states, and chemical disorder. Each choice affects the physical meaning of the result, yet it is often hidden inside a sequence of one-off scripts.

GPUMD-Wizard targets researchers who develop or apply NEP and other ASE-compatible interatomic potentials, particularly for metals, multicomponent alloys, defects, and irradiation-related simulations. It links structure and dataset preparation to property calculations and GPUMD simulations. The package provides reusable operations for constructing and perturbing atomic configurations, evaluating physically interpretable properties, selecting candidate training frames, and preparing GPUMD runs. For methods that produce both structures and scalar summaries, it writes them side by side so that they can be inspected together.

Energies and forces are supplied by the selected calculator, while the reliability of a result also depends on reference states, supercell sizes, boundary conditions, workflow definitions, and convergence parameters. GPUMD-Wizard makes these decisions visible at the Python-workflow level, where they can be inspected, changed, and reused.

# State of the field

The atomistic-simulation ecosystem provides complementary tools for structures, force evaluation, and analysis. ASE supplies a common structure representation, calculator interfaces, and simulation algorithms [@Larsen2017jpcm]. GPUMD implements NEP model construction and GPU-accelerated molecular dynamics [@Fan2022jcp; @Xu2025mge]. calorine connects NEP models to Python and ASE, supports GPUMD input and output, and provides model analysis and sampling workflows [@Lindgren2024joss]. Phonopy supplies methods for phonon calculations and related thermal properties [@Togo2023JPCM]. These packages already support many of the calculations used in potential assessment.

GPUMD-Wizard organizes these capabilities around materials questions, particularly alloy configurations, defects, and irradiation. A defect-energy workflow, for example, couples a reference crystal and a defective configuration with relaxation, an energy reference, and recording of the resulting structures and energies. The package brings such choices together with composition handling, structure perturbations, frame selection, and GPUMD run preparation. These requirements span several libraries. Maintaining the coupled workflows in a separate package allows their physical assumptions and output conventions to evolve together, while retaining ASE objects and calculators as shared interfaces. The property workflows reuse structure operations, relaxation algorithms, and analysis routines from the underlying libraries. GPUMD-Wizard provides the material-specific sequences that connect a configuration to a property estimate or simulation input.

# Software design

The design follows three related choices. First, ASE `Atoms` is the common structure representation. Structures generated for bulk crystals, random alloys, interstitials, vacancies, predefined body-centered-cubic screw-dislocation configurations, or primary knock-on atoms remain ordinary ASE objects. Structures can therefore enter ASE-compatible workflows without a package-specific structure type; GPUMD-Wizard-specific forces, groups, magnetic moments, and labels use explicit `atoms.info` conventions.

Second, the public-facing code is organized around physical tasks. Structure objects describe composition and lattice prototypes, while morphology operations make explicit changes to an existing configuration. Property workflows accept an ASE calculator and produce quantities such as equations of state, elastic constants, phonons, surfaces, defect formation energies, migration barriers, and generalized stacking faults. The implementation delegates relaxation, force evaluation, elastic analysis, phonons, and nudged-elastic-band calculations to established libraries. Separate drivers provide calculator-based molecular dynamics and interleave molecular dynamics with Monte Carlo moves for sampling. This organization keeps the physical workflow explicit and leaves the calculator replaceable where it supplies the required ASE properties.

Third, GPUMD-Wizard uses files as transparent workflow boundaries. Many property methods append generated configurations to `MaterialProperties.xyz` and scalar summaries to `MaterialProperties.out`, while curve and phonon workflows use task-specific outputs. Extended-XYZ helpers handle cells, periodicity, and selected metadata---including energy, stress and derived virial, forces, velocities, groups, magnetic moments, configuration labels, and weights---under the package's `atoms.info` conventions. GPUMD preparation writes `model.xyz` and `run.in` and copies a specified NEP file plus an optional electron-stopping file into a run directory. The same workflow can launch the external GPUMD executable. These outputs aid inspection and downstream processing, while run settings and convergence choices remain explicit in the research scripts.

# Research impact statement

Version 1.0 of GPUMD-Wizard was archived in 2024 [@wizard]. The UNEP-v1 study used GPUMD-Wizard to calculate energetics, elastic properties, and phonon dispersion relations for a general-purpose potential covering 16 metals and their alloys, and cited the archived release [@Song2024nc; @wizard]. The GPUMD 4.0 ecosystem article subsequently described GPUMD-Wizard as a package for automated material-property and GPUMD workflows [@Xu2025mge]. More recently, the NEP89 study used GPUMD-Wizard in its static-property benchmarks and cited the archived software release [@liang2026ncs; @wizard].

The repository complements these publications with worked scripts and selected outputs for property evaluation, structure generation, dataset processing, and GPUMD preparation. Examples include elemental-metal and multicomponent-alloy calculations, connecting potential files and atomic structures to property summaries and simulation inputs. The published applications and accompanying examples document use across element-specific, multicomponent, and foundation-potential projects.

# AI usage disclosure

The author primarily wrote the GPUMD-Wizard software and made its core design decisions. OpenAI Codex, using GPT-5.5, GPT-5.6, and GPT-6, assisted mainly with code completion and was also used to generate test code. Codex was used to draft and revise the manuscript and to revise repository documentation. The author is responsible for reviewing, editing, and validating AI-assisted code, tests, and manuscript text, and retains responsibility for the software and this paper.

# Acknowledgements

The author acknowledges the developers and maintainers of ASE, GPUMD, calorine, and phonopy, on which GPUMD-Wizard builds.

# References
