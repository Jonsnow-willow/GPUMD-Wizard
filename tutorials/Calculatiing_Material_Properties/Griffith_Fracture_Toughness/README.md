# Orientation-dependent Griffith fracture toughness

This tutorial turns the exploratory calculation in the source `crack.ipynb`
record notebook into a reproducible NumPy script. It evaluates the ideal
brittle-cleavage threshold

\[
G_I = 2\gamma_s, \qquad K_{IG}=\sqrt{G_I/B},
\]

where \(B\) is obtained from the rotated elastic compliance of a cubic
crystal. This is an ideal Griffith threshold, not an experimental \(K_{IC}\)
and not a simulation of crack-tip evolution.

## Coordinate convention

The local crack basis follows Hiremath *et al.*:

- \(x_1\): crack-propagation direction;
- \(x_2\): crack-plane normal;
- \(x_3=x_1\times x_2\): crack-front direction.

The script constructs and validates this orthonormal basis rather than taking
three independently specified directions. This corrects the ambiguous axis
ordering in the original notebook.

## Run the example

From the GPUMD-Wizard repository root:

```bash
python tutorials/Calculatiing_Material_Properties/Griffith_Fracture_Toughness/griffith_toughness.py
```

The input convention is:

- cubic elastic constants \(C_{11}, C_{12}, C_{44}\): GPa;
- surface and grain-boundary energies: J/m²;
- returned \(B\): Pa⁻¹;
- returned \(K\): MPa√m.

The W potential table is copied verbatim from that source notebook so that the
example remains traceable. Verify the table's original references before using
its numbers in a publication.

## About the `b66` convention

The default `reduction="hiremath"` reproduces Eq. (4) of Hiremath *et al.*,
which prints the coupling term as \(s_{26}\). The optional
`reduction="schur"` uses \(s_{36}\), corresponding to the general
plane-strain Schur complement for indices \((1,2,6)\). Keeping both choices
explicit is why this calculation belongs in a tutorial before it is promoted
to a stable Wizard API.

Reference: P. Hiremath *et al.*, “Effects of interatomic potential on fracture
behaviour in single- and bicrystalline tungsten,” *Computational Materials
Science* **207** (2022) 111283,
[doi:10.1016/j.commatsci.2022.111283](https://doi.org/10.1016/j.commatsci.2022.111283).
