"""Orientation-dependent Griffith fracture toughness for cubic crystals.

This tutorial migrates the calculation from the source ``crack.ipynb`` record
notebook into a small, reproducible NumPy script.  The local crack coordinates
are

    x1: crack-propagation direction
    x2: crack-plane normal
    x3: crack-front direction = x1 cross x2

Elastic constants are supplied in GPa, surface energies in J/m^2, and the
reported stress-intensity factor is in MPa sqrt(m).

The default ``b66`` reduction follows Eq. (4) of Hiremath et al., Comput.
Mater. Sci. 207 (2022) 111283.  Set ``reduction="schur"`` to use the general
plane-strain Schur complement, whose b66 term contains s36 instead of s26.
"""

from __future__ import annotations

from math import sqrt

import numpy as np


VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def cubic_stiffness_voigt(c11_gpa: float, c12_gpa: float, c44_gpa: float) -> np.ndarray:
    """Return the 6x6 engineering-Voigt stiffness matrix of a cubic crystal."""
    stiffness = np.zeros((6, 6), dtype=float)
    stiffness[:3, :3] = c12_gpa
    np.fill_diagonal(stiffness[:3, :3], c11_gpa)
    np.fill_diagonal(stiffness[3:, 3:], c44_gpa)
    return stiffness


def voigt_to_tensor4(stiffness: np.ndarray) -> np.ndarray:
    """Expand a stiffness matrix from engineering Voigt form to C_ijkl."""
    stiffness = np.asarray(stiffness, dtype=float)
    if stiffness.shape != (6, 6):
        raise ValueError("stiffness must have shape (6, 6)")

    tensor = np.zeros((3, 3, 3, 3), dtype=float)
    for alpha, (i, j) in enumerate(VOIGT_PAIRS):
        for beta, (k, l) in enumerate(VOIGT_PAIRS):
            value = stiffness[alpha, beta]
            tensor[i, j, k, l] = value
            tensor[j, i, k, l] = value
            tensor[i, j, l, k] = value
            tensor[j, i, l, k] = value
    return tensor


def tensor4_to_voigt(tensor: np.ndarray) -> np.ndarray:
    """Contract C_ijkl to a 6x6 engineering-Voigt stiffness matrix."""
    tensor = np.asarray(tensor, dtype=float)
    if tensor.shape != (3, 3, 3, 3):
        raise ValueError("tensor must have shape (3, 3, 3, 3)")

    stiffness = np.empty((6, 6), dtype=float)
    for alpha, (i, j) in enumerate(VOIGT_PAIRS):
        for beta, (k, l) in enumerate(VOIGT_PAIRS):
            stiffness[alpha, beta] = tensor[i, j, k, l]
    return stiffness


def crack_basis(propagation: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """Build rows [x1, x2, x3] of the crack basis in crystal coordinates."""
    x1 = np.array(propagation, dtype=float, copy=True)
    x2 = np.array(plane_normal, dtype=float, copy=True)
    if x1.shape != (3,) or x2.shape != (3,):
        raise ValueError("propagation and plane_normal must be three-vectors")
    if np.linalg.norm(x1) == 0.0 or np.linalg.norm(x2) == 0.0:
        raise ValueError("crack directions must be non-zero")

    x1 /= np.linalg.norm(x1)
    x2 -= np.dot(x2, x1) * x1
    if np.linalg.norm(x2) < 1.0e-12:
        raise ValueError("propagation and plane_normal must not be parallel")
    x2 /= np.linalg.norm(x2)
    x3 = np.cross(x1, x2)
    return np.vstack((x1, x2, x3))


def rotate_stiffness(stiffness: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Rotate stiffness from crystal axes into the local crack basis."""
    basis = np.asarray(basis, dtype=float)
    if basis.shape != (3, 3) or not np.allclose(basis @ basis.T, np.eye(3), atol=1.0e-10):
        raise ValueError("basis must be a 3x3 orthonormal matrix")
    tensor = voigt_to_tensor4(stiffness)
    rotated = np.einsum("ai,bj,ck,dl,ijkl->abcd", basis, basis, basis, basis, tensor)
    return tensor4_to_voigt(rotated)


def crack_compliance(
    stiffness_gpa: np.ndarray,
    propagation: np.ndarray,
    plane_normal: np.ndarray,
    *,
    reduction: str = "hiremath",
) -> dict[str, float]:
    """Return the reduced crack compliance B and its intermediate terms.

    ``reduction="hiremath"`` reproduces the published s26 term.  The
    ``"schur"`` option uses s36, as obtained by eliminating direction 3 from
    the compliance submatrix indexed by (1, 2, 6).
    """
    rotated_pa = (
        rotate_stiffness(stiffness_gpa, crack_basis(propagation, plane_normal))
        * 1.0e9
    )
    compliance = np.linalg.inv(rotated_pa)

    s11, s22, s33 = compliance[0, 0], compliance[1, 1], compliance[2, 2]
    s12, s13, s23 = compliance[0, 1], compliance[0, 2], compliance[1, 2]
    s66 = compliance[5, 5]
    if reduction == "hiremath":
        shear_coupling = compliance[1, 5]  # s26, as printed in Hiremath Eq. (4)
    elif reduction == "schur":
        shear_coupling = compliance[2, 5]  # s36, general plane-strain reduction
    else:
        raise ValueError("reduction must be 'hiremath' or 'schur'")

    b11 = s11 - s13 * s13 / s33
    b22 = s22 - s23 * s23 / s33
    b12 = s12 - s13 * s23 / s33
    b66 = s66 - shear_coupling * shear_coupling / s33
    radicand = 0.5 * b11 * b22 * (
        sqrt(b22 / b11) + (2.0 * b12 + b66) / (2.0 * b11)
    )
    if radicand <= 0.0:
        raise ValueError("elastic constants do not give a positive crack compliance")

    return {
        "B_1_per_pa": sqrt(radicand),
        "b11": b11,
        "b22": b22,
        "b12": b12,
        "b66": b66,
    }


def griffith_cleavage_toughness(
    stiffness_gpa: np.ndarray,
    surface_energy_j_m2: float,
    propagation: np.ndarray,
    plane_normal: np.ndarray,
    *,
    reduction: str = "hiremath",
) -> dict[str, float]:
    """Calculate ideal cleavage G_I and K_IG for one crack orientation."""
    if surface_energy_j_m2 <= 0.0:
        raise ValueError("surface_energy_j_m2 must be positive")
    result = crack_compliance(
        stiffness_gpa, propagation, plane_normal, reduction=reduction
    )
    energy_release_rate = 2.0 * surface_energy_j_m2
    result.update(
        {
            "G_I_j_m2": energy_release_rate,
            "K_IG_mpa_sqrt_m": sqrt(energy_release_rate / result["B_1_per_pa"]) / 1.0e6,
        }
    )
    return result


def griffith_gb_toughness(
    crack_compliance_1_per_pa: float,
    surface_energy_1_j_m2: float,
    surface_energy_2_j_m2: float,
    grain_boundary_energy_j_m2: float,
) -> float:
    """Return ideal grain-boundary fracture toughness in MPa sqrt(m)."""
    separation_energy = (
        surface_energy_1_j_m2 + surface_energy_2_j_m2 - grain_boundary_energy_j_m2
    )
    if crack_compliance_1_per_pa <= 0.0 or separation_energy <= 0.0:
        raise ValueError(
            "crack compliance and grain-boundary separation energy must be positive"
        )
    return sqrt(separation_energy / crack_compliance_1_per_pa) / 1.0e6


# Example values migrated verbatim from the source crack.ipynb record notebook.
# Confirm the original references before using this table in a publication.
POTENTIALS = {
    "BOP-Juslin": (542.0, 191.0, 162.0, {"001": 1.446, "110": 0.931, "111": 1.720}),
    "BOP-Li": (515.0, 188.0, 162.0, {"001": 1.587, "110": 0.948, "111": 3.222}),
    "EAM-Wang": (544.0, 208.0, 158.0, {"001": 2.721, "110": 2.306, "111": 2.963}),
    "EAM-Mason": (516.0, 201.0, 146.0, {"001": 3.900, "110": 3.274, "111": 4.261}),
    "DP-WH": (522.0, 204.0, 161.0, {"001": 2.990, "110": 3.220, "111": 3.563}),
    "Previous-DFT": (523.0, 203.0, 160.0, {"001": 4.021, "110": 3.268, "111": 3.556}),
    "NEP-WH": (534.0, 199.0, 156.0, {"001": 4.007, "110": 3.180, "111": 3.491}),
}

CRACK_SYSTEMS = (
    ("(001)[0-10]", (0, -1, 0), (0, 0, 1), "001"),
    ("(001)[1-10]", (1, -1, 0), (0, 0, 1), "001"),
    ("(110)[1-10]", (1, -1, 0), (1, 1, 0), "110"),
    ("(01-1)[100]", (1, 0, 0), (0, 1, -1), "110"),
    ("(111)[112-]", (1, 1, -2), (1, 1, 1), "111"),
    ("(111)[1-10]", (-1, 1, 0), (1, 1, 1), "111"),
)


def main() -> None:
    """Print the migrated tungsten example as a compact comparison table."""
    header = (
        f"{'Potential':<14} {'Crack system':<16} {'B (1/TPa)':>11} "
        f"{'K_IG (MPa sqrt(m))':>20}"
    )
    print(header)
    print("-" * len(header))
    for name, (c11, c12, c44, surface_energies) in POTENTIALS.items():
        stiffness = cubic_stiffness_voigt(c11, c12, c44)
        for label, propagation, normal, surface_key in CRACK_SYSTEMS:
            result = griffith_cleavage_toughness(
                stiffness,
                surface_energies[surface_key],
                np.asarray(propagation),
                np.asarray(normal),
            )
            print(
                f"{name:<14} {label:<16} "
                f"{result['B_1_per_pa'] * 1.0e12:11.5f} "
                f"{result['K_IG_mpa_sqrt_m']:20.5f}"
            )


if __name__ == "__main__":
    main()
