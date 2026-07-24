"""Analytical NEP descriptor force and virial contraction.

The formulas follow GPUMD's ``nep_utilities.cuh``. Geometry is treated as
fixed data, while the returned forces remain differentiable with respect to
descriptor coefficients and energy-model parameters.
"""

import torch


def _type_contraction(basis, center, neighbor, types, coefficients):
    if center.numel() == 0:
        return basis.new_zeros((0, coefficients.shape[2]))
    pair_coefficients = coefficients[types[center], types[neighbor]]
    return torch.einsum("pnk,pk->pn", pair_coefficients, basis)


def _extra_gradient(moments, coefficients, output, left, right):
    values = (
        coefficients.to(dtype=moments.dtype)
        * moments.index_select(2, left)
        * moments.index_select(2, right)
    )
    indices = output.view(1, 1, -1).expand_as(values)
    return torch.zeros_like(moments).scatter_add(2, indices, values)


def _angular_weight(descriptor, energy_gradient, moments):
    angular = descriptor.angular
    n_atoms, n_desc, _ = moments.shape
    weight = torch.zeros_like(moments)
    radial_dim = descriptor.radial.n_desc
    offset = radial_dim

    three_body = energy_gradient[
        :, offset : offset + angular.L_max * n_desc
    ].reshape(n_atoms, angular.L_max, n_desc)
    offset += angular.L_max * n_desc

    for angular_index in range(angular.L_max):
        order = angular_index + 1
        start = order * order - 1
        terms = 2 * order + 1
        coefficients = angular.C3B[start : start + terms].to(moments)
        moment_block = moments[:, :, start : start + terms]
        factors = torch.full_like(coefficients, 4.0)
        factors[0] = 2.0
        derivative = factors * coefficients * moment_block
        weight[:, :, start : start + terms] = (
            weight[:, :, start : start + terms]
            + three_body[:, angular_index].unsqueeze(-1) * derivative
        )

    s10, s11r, s11i = moments[:, :, 0], moments[:, :, 1], moments[:, :, 2]
    if moments.shape[2] >= 8:
        s20, s21r, s21i = moments[:, :, 3], moments[:, :, 4], moments[:, :, 5]
        s22r, s22i = moments[:, :, 6], moments[:, :, 7]

    if angular.has_q_222:
        gradient = energy_gradient[:, offset : offset + n_desc]
        offset += n_desc
        c = angular.C4B.to(moments)
        weight[:, :, 3] = weight[:, :, 3] + gradient * (
            3 * c[0] * s20**2
            + c[1] * (s21r**2 + s21i**2)
            + c[2] * (s22r**2 + s22i**2)
        )
        weight[:, :, 4] = weight[:, :, 4] + gradient * (
            2 * c[1] * s20 * s21r - 2 * c[3] * s22r * s21r + c[4] * s21i * s22i
        )
        weight[:, :, 5] = weight[:, :, 5] + gradient * (
            2 * c[1] * s20 * s21i + 2 * c[3] * s22r * s21i + c[4] * s21r * s22i
        )
        weight[:, :, 6] = weight[:, :, 6] + gradient * (
            2 * c[2] * s20 * s22r + c[3] * (s21i**2 - s21r**2)
        )
        weight[:, :, 7] = weight[:, :, 7] + gradient * (
            2 * c[2] * s20 * s22i + c[4] * s21r * s21i
        )

    if angular.has_q_1111:
        gradient = energy_gradient[:, offset : offset + n_desc]
        offset += n_desc
        c = angular.C5B.to(moments)
        s0_squared = s10**2
        s1_squared = s11r**2 + s11i**2
        factor = c[1] * s0_squared + 2 * c[2] * s1_squared
        weight[:, :, 0] = weight[:, :, 0] + gradient * (
            2 * s10 * (2 * c[0] * s0_squared + c[1] * s1_squared)
        )
        weight[:, :, 1] = weight[:, :, 1] + gradient * 2 * s11r * factor
        weight[:, :, 2] = weight[:, :, 2] + gradient * 2 * s11i * factor

    if angular.has_q_112:
        gradient = energy_gradient[:, offset : offset + n_desc]
        offset += n_desc
        c = angular.C4B2.to(moments)
        a, b, cc = s10, s11r, s11i
        d, e, f, g, h = s20, s21r, s21i, s22r, s22i
        weight[:, :, 0] = weight[:, :, 0] + gradient * (
            2 * c[0] * a * d + c[1] * (b * e + cc * f)
        )
        weight[:, :, 1] = weight[:, :, 1] + gradient * (
            c[1] * a * e + 2 * c[2] * d * b + 2 * c[3] * g * b + c[4] * cc * h
        )
        weight[:, :, 2] = weight[:, :, 2] + gradient * (
            c[1] * a * f + 2 * c[2] * d * cc - 2 * c[3] * g * cc + c[4] * b * h
        )
        weight[:, :, 3] = weight[:, :, 3] + gradient * (
            c[0] * a * a + c[2] * (b * b + cc * cc)
        )
        weight[:, :, 4] = weight[:, :, 4] + gradient * c[1] * a * b
        weight[:, :, 5] = weight[:, :, 5] + gradient * c[1] * a * cc
        weight[:, :, 6] = weight[:, :, 6] + gradient * c[3] * (b * b - cc * cc)
        weight[:, :, 7] = weight[:, :, 7] + gradient * c[4] * b * cc

    for enabled, name in (
        (angular.has_q_123, "q123"),
        (angular.has_q_233, "q233"),
        (angular.has_q_134, "q134"),
    ):
        if enabled:
            gradient = energy_gradient[:, offset : offset + n_desc]
            offset += n_desc
            weight = weight + gradient.unsqueeze(-1) * _extra_gradient(
                moments,
                getattr(angular, f"{name}_gradient_coefficients"),
                getattr(angular, f"{name}_gradient_output"),
                getattr(angular, f"{name}_gradient_left"),
                getattr(angular, f"{name}_gradient_right"),
            )

    return weight


def _angular_basis_derivative(
    x,
    y,
    z,
    exponents,
    coefficients,
):
    features = (
        x.unsqueeze(-1).pow(exponents[:, 0])
        * y.unsqueeze(-1).pow(exponents[:, 1])
        * z.unsqueeze(-1).pow(exponents[:, 2])
    )
    return torch.einsum("pm,xmc->pcx", features, coefficients)


def _accumulate_pair_gradient(forces, virial, pair, gradient):
    forces.scatter_add_(0, pair.center.unsqueeze(-1).expand_as(gradient), gradient)
    forces.scatter_add_(0, pair.neighbor.unsqueeze(-1).expand_as(gradient), -gradient)
    if virial is not None:
        pair_virial = -(
            pair.vectors.unsqueeze(-1) * gradient.unsqueeze(-2)
        ).reshape(-1, 9)
        virial.scatter_add_(
            0,
            pair.neighbor.unsqueeze(-1).expand_as(pair_virial),
            pair_virial,
        )


def contract_forces(descriptor, energy_gradient, state, compute_virial=True):
    """Contract ``dE/dq`` with analytical descriptor coordinate derivatives."""
    n_atoms = energy_gradient.shape[0]
    forces = energy_gradient.new_zeros((n_atoms, 3))
    virial = energy_gradient.new_zeros((n_atoms, 9)) if compute_virial else None

    radial = state.radial
    radial_derivative = _type_contraction(
        radial.basis_derivative,
        radial.center,
        radial.neighbor,
        state.types,
        descriptor.radial.c_table,
    )
    if radial.center.numel():
        radial_scalar = (
            energy_gradient[radial.center, : descriptor.radial.n_desc]
            * radial_derivative
        ).sum(-1)
        radial_gradient = (
            radial_scalar * radial.inverse_distance
        ).unsqueeze(-1) * radial.vectors
        _accumulate_pair_gradient(forces, virial, radial, radial_gradient)

    angular = state.angular
    if angular.center.numel():
        angular_weight = _angular_weight(
            descriptor, energy_gradient, angular.moments
        )
        pair_weight = angular_weight[angular.center]
        angular_radial_derivative = _type_contraction(
            angular.basis_derivative,
            angular.center,
            angular.neighbor,
            state.types,
            descriptor.angular.c_table,
        )
        radial_basis_gradient = (
            angular_radial_derivative.unsqueeze(-1)
            * angular.basis.unsqueeze(1)
        )
        radial_scalar = (pair_weight * radial_basis_gradient).sum((1, 2))
        radial_part = (
            radial_scalar * angular.inverse_distance
        ).unsqueeze(-1) * angular.vectors

        unit = angular.vectors * angular.inverse_distance.unsqueeze(-1)
        with torch.no_grad():
            basis_derivative = _angular_basis_derivative(
                unit[:, 0],
                unit[:, 1],
                unit[:, 2],
                descriptor.angular.basis_derivative_exponents,
                descriptor.angular.basis_derivative_coefficients,
            )
        weighted_radial = (
            pair_weight * angular.radial_values.unsqueeze(-1)
        ).sum(1)
        direction_part = (
            weighted_radial.unsqueeze(-1) * basis_derivative
        ).sum(1) * angular.inverse_distance.unsqueeze(-1)
        projected = (
            weighted_radial
            * (unit.unsqueeze(1) * basis_derivative).sum(-1)
        ).sum(1) * angular.inverse_distance
        angular_gradient = (
            radial_part + direction_part - projected.unsqueeze(-1) * unit
        )
        _accumulate_pair_gradient(forces, virial, angular, angular_gradient)

    return forces, virial
