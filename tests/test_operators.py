"""Verification of the discrete operators against exact analytical solutions.

Two kinds of checks:
* exactness: operators must reproduce linear (or, for the Laplacian,
  quadratic) fields to round-off on Cartesian grids;
* convergence: on refined grids the error must drop at the expected order.
"""
import numpy as np
import pytest

from fvm_helpers import (cartesian_grid, centers, divergence, gradient, laplacian,
                         load_grid, max_error, observed_order, rotor)

TOL = 1e-10
SIZES = [11, 21, 41]


# --------------------------------------------------------------------------
# Exactness on Cartesian grids
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["base", "fine"])
def test_gradient_exact_for_linear_field(name):
    NI, NJ, X, Y, m = load_grid(name)
    x, y = centers(m)
    G = gradient(NI, NJ, 2.0 * x - 3.0 * y + 1.0, m)
    assert max_error(G[..., 0], np.full_like(x, 2.0), NI, NJ) < TOL
    assert max_error(G[..., 1], np.full_like(x, -3.0), NI, NJ) < TOL


@pytest.mark.parametrize("name", ["base", "fine"])
def test_laplacian_exact_for_quadratic_field(name):
    NI, NJ, X, Y, m = load_grid(name)
    x, y = centers(m)
    P = x**2 + y**2 + x + y + 1.0
    L = laplacian(NI, NJ, P, gradient(NI, NJ, P, m), m)
    assert max_error(L, np.full_like(x, 4.0), NI, NJ) < 1e-8


@pytest.mark.parametrize("name", ["base", "fine"])
def test_divergence_and_rotor_exact_for_linear_field(name):
    NI, NJ, X, Y, m = load_grid(name)
    x, y = centers(m)
    V = np.stack([x + y, y + 1.0], axis=-1)    # div V = 2, rot V = -1
    assert max_error(divergence(NI, NJ, V, m), np.full_like(x, 2.0), NI, NJ) < TOL
    assert max_error(rotor(NI, NJ, V, m), np.full_like(x, -1.0), NI, NJ) < TOL


# --------------------------------------------------------------------------
# Skewed grids: iterative Green-Gauss and skewness correction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["skew", "1"])
def test_green_gauss_iterations_recover_linear_gradient_on_skewed_grid(name):
    NI, NJ, X, Y, m = load_grid(name)
    x, y = centers(m)
    P = x + y + 1.0
    exact = np.ones_like(x)
    err_1 = max_error(gradient(NI, NJ, P, m, iterations=1)[..., 0], exact, NI, NJ)
    err_10 = max_error(gradient(NI, NJ, P, m, iterations=10)[..., 0], exact, NI, NJ)
    assert err_10 < 1e-8
    assert err_10 < 1e-3 * err_1


@pytest.mark.parametrize("name", ["skew", "1"])
def test_laplacian_of_linear_field_vanishes_in_interior_of_skewed_grid(name):
    # skewness correction must make the interior Laplacian of a linear field zero;
    # cells next to the boundary are excluded (see "Known limitations" in README)
    NI, NJ, X, Y, m = load_grid(name)
    x, y = centers(m)
    P = x + y + 1.0
    L = laplacian(NI, NJ, P, gradient(NI, NJ, P, m, iterations=10), m)
    assert max_error(L, np.zeros_like(x), NI, NJ, ring=1) < 1e-7


# --------------------------------------------------------------------------
# Grid convergence on Cartesian grids with a smooth field
# --------------------------------------------------------------------------
def _convergence(kind):
    errors, errors_interior = [], []
    for n in SIZES:
        NI, NJ, X, Y, m = cartesian_grid(n)
        x, y = centers(m)
        P = np.sin(np.pi * x) * np.cos(np.pi * y)
        if kind == "gradient":
            num = gradient(NI, NJ, P, m, iterations=3)[..., 0]
            exact = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
        elif kind == "laplacian":
            num = laplacian(NI, NJ, P, gradient(NI, NJ, P, m, iterations=3), m)
            exact = -2.0 * np.pi**2 * P
        else:  # divergence
            V = np.stack([np.sin(np.pi * x), np.cos(np.pi * y)], axis=-1)
            num = divergence(NI, NJ, V, m)
            exact = np.pi * np.cos(np.pi * x) - np.pi * np.sin(np.pi * y)
        errors.append(max_error(num, exact, NI, NJ))
        errors_interior.append(max_error(num, exact, NI, NJ, ring=1))
    return observed_order(errors, SIZES), observed_order(errors_interior, SIZES)


def test_gradient_converges_second_order():
    order_all, _ = _convergence("gradient")
    assert order_all > 1.9


@pytest.mark.parametrize("kind", ["laplacian", "divergence"])
def test_interior_second_order_and_boundary_first_order(kind):
    order_all, order_interior = _convergence(kind)
    assert order_interior > 1.9   # second order away from boundaries
    assert order_all > 0.9        # at least first order including boundary cells
