"""Grid metric: cell volumes, cell centres and face vectors."""
import numpy as np
import pytest

from fvm_helpers import cartesian_grid, load_grid


@pytest.mark.parametrize("name", ["base", "skew", "fine"])
def test_cell_volumes_sum_to_domain_area(name):
    # base, skew and fine all discretise the unit square
    *_, metric = load_grid(name)
    assert metric.CellVolume.sum() == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("name", ["base", "skew", "fine", "1"])
def test_cell_volumes_are_positive(name):
    *_, metric = load_grid(name)
    assert np.all(metric.CellVolume > 0.0)


def test_uniform_grid_cell_volumes_and_centres():
    n = 11
    h = 1.0 / (n - 1)
    NI, NJ, X, Y, metric = cartesian_grid(n)
    assert np.allclose(metric.CellVolume, h * h)
    expected = 0.5 * (X[:-1, :-1] + X[1:, 1:])
    assert np.allclose(metric.CellCenter[1:NI, 1:NJ, 0], expected)


@pytest.mark.parametrize("name", ["base", "skew"])
def test_face_vectors_of_each_cell_sum_to_zero(name):
    # a closed cell contour: sum of outward face vectors must vanish
    NI, NJ, X, Y, m = load_grid(name)
    total = (m.IFaceVector[1:, :, :] - m.IFaceVector[:-1, :, :]
             + m.JFaceVector[:, 1:, :] - m.JFaceVector[:, :-1, :])
    assert np.allclose(total, 0.0, atol=1e-14)
