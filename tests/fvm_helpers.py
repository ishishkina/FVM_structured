"""Shared helpers for the test suite: grids, operator wrappers, error norms."""
import os

import numpy as np

from CalcOperators import B_CalcDivergence, B_CalcGradient, B_CalcLaplacian, B_CalcRotor
from GridTools import B_ReadMesh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_DIR = os.path.join(REPO_ROOT, "Grids")


def load_grid(name):
    """Read a mesh from Grids/<name>.msh and compute its metric."""
    NI, NJ, X, Y = B_ReadMesh.read_mesh(os.path.join(GRID_DIR, f"{name}.msh"))
    return NI, NJ, X, Y, B_ReadMesh.compute_metric(NI, NJ, X, Y)


def cartesian_grid(n, length=1.0):
    """Uniform n x n node Cartesian grid on [0, length]^2."""
    s = np.linspace(0.0, length, n)
    X, Y = np.meshgrid(s, s, indexing="ij")
    X, Y = X.copy(), Y.copy()
    return n, n, X, Y, B_ReadMesh.compute_metric(n, n, X, Y)


def centers(metric):
    return metric.CellCenter[..., 0], metric.CellCenter[..., 1]


def gradient(NI, NJ, P, metric, iterations=1):
    GradP = np.zeros((NI + 1, NJ + 1, 2))
    for _ in range(iterations):
        B_CalcGradient.CalcGradient(NI, NJ, P, GradP, metric.CellVolume, metric.CellCenter,
                                    metric.IFaceCenter, metric.IFaceVector,
                                    metric.JFaceCenter, metric.JFaceVector)
    return GradP


def laplacian(NI, NJ, P, GradP, metric):
    LapP = np.zeros((NI + 1, NJ + 1))
    B_CalcLaplacian.CalcLaplacian(NI, NJ, P, GradP, LapP, metric.CellVolume, metric.CellCenter,
                                  metric.IFaceCenter, metric.IFaceVector,
                                  metric.JFaceCenter, metric.JFaceVector)
    return LapP


def divergence(NI, NJ, V, metric, mode=0, P=None, GradP=None):
    divV = np.zeros((NI + 1, NJ + 1))
    P = np.zeros((NI + 1, NJ + 1)) if P is None else P
    GradP = np.zeros((NI + 1, NJ + 1, 2)) if GradP is None else GradP
    B_CalcDivergence.CalcDivergence(mode, NI, NJ, V, divV, P, GradP, metric.CellVolume,
                                    metric.CellCenter, metric.IFaceCenter, metric.IFaceVector,
                                    metric.JFaceCenter, metric.JFaceVector)
    return divV


def rotor(NI, NJ, V, metric):
    rotV = np.zeros((NI + 1, NJ + 1))
    B_CalcRotor.CalcRotor(NI, NJ, V, rotV, metric.CellVolume, metric.CellCenter,
                          metric.IFaceCenter, metric.IFaceVector,
                          metric.JFaceCenter, metric.JFaceVector)
    return rotV


def max_error(numerical, exact, NI, NJ, ring=0):
    """Max abs error over computational cells, optionally skipping `ring`
    layers of cells next to the boundary."""
    a, b = 1 + ring, NI - ring
    c, d = 1 + ring, NJ - ring
    return float(np.max(np.abs(numerical[a:b, c:d] - exact[a:b, c:d])))


def observed_order(errors, sizes):
    """Convergence order from the two finest grids (h ~ 1 / (n - 1))."""
    h = [1.0 / (n - 1) for n in sizes]
    return float(np.log(errors[-2] / errors[-1]) / np.log(h[-2] / h[-1]))
