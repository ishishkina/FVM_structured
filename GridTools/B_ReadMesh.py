"""Reading structured meshes and computing grid metrics.

Thin helpers around the original routines, so that the main script and the
test suite share one implementation.
"""
from collections import namedtuple

import numpy as np

from GridTools import B_CalcMetric

Metric = namedtuple(
    "Metric",
    ["CellCenter", "CellVolume", "IFaceCenter", "IFaceVector", "JFaceCenter", "JFaceVector"],
)


def read_mesh(path):
    """Read a structured 2D mesh file.

    The first line holds the node counts ``NI NJ``; it is followed by
    ``NI * NJ`` lines with ``x y`` node coordinates (I runs fastest).
    Returns ``NI, NJ, X, Y`` with X and Y of shape (NI, NJ).
    """
    with open(path, "r") as mesh:
        parts = mesh.readline().split()
        NI, NJ = int(parts[0]), int(parts[1])
        X = np.zeros((NI, NJ), dtype=float)
        Y = np.zeros((NI, NJ), dtype=float)
        for J in range(NJ):
            for I in range(NI):
                values = mesh.readline().split()
                X[I, J] = float(values[0])
                Y[I, J] = float(values[1])
    return NI, NJ, X, Y


def compute_metric(NI, NJ, X, Y):
    """Allocate metric arrays and fill them with B_CalcMetric.CalcMetric."""
    metric = Metric(
        CellCenter=np.zeros((NI + 1, NJ + 1, 2), dtype=float),
        CellVolume=np.zeros((NI - 1, NJ - 1), dtype=float),
        IFaceCenter=np.zeros((NI, NJ - 1, 2), dtype=float),
        IFaceVector=np.zeros((NI, NJ - 1, 2), dtype=float),
        JFaceCenter=np.zeros((NI - 1, NJ, 2), dtype=float),
        JFaceVector=np.zeros((NI - 1, NJ, 2), dtype=float),
    )
    B_CalcMetric.CalcMetric(NI, NJ, X, Y, *metric)
    return metric
