import numpy as np


def CalcMetric(NI, NJ, X, Y, CellCenter, CellVolume, IFaceCenter, IFaceVector, JFaceCenter, JFaceVector):
    r = np.zeros((2), dtype=float)  # vector from one node to another
    # calculation of face centers and face vectors
    # for I-direction
    for J in range(NJ - 1):
        for I in range(NI):
            r[0] = X[I, J + 1] - X[I, J]  # rx
            r[1] = Y[I, J + 1] - Y[I, J]  # ry
            IFaceVector[I, J, 0] = r[1]  # IFaceVector equals r rotated on 90 degree
            IFaceVector[I, J, 1] = -r[0]  # IFaceVector directed to increasing I-index
            IFaceCenter[I, J, 0] = 0.5 * (X[I, J] + X[I, J + 1])  # x-component for iface center
            IFaceCenter[I, J, 1] = 0.5 * (Y[I, J] + Y[I, J + 1])  # y-component for iface center

    # for J-direction
    for J in range(NJ):
        for I in range(NI - 1):
            r[0] = X[I + 1, J] - X[I, J]  # rx
            r[1] = Y[I + 1, J] - Y[I, J]  # ry
            JFaceVector[I, J, 0] = -r[1]  # JFaceVector equals r rotated on -90 degree
            JFaceVector[I, J, 1] = r[0]  # JFaceVector directed to increasing J-index
            JFaceCenter[I, J, 0] = 0.5 * (X[I, J] + X[I+1, J])  # x-component for jface center
            JFaceCenter[I, J, 1] = 0.5 * (Y[I, J] + Y[I+1, J])  # y-component for jface center

    # calculation of cell volumes
    for J in range(NJ-1):
        for I in range(NI-1):
            r[0] = X[I+1, J+1] - X[I, J]
            r[1] = Y[I+1, J+1] - Y[I, J]
            CellVolume[I, J] = 0.5*np.dot(IFaceVector[I, J, :], r)+0.5*np.dot(JFaceVector[I, J, :], r)

    # calculation of cell centers
    # for inner cells: center of contour (sum of FaceCenter*FaceLength/Perimeter)
    for J in range(1, NJ):
        for I in range(1, NI):
            a1 = IFaceCenter[I - 1, J - 1, :]*np.linalg.norm(IFaceVector[I - 1, J - 1, :])
            a2 = IFaceCenter[I, J - 1, :]*np.linalg.norm(IFaceVector[I, J - 1, :])
            a3 = JFaceCenter[I - 1, J - 1, :] * np.linalg.norm(JFaceVector[I - 1, J - 1, :])
            a4 = JFaceCenter[I - 1, J, :]*np.linalg.norm(JFaceVector[I - 1, J, :])
            b1 = np.linalg.norm(IFaceVector[I - 1, J - 1, :]) + np.linalg.norm(IFaceVector[I, J - 1, :])
            b2 = np.linalg.norm(JFaceVector[I - 1, J - 1, :]) + np.linalg.norm(JFaceVector[I - 1, J, :])
            CellCenter[I, J, :] = (a1+a2+a3+a4) / (b1+b2)
    # for dummy cells on boundaries: cell center = face center
    # I - boundaries
    for J in range(1, NJ):
        CellCenter[0, J, :] = IFaceCenter[0, J - 1, :]
        CellCenter[-1, J, :] = IFaceCenter[-1, J - 1, :]

    # J - boundaries
    for I in range(1, NI):
        CellCenter[I, 0, :] = JFaceCenter[I - 1, 0, :]
        CellCenter[I, -1, :] = JFaceCenter[I - 1, -1, :]