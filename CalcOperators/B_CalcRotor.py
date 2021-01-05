import numpy as np
from Functions import B_Functions


def CalcRotor(NI, NJ, V, rotV, CellVolume, CellCenter, IFaceCenter, IFaceVector, JFaceCenter, JFaceVector):

    # initialization
    RF = np.zeros((4, 2), dtype=float)  # radius-vector for face-center
    SF = np.zeros((4, 2), dtype=float)  # face vector
    RC = np.zeros((2), dtype=float)  # radius-vector for cell-center
    RN = np.zeros((2), dtype=float)  # radius-vector for cell-center of adjacent cell

    for I in range(1, NI):
        for J in range(1, NJ):
            NCELL = np.array([[I - 1, J], [I + 1, J],  [I, J - 1], [I, J + 1]], dtype=int)

            # calculation of radius-vector for face-center
            RF[0, :] = IFaceCenter[I - 1, J - 1, :]
            RF[1, :] = IFaceCenter[I, J - 1, :]
            RF[2, :] = JFaceCenter[I - 1, J - 1, :]
            RF[3, :] = JFaceCenter[I - 1, J, :]

            # calculation of face vector
            SF[0, :] = -IFaceVector[I - 1, J - 1, :]
            SF[1, :] = IFaceVector[I, J - 1, :]
            SF[2, :] = -JFaceVector[I - 1, J - 1, :]
            SF[3, :] = JFaceVector[I - 1, J, :]

            # calculation of cell volume
            VOL = CellVolume[I - 1, J - 1]
            # calculation of radius-vector for cell center
            RC[:] = CellCenter[I, J, :]

            # loop for faces
            for IFACE in range(4):
                IN = NCELL[IFACE][0] # I-index for adjacent cell
                JN = NCELL[IFACE][1] # J-index for adjacent cell
                RN[:] = CellCenter[IN, JN, :]
                DC = np.linalg.norm(RF[IFACE, :] - RC[:]) # distance between points C and F
                DN = np.linalg.norm(RF[IFACE, :] - RN[:]) # distance between points N and F

                VF = np.array([B_Functions.RLinearInterp(DC, DN, V[I, J, k], V[IN, JN, k]) for k in range(2)])
                rotV[I, J] = rotV[I, J] + np.cross(SF[IFACE], VF)

            rotV[I, J] = rotV[I, J] / VOL
