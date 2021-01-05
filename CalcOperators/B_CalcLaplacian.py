import numpy as np
from Functions import B_Functions


def CalcLaplacian(NI, NJ, P, GradP, LapP, CellVolume, CellCenter, IFaceCenter, IFaceVector, JFaceCenter, JFaceVector):
    # initialization
    RF = np.zeros((4, 2), dtype=float)  # radius-vector for face-center
    SF = np.zeros((4, 2), dtype=float)  # face vector
    RC = np.zeros((2), dtype=float)  # radius-vector for cell-center
    RN = np.zeros((2), dtype=float)  # radius-vector for cell-center of adjacent cell
    NF = np.zeros((2), dtype=float)  # normal-vector for face
    RNC = np.zeros((2), dtype=float)  # unit radius-vector from point C to point N (ksi-vector)
    GF = np.zeros((2), dtype=float)  # pressure gradient

    for I in range(1, NI):
        for J in range(1, NJ):
            NCELL = np.array([[I - 1, J], [I + 1, J], [I, J - 1], [I, J + 1]], dtype=int)

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
                IN = NCELL[IFACE][0]  # I-index for adjacent cell
                JN = NCELL[IFACE][1]  # J-index for adjacent cell
                RN[:] = CellCenter[IN, JN, :]
                DC = np.linalg.norm(RF[IFACE, :] - RC[:]) # distance between points C and F
                DN = np.linalg.norm(RF[IFACE, :] - RN[:]) # distance between points N and F
                DNC = np.linalg.norm(CellCenter[IN, JN, :] - CellCenter[I, J, :])  # distance between points N and C
                NF[:] = SF[IFACE, :] / np.linalg.norm(SF[IFACE, :])

                # for skew correction
                RNC[:] = (CellCenter[IN, JN, :] - CellCenter[I, J,
                                                  :]) / DNC  # unit radius-vector from point C to point N (ksi-vector)
                GF[0] = B_Functions.RLinearInterp(DC, DN, GradP[I, J, 0],
                                                  GradP[
                                                      IN, JN, 0])  # linear interpolation of x-coordinate of pressure gradient at point F
                GF[1] = B_Functions.RLinearInterp(DC, DN, GradP[I, J, 1],
                                                  GradP[
                                                      IN, JN, 1])  # linear interpolation of y-coordinate of pressure gradient at point F

                dpdn = (P[IN, JN] - P[
                    I, J]) / DNC  # calculation of directional derivative with central difference method on face
                # boundaries
                if DN < 1e-5:
                    dpdn_c = np.dot(GradP[I, J, :], NF[:])  # calculation of directional derivative on cell-center
                    dpdn = 5/3*dpdn - 2/3*dpdn_c
                    GF[:] = GradP[I, J, :]

                # skew correction - hybrid method
                dpdn = dpdn + np.dot(NF[:] - RNC[:], GF[:])

                LapP[I, J] = LapP[I, J] + dpdn * np.linalg.norm(SF[IFACE, :])
            LapP[I, J] = LapP[I,J] / VOL
