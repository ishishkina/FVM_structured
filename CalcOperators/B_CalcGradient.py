import numpy as np
from Functions import B_Functions


def CalcGradient(NI: int, NJ: int, P: float, GradP: float, CellVolume: float, CellCenter: float, IFaceCenter: float, IFaceVector: float, JFaceCenter: float, JFaceVector: float) -> float:

    # initialization
    RF = np.zeros((4, 2), dtype=float)  # radius-vector for face-center
    SF = np.zeros((4, 2), dtype=float)  # face vector
    RC = np.zeros((2), dtype=float)  # radius-vector for cell-center
    RN = np.zeros((2), dtype=float)  # radius-vector for cell-center of adjacent cell
    RM = np.zeros((2), dtype=float)  # radius-vector for point M
    GPM = np.zeros((2), dtype=float)  # pressure gradient at point M

    for I in range(1, NI):
        for J in range(1, NJ):
            NCELL = np.array([[I - 1, J], [I + 1, J],  [I, J - 1], [I, J + 1]], dtype=int)

            # calculation of radius-vector for face-center
            RF[0, :] = IFaceCenter[I - 1, J - 1, :]
            RF[1, :] = IFaceCenter[I, J - 1, :]
            RF[2, :] = JFaceCenter[I - 1, J - 1, :]
            RF[3, :] = JFaceCenter[I - 1, J, :]

            # calculation face vector
            SF[0, :] = -IFaceVector[I - 1, J - 1, :]
            SF[1, :] = IFaceVector[I, J - 1, :]
            SF[2, :] = -JFaceVector[I - 1, J - 1, :]
            SF[3, :] = JFaceVector[I - 1, J, :]

            # calculation of cell volume
            VOL = CellVolume[I - 1, J - 1]
            # calculation of radius-vector for cell center
            RC[:] = CellCenter[I, J, :]

            # calculation of pressure gradient by Green-Gauss method with iterations
            GP = np.zeros((2), dtype=float)  # pressure gradient
            # loop for faces
            for IFACE in range(4):
                IN = NCELL[IFACE][0] # I-index for adjacent cell
                JN = NCELL[IFACE][1] # J-index for adjacent cell
                RN[:] = CellCenter[IN, JN, :]
                DC = np.linalg.norm(RF[IFACE, :] - RC[:]) # distance between points C and F
                DN = np.linalg.norm(RF[IFACE, :] - RN[:]) # distance between points N and F

                PM = B_Functions.RLinearInterp(DC, DN, P[I, J], P[IN, JN])
                RM[0] = B_Functions.RLinearInterp(DC, DN, CellCenter[I, J, 0],
                                                  CellCenter[IN, JN, 0]) # linear interpolation of x-coordinate for point M
                RM[1] = B_Functions.RLinearInterp(DC, DN, CellCenter[I, J, 1],
                                                  CellCenter[IN, JN, 1])  # linear interpolation of y-coordinate for point M
                GPM[0] = B_Functions.RLinearInterp(DC, DN, GradP[I, J, 0],
                                                   GradP[IN, JN, 0])  # linear interpolation of x-coordinate of pressure gradient at point M
                GPM[1] = B_Functions.RLinearInterp(DC, DN, GradP[I, J, 1],
                                                   GradP[IN, JN, 1])  # linear interpolation of y-coordinate of pressure gradient at point M

                PF = PM + np.dot((RF[IFACE, :] - RM[:]), GPM[:])
                GP[:] = GP[:] + PF*SF[IFACE, :]

            GradP[I, J, :] = GP[:] / VOL
