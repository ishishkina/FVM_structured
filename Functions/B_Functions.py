import numpy as np


def Pressure(X, Y):
    return X + Y + 1
    # X + Y + 1
    # X**2 + Y**2 + X + Y + 1.0
    # X + Y + 1  X**3 + Y**2 + X + Y + 1.0


def RLinearInterp(d1, d2, x1, x2):
    return (x1 * d2 + x2 * d1) / (d1 + d2)


def Velocity(X, Y):
    return np.array([Y + X, Y + 1.0])
    # np.array([Y + X, Y + 1.0]
    # np.array([Y**2 + X, Y**2 + X + 1.0])
    # np.array([Y**3 + X, Y**3 + X + 1.0])


def CalcGradPExact(X, Y):
    return np.array([1.0, 1.0])
    # np.array([1.0, 1.0])
    # np.array([2*X + 1.0, 2*Y + 1.0])
    # np.array([3*X**2 + 1.0, 2*Y + 1.0])


def DivVelocityExact(X, Y, mode):
    if mode == 0:
        return 2.0
        # 2.0
        # 2*Y + 1.0
        # 3*Y**2 + 1.0
    else:
        return 3*X + 4*Y + 3
        # 3*X + 4*Y + 3
        # X**2*(2*Y + 3) + X*(2*Y**2 + 4*Y + 3) + 4*Y**3 + 5*Y**2 + 5*Y + 2.0
        # 3*X**2*(Y**2 + 1.0) + X*(2*Y**3 + 3*Y**2 + 2*Y + 3.0) + 5*Y**4 + 5*Y**3 + 4*Y**2 + 3*Y + 2.0


def RotVelocityExact(X, Y):
    return -1.0
    # -1.0
    # 1.0 - 2*Y
    # 1.0 - 3*Y**2


def LapPressureExact(X, Y):
    return 1.0
    # 0.0
    # 4.0
    # 6*X + 2.0
