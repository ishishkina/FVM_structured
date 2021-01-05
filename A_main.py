#!/usr/bin/env python

import os
import numpy as np
from CalcOperators import B_CalcDivergence, B_CalcGradient, B_CalcLaplacian, B_CalcRotor
from GridTools import B_CalcMetric, B_OutputFields
from Functions import B_Functions
import InputTools

initializer = InputTools.Initializer('input.txt')
#InputFile = 'input.txt'  # name of input file

if not os.path.exists('Results'):
    os.mkdir('Results')
counter = 0
filelist = os.listdir(path='Results')
for name in filelist:
    if name.startswith('data') and name.endswith('.plt'):
        counter += 1
OutputFile = os.path.join('Results', f'data{counter}.plt')  # name of output file

# Read the name of computational mesh from input file
print('Read input file:', initializer.mesh)
MeshFile = initializer.mesh

# Read the name of file with flow fields from input file
SolutionFile = initializer.data

# Read number of iteration for Green-Gauss method from file with mesh
Niter = initializer.iterations
print('Number of iterations:', Niter)

# Read the name of the method for calculation of convective term from input file
mode = initializer.mod
if mode == 0:
    method = 'we do not have convective terms'
elif mode == 1:
    method = 'central'
elif mode == 2:
    method = 'first order upwind'
else:
    method = 'second order upwind'
print('Method for calculation of convective terms :', method)

# Read nodes number (NI,NJ) from file with mesh
print('Read nodes number from file:', MeshFile)
mesh = open(MeshFile, 'r')
values = mesh.readline()
parts = values.split(' ')
NI = int(parts[0])
NJ = int(parts[1])
print('NI=', NI, ', NJ = ', NJ)

# Initialization of all arrays
X = np.zeros((NI, NJ), dtype=float)  # mesh nodes X-coordinates
Y = np.zeros((NI, NJ), dtype=float)  # mesh nodes Y-coordinates

P = np.zeros((NI + 1, NJ + 1), dtype=float)  # pressure
GradP = np.zeros((NI + 1, NJ + 1, 2), dtype=float)  # pressure gradient
GradPExact = np.zeros((NI + 1, NJ + 1, 2), dtype=float)  # exact value of pressure gradient
GradPError = np.zeros((NI + 1, NJ + 1, 2), dtype=float)  # error in calculating of pressure gradient
LapP = np.zeros((NI + 1, NJ + 1), dtype=float)  # pressure laplacian
LapPExact = np.zeros((NI + 1, NJ + 1), dtype=float)  # exact value of pressure laplacian
LapPError = np.zeros((NI + 1, NJ + 1), dtype=float)  # error in calculating of pressure laplacian


V = np.zeros((NI + 1, NJ + 1, 2), dtype=float)  # velocity
divV = np.zeros((NI + 1, NJ + 1), dtype=float)  # velocity divergence
divVExact = np.zeros((NI + 1, NJ + 1), dtype=float)  # exact value of velocity divergence
divVError = np.zeros((NI + 1, NJ + 1), dtype=float)  # error in calculating of velocity divergence
rotV = np.zeros((NI + 1, NJ + 1), dtype=float)  # velocity rotor
rotVExact = np.zeros((NI + 1, NJ + 1), dtype=float)  # exact value of velocity rotor
rotVError = np.zeros((NI + 1, NJ + 1), dtype=float)  # error in calculating of velocity rotor

CellVolume = np.zeros((NI - 1, NJ - 1), dtype=float)  # Cell Volumes
CellCenter = np.zeros((NI + 1, NJ + 1, 2), dtype=float)  # Cell Centers
IFaceCenter = np.zeros((NI, NJ - 1, 2), dtype=float)  # Face Centers for I-faces
IFaceVector = np.zeros((NI, NJ - 1, 2), dtype=float)  # Face Vectors for I-faces
JFaceCenter = np.zeros((NI - 1, NJ, 2), dtype=float)  # Face Centers for J-faces
JFaceVector = np.zeros((NI - 1, NJ, 2), dtype=float)  # Face Vectors for J-faces

# Read grid from file with mesh
for J in range(NJ):
    for I in range(NI):
        values = mesh.readline()
        parts = values.split()
        X[I, J] = float(parts[0])
        Y[I, J] = float(parts[1])
        #print(X[I, J], Y[I, J])
mesh.close()

# Calculate metric
print('Calculate metric')
B_CalcMetric.CalcMetric(NI, NJ, X, Y, CellCenter, CellVolume, IFaceCenter, IFaceVector, JFaceCenter, JFaceVector)

if initializer.data:
    # Read solution
    print('Read solution data from file')
    with open(SolutionFile) as result:
        result.readline()
        result.readline()
        for J in range(NJ+1):
            for I in range(NI+1):
                values = result.readline().strip().split()
                _, _, V[I, J, 0], V[I, J, 1], _, P[I, J], *_ = map(float, values)
else:
    # Initiate fields
    print('Initiate fields')
    for J in range(NJ + 1):
        for I in range(NI + 1):
            P[I, J] = B_Functions.Pressure(CellCenter[I, J, 0], CellCenter[I, J, 1])
            GradPExact[I, J, :] = B_Functions.CalcGradPExact(CellCenter[I, J, 0], CellCenter[I, J, 1])
            V[I, J, :] = B_Functions.Velocity(CellCenter[I, J, 0], CellCenter[I, J, 1])
            divVExact[I, J] = B_Functions.DivVelocityExact(CellCenter[I, J, 0], CellCenter[I, J, 1], mode)
            rotVExact[I, J] = B_Functions.RotVelocityExact(CellCenter[I, J, 0], CellCenter[I, J, 1])
            LapPExact[I, J] = B_Functions.LapPressureExact(CellCenter[I, J, 0], CellCenter[I, J, 1])

# Calculate gradient
print('Calculate gradient')
for I in range(Niter):
    B_CalcGradient.CalcGradient(NI, NJ, P, GradP, CellVolume, CellCenter, IFaceCenter, IFaceVector, JFaceCenter, JFaceVector)
    if not initializer.data:
       GradPError[1:NI, 1:NJ, :] = np.absolute((GradPExact[1:NI, 1:NJ, :] - GradP[1:NI, 1:NJ, :]) / GradPExact[1:NI, 1:NJ, :])
       print('Maximum GradP-Error', np.amax(GradPError[:, :, :]))

if not initializer.data:
    print('Maximum GradPx-Error', np.amax(GradPError[:, :, :]))
    print('Maximum GradPy-Error', np.amax(GradPError[:, :, :]))

# Calculate divergence
print('Calculate divergence')
B_CalcDivergence.CalcDivergence(mode, NI, NJ, V, divV, P, GradP, CellVolume, CellCenter, IFaceCenter, IFaceVector, JFaceCenter, JFaceVector)
if not initializer.data:
    divVError[1:NI, 1:NJ] = np.absolute((divVExact[1:NI, 1:NJ] - divV[1:NI, 1:NJ]) / divVExact[1:NI, 1:NJ])
    print('Maximum divV-Error', np.amax(divVError[:, :]))

# Calculate rotor
print('Calculate rotor')
B_CalcRotor.CalcRotor(NI, NJ, V, rotV, CellVolume, CellCenter, IFaceCenter, IFaceVector, JFaceCenter, JFaceVector)
if not initializer.data:
    rotVError[1:NI, 1:NJ] = np.absolute((rotVExact[1:NI, 1:NJ] - rotV[1:NI, 1:NJ]) / abs(rotVExact[1:NI, 1:NJ]))
    print('Maximum rotV-Error', np.amax(rotVError[:, :]))

# Calculate laplacian
print('Calculate laplacian')
B_CalcLaplacian.CalcLaplacian(NI, NJ, P, GradP, LapP, CellVolume, CellCenter, IFaceCenter, IFaceVector, JFaceCenter, JFaceVector)
if not initializer.data:
    LapPError[1:NI, 1:NJ] = np.absolute((LapPExact[1:NI, 1:NJ] - LapP[1:NI, 1:NJ]) / LapPExact[1:NI, 1:NJ])
    print('Maximum LapP-Error', np.amax(LapPError[:, :]))

# Output fields
print('Output fields to file:', OutputFile)
out = open(OutputFile, 'w')
B_OutputFields.B_OutField(out, NI, NJ, X, Y, P, GradP, GradPError, V, divV, divVError, rotV, rotVError, LapP, LapPError)
print('Work done!')
out.close()
