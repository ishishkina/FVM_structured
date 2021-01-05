import numpy as np

def B_OutField(out, NI, NJ, X, Y, P, GradP, GradPError,V,divV,divVError, rotV, rotVError, LapP, LapPError):
    out.write('VARIABLES = "X", "Y", "P", "GradPx","GradPy","GradPErrorX","GradPErrorY", "Vx", "Vy", "divV", "divVError", "rotV", "rotVError", "LapP", "LapPError"\n')
    out.write(f'ZONE I={NI}, J={NJ}, DATAPACKING=BLOCK, VARLOCATION=([3-15]=CELLCENTERED)\n')

    np.savetxt(out, X.T.reshape(1, NI*NJ), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, Y.T.reshape(1, NI*NJ), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, P[1:NI, 1:NJ].T.reshape(1, (NI-1)*(NJ-1)), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, GradP[1:NI, 1:NJ, 0].T.reshape(1, (NI - 1) * (NJ - 1)), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, GradP[1:NI, 1:NJ, 1].T.reshape(1, (NI - 1) * (NJ - 1)), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, GradPError[1:NI, 1:NJ, 0].T.reshape(1, (NI - 1) * (NJ - 1)), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, GradPError[1:NI, 1:NJ, 1].T.reshape(1, (NI - 1) * (NJ - 1)), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, V[1:NI, 1:NJ, 0].T.reshape(1, (NI - 1) * (NJ - 1)), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, V[1:NI, 1:NJ, 1].T.reshape(1, (NI - 1) * (NJ - 1)), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, divV[1:NI, 1:NJ].T.reshape(1, (NI - 1) * (NJ - 1)), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, divVError[1:NI, 1:NJ].T.reshape(1, (NI - 1) * (NJ - 1)), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, rotV[1:NI, 1:NJ].T.reshape(1, (NI - 1) * (NJ - 1)), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, rotVError[1:NI, 1:NJ].T.reshape(1, (NI - 1) * (NJ - 1)), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, LapP[1:NI, 1:NJ].T.reshape(1, (NI - 1) * (NJ - 1)), fmt='%5.7f', delimiter='\n')
    np.savetxt(out, LapPError[1:NI, 1:NJ].T.reshape(1, (NI - 1) * (NJ - 1)), fmt='%5.7f', delimiter='\n')