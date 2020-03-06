from collections import Counter
import numpy as np
import scipy.io as sio
from scipy.special import expit as sigmoid
    

if __name__=='__main__':
    ## general setup

    Nsample = 100
    D_L = 2
    D_M = 2

    random_state = 2020
    np.random.seed(random_state)

    ## generate L

    # L contains both binary, categorical, and continuous
    L = np.random.randn(Nsample, D_L)
    L[:,0] = np.random.randint(0, 2, size=Nsample) # binary
    L[:,1] = np.random.randint(0, 10, size=Nsample) # categorical

    ## generate A from L

    coef_A_L = np.array([1,2])
    noise_A_L = np.random.randn(Nsample)*0.1
    A = np.dot(L, coef_A_L) + noise_A_L
    A = sigmoid(A-A.mean())
    # A is binary
    A = (np.random.rand(*A.shape)<A).astype(int)
    print('A', Counter(A))

    ## generate M from A and L

    coef_M_AL = np.array([[1,2],
                          [3,4],
                          [5,6],])
    noise_M_AL = np.random.randn(Nsample, D_M)*1
    M = np.dot(np.c_[A,L], coef_M_AL) + noise_M_AL
    M = sigmoid(M - M.mean())
    # M is binary
    M = (np.random.rand(*M.shape)<M).astype(int)
    print('M', Counter(M.flatten()))

    ## generate Y from A, L, and M

    coef_Y_ALM = np.array([1,2,3,4,5])
    noise_Y_ALM = np.random.randn(Nsample)*0.1
    Y = np.dot(np.c_[A,L,M], coef_Y_ALM) + noise_Y_ALM

    sio.savemat('simulated_data.mat',
                {'A':A, 'L':L, 'Y':Y, 'M':M,
                 'coef_A_L':coef_A_L,
                 'coef_M_AL':coef_M_AL,
                 'coef_Y_ALM':coef_Y_ALM,})

