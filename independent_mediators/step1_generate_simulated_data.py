from collections import Counter
import numpy as np
import scipy.io as sio
from scipy.special import expit as sigmoid
    

if __name__=='__main__':
    ## general setup

    N = 1000
    D_L = 10
    D_M = 2
    b = np.ones(N)

    random_state = 2020
    np.random.seed(random_state)

    ## generate L

    # L contains both binary, categorical, and continuous
    L = np.random.randn(N, D_L)
    L[:,0] = np.random.randint(0, 2, size=N) # binary
    L[:,1] = np.random.randint(0, 10, size=N) # categorical

    ## generate A from L

    coef_A_L = np.array([1,2,1,2,3,2,1,4,2,3,-9]).astype(float)
    noise_A_L = np.random.randn(N)*1
    A = np.dot(np.c_[L,b], coef_A_L) + noise_A_L
    A = sigmoid(A)
    # A is binary
    A = (A>0.5).astype(float)
    print('A', Counter(A))

    ## generate M from A and L

    coef_M_AL = np.array([[1,2],
                          [3,4],
                          [5,6],
                          [4,3],
                          [2,1],
                          [3,4],
                          [2,5],
                          [1,3],
                          [2,2],
                          [3,1],
                          [2,3],
                          [-25,-25],]).astype(float)
    noise_M_AL = np.random.randn(N, D_M)*1
    M = np.dot(np.c_[A,L,b], coef_M_AL) + noise_M_AL
    M = sigmoid(M)
    # M is binary
    M = (M>0.5).astype(float)
    print('M', Counter(M.flatten()))
    
    ## generate Y from A, L, and M

    coef_Y_ALM = np.array([1,3,1,2,1,2,3,2,3,2,3,4,5]).astype(float)
    noise_Y_ALM = np.random.randn(N)*1
    Y = np.dot(np.c_[A,L,M], coef_Y_ALM) + noise_Y_ALM
    
    ## get ground truth
    
    aa=1;M1_1=sigmoid(np.dot(np.c_[np.zeros(N)+aa,L,b], coef_M_AL)[:,0]).mean()
    aa=0;M0_1=sigmoid(np.dot(np.c_[np.zeros(N)+aa,L,b], coef_M_AL)[:,0]).mean()
    aa=1;mm=1;Y11_1=np.dot(np.c_[np.zeros(N)+aa, L, np.zeros(N)+mm, sigmoid(np.dot(np.c_[np.zeros(N)+aa,L,b], coef_M_AL)[:,1])],coef_Y_ALM).mean()
    aa=1;mm=0;Y10_1=np.dot(np.c_[np.zeros(N)+aa, L, np.zeros(N)+mm, sigmoid(np.dot(np.c_[np.zeros(N)+aa,L,b], coef_M_AL)[:,1])],coef_Y_ALM).mean()
    aa=0;mm=1;Y01_1=np.dot(np.c_[np.zeros(N)+aa, L, np.zeros(N)+mm, sigmoid(np.dot(np.c_[np.zeros(N)+aa,L,b], coef_M_AL)[:,1])],coef_Y_ALM).mean()
    aa=0;mm=0;Y00_1=np.dot(np.c_[np.zeros(N)+aa, L, np.zeros(N)+mm, sigmoid(np.dot(np.c_[np.zeros(N)+aa,L,b], coef_M_AL)[:,1])],coef_Y_ALM).mean()
    
    aa=1;M1_2=sigmoid(np.dot(np.c_[np.zeros(N)+aa,L,b], coef_M_AL)[:,1]).mean()
    aa=0;M0_2=sigmoid(np.dot(np.c_[np.zeros(N)+aa,L,b], coef_M_AL)[:,1]).mean()
    aa=1;mm=1;Y11_2=np.dot(np.c_[np.zeros(N)+aa, L, sigmoid(np.dot(np.c_[np.zeros(N)+aa,L,b], coef_M_AL)[:,0]), np.zeros(N)+mm],coef_Y_ALM).mean()
    aa=1;mm=0;Y10_2=np.dot(np.c_[np.zeros(N)+aa, L, sigmoid(np.dot(np.c_[np.zeros(N)+aa,L,b], coef_M_AL)[:,0]), np.zeros(N)+mm],coef_Y_ALM).mean()
    aa=0;mm=1;Y01_2=np.dot(np.c_[np.zeros(N)+aa, L, sigmoid(np.dot(np.c_[np.zeros(N)+aa,L,b], coef_M_AL)[:,0]), np.zeros(N)+mm],coef_Y_ALM).mean()
    aa=0;mm=0;Y00_2=np.dot(np.c_[np.zeros(N)+aa, L, sigmoid(np.dot(np.c_[np.zeros(N)+aa,L,b], coef_M_AL)[:,0]), np.zeros(N)+mm],coef_Y_ALM).mean()
    
    CDE0_1 = Y10_1-Y00_1
    CIE1_1 = Y11_1-Y10_1
    CIE0_1 = Y01_1-Y00_1
    sCIE1 = CIE1_1*M1_1-CIE0_1*M0_1
    TE1 = CDE0_1+sCIE1
    
    CDE0_2 = Y10_2-Y00_2
    CIE1_2 = Y11_2-Y10_2
    CIE0_2 = Y01_2-Y00_2
    sCIE2 = CIE1_2*M1_2-CIE0_2*M0_2
    TE2 = CDE0_2+sCIE2
    
    TE = (TE1+TE2)/2
    
    m0=sigmoid(np.dot(np.c_[np.zeros(N),L,b],coef_M_AL))
    m1=sigmoid(np.dot(np.c_[np.zeros(N)+1,L,b],coef_M_AL))
    Y0=np.dot(np.c_[np.zeros(N),L,m0],coef_Y_ALM).mean()
    Y1=np.dot(np.c_[np.zeros(N)+1,L,m1],coef_Y_ALM).mean()
    assert np.abs(Y1-Y0-TE)<1e-4
    import pdb;pdb.set_trace()
    
    sio.savemat('simulated_data.mat',
                {'A':A, 'L':L, 'Y':Y, 'M':M,
                
                 'coef_A_L':coef_A_L,
                 'coef_M_AL':coef_M_AL,
                 'coef_Y_ALM':coef_Y_ALM,
                 
                 'CDE0_1':CDE0_1,
                 'CIE1_1':CIE1_1,
                 'CIE0_1':CIE0_1,
                 'sCIE1':sCIE1,
                 'TE1':TE1,
                 'CDE0_2':CDE0_2,
                 'CIE1_2':CIE1_2,
                 'CIE0_2':CIE0_2,
                 'sCIE2':sCIE2,
                 'TE2':TE2,
                 'TE':TE
                 })

