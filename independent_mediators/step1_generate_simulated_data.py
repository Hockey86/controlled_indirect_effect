from collections import Counter
import numpy as np
import scipy.io as sio
from scipy.special import expit as sigmoid
    

if __name__=='__main__':
    ## general setup

    Nsample = 1000
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

    coef_A_L = np.array([1,2,-9])
    noise_A_L = np.random.randn(Nsample)*0.1
    A = np.dot(np.c_[L,np.ones(len(L))], coef_A_L) + noise_A_L
    A = sigmoid(A)
    # A is binary
    A = (np.random.rand(*A.shape)<A).astype(int)
    print('A', Counter(A))

    ## generate M from A and L

    coef_M_AL = np.array([[1,2],
                          [3,4],
                          [5,6],
                          [-25,-25],])
    noise_M_AL = np.random.randn(Nsample, D_M)*1
    M = np.dot(np.c_[A,L,np.ones(len(A))], coef_M_AL) + noise_M_AL
    M = sigmoid(M)
    # M is binary
    M = (M>0.5).astype(int)
    print('M', Counter(M.flatten()))
    
    ## generate Y from A, L, and M

    coef_Y_ALM = np.array([1,2,3,4,5])
    noise_Y_ALM = np.random.randn(Nsample)*0.1
    Y = np.dot(np.c_[A,L,M], coef_Y_ALM) + noise_Y_ALM
    
    ## get ground truth
    
    aa=1;mm=1;Y11_1=np.dot(np.c_[np.zeros(len(A))+aa, L, np.zeros(len(A))+mm, sigmoid(np.dot(np.c_[np.zeros(len(A))+aa,L,np.ones(len(A))], coef_M_AL)[:,1])],coef_Y_ALM).mean()
    aa=1;mm=0;Y10_1=np.dot(np.c_[np.zeros(len(A))+aa, L, np.zeros(len(A))+mm, sigmoid(np.dot(np.c_[np.zeros(len(A))+aa,L,np.ones(len(A))], coef_M_AL)[:,1])],coef_Y_ALM).mean()
    aa=0;mm=1;Y01_1=np.dot(np.c_[np.zeros(len(A))+aa, L, np.zeros(len(A))+mm, sigmoid(np.dot(np.c_[np.zeros(len(A))+aa,L,np.ones(len(A))], coef_M_AL)[:,1])],coef_Y_ALM).mean()
    aa=0;mm=0;Y00_1=np.dot(np.c_[np.zeros(len(A))+aa, L, np.zeros(len(A))+mm, sigmoid(np.dot(np.c_[np.zeros(len(A))+aa,L,np.ones(len(A))], coef_M_AL)[:,1])],coef_Y_ALM).mean()
    aa=1;M1_1=sigmoid(np.dot(np.c_[np.zeros(len(A))+aa,L,np.ones(len(A))], coef_M_AL)[:,0]).mean()
    aa=0;M0_1=sigmoid(np.dot(np.c_[np.zeros(len(A))+aa,L,np.ones(len(A))], coef_M_AL)[:,0]).mean()
    
    aa=1;mm=1;Y11_2=np.dot(np.c_[np.zeros(len(A))+aa, L, sigmoid(np.dot(np.c_[np.zeros(len(A))+aa,L,np.ones(len(A))], coef_M_AL)[:,0]), np.zeros(len(A))+mm],coef_Y_ALM).mean()
    aa=1;mm=0;Y10_2=np.dot(np.c_[np.zeros(len(A))+aa, L, sigmoid(np.dot(np.c_[np.zeros(len(A))+aa,L,np.ones(len(A))], coef_M_AL)[:,0]), np.zeros(len(A))+mm],coef_Y_ALM).mean()
    aa=0;mm=1;Y01_2=np.dot(np.c_[np.zeros(len(A))+aa, L, sigmoid(np.dot(np.c_[np.zeros(len(A))+aa,L,np.ones(len(A))], coef_M_AL)[:,0]), np.zeros(len(A))+mm],coef_Y_ALM).mean()
    aa=0;mm=0;Y00_2=np.dot(np.c_[np.zeros(len(A))+aa, L, sigmoid(np.dot(np.c_[np.zeros(len(A))+aa,L,np.ones(len(A))], coef_M_AL)[:,0]), np.zeros(len(A))+mm],coef_Y_ALM).mean()
    aa=1;M1_2=sigmoid(np.dot(np.c_[np.zeros(len(A))+aa,L,np.ones(len(A))], coef_M_AL)[:,1]).mean()
    aa=0;M0_2=sigmoid(np.dot(np.c_[np.zeros(len(A))+aa,L,np.ones(len(A))], coef_M_AL)[:,1]).mean()
    
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

