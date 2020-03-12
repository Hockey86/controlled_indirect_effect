from collections import Counter
import numpy as np
import scipy.io as sio
from scipy.special import expit as sigmoid
    

if __name__=='__main__':
    ## general setup

    N = 1000
    D_L = 2
    D_M = 3
    b = np.ones(N)
    
    random_state = 2020
    np.random.seed(random_state)

    ## generate L

    # L contains both binary, categorical, and continuous
    L = np.random.randn(N, D_L)
    L[:,0] = np.random.randint(0, 2, size=N) # binary
    L[:,1] = np.random.randint(0, 10, size=N) # categorical

    ## generate A from L

    coef_A_L = np.array([1,2,-9])*0.1
    noise_A_L = np.random.randn(N)*0.1
    A = np.dot(np.c_[L,b], coef_A_L) + noise_A_L
    A = sigmoid(A)
    # A is binary
    A = (A>0.5).astype(int)
    print('A', Counter(A))

    ## generate M1 from A and L
    coef_M1_AL = np.array([2,1,3,-15])*0.1
    noise_M1_AL = np.random.randn(N)*0.1
    M1 = sigmoid(np.dot(np.c_[A,L,b], coef_M1_AL) + noise_M1_AL)
    M1 = (M1>0.5).astype(int)

    ## generate M2 from A, L and M1

    coef_M2_ALM1 = np.array([2,4,1,5,-15])*0.1
    noise_M2_ALM1 = np.random.randn(N)*0.1
    M2 = sigmoid(np.dot(np.c_[A,L,M1,b], coef_M2_ALM1) + noise_M2_ALM1)
    M2 = (M2>0.5).astype(int)

    ## generate M3 from A, L, M1 and M2

    coef_M3_ALM1M2 = np.array([3,5,1,8,12,-30])*0.1
    noise_M3_ALM1M2 = np.random.randn(N)*0.1
    M3 = sigmoid(np.dot(np.c_[A,L,M1,M2,b], coef_M3_ALM1M2) + noise_M3_ALM1M2)
    M3 = (M3>0.5).astype(int)    
    
    M = np.c_[M1, M2, M3]
    print('M1', Counter(M[:,0].flatten()))
    print('M2', Counter(M[:,1].flatten()))
    print('M3', Counter(M[:,2].flatten()))
    
    ## generate Y from A, L, and M

    coef_Y_ALM = np.array([1,2,3,4,5,6])
    noise_Y_ALM = np.random.randn(N)*0.1
    Y = np.dot(np.c_[A,L,M], coef_Y_ALM) + noise_Y_ALM
    
    ## get ground truth
    
    aa0 = np.zeros(N)
    aa1 = np.zeros(N)+1
    mm0 = np.zeros(N)
    mm1 = np.zeros(N)+1
    
    M1 = sigmoid(np.dot(np.c_[aa0,L,b], coef_M1_AL))#>0.5).astype(int)
    M2 = sigmoid(np.dot(np.c_[aa0,L,M1,b], coef_M2_ALM1))#>0.5).astype(int)
    M3 = sigmoid(np.dot(np.c_[aa0,L,M1,M2,b], coef_M3_ALM1M2))#>0.5).astype(int)
    Y0 = np.dot(np.c_[aa0,L,M1,M2,M3], coef_Y_ALM).mean()
    
    M1 = sigmoid(np.dot(np.c_[aa1,L,b], coef_M1_AL))#>0.5).astype(int)
    M2 = sigmoid(np.dot(np.c_[aa1,L,M1,b], coef_M2_ALM1))#>0.5).astype(int)
    M3 = sigmoid(np.dot(np.c_[aa1,L,M1,M2,b], coef_M3_ALM1M2))#>0.5).astype(int)
    Y1 = np.dot(np.c_[aa1,L,M1,M2,M3], coef_Y_ALM).mean()
    
    TE = Y1-Y0
    
            
    CDE0 = []
    CIE1 = []
    CIE0 = []
    sCIE = []
    TEs = []
    """
    logit(M2_11).mean()*M1 + logit(M2_10).mean()*(1-M1)
    logit(M2).mean()

    from scipy.special import logit
    logit(M3_111).mean()*M1*M2_11.mean()+logit(M3_110).mean()*M1*(1-M2_11.mean())+logit(M3_101).mean()*(1-M1)*M2_10.mean()+logit(M3_100).mean()*(1-M1)*(1-M2_10.mean())
    logit(M3).mean()
    """
    for mi in range(M.shape[1]):
        if mi==0:
            M1_1 = sigmoid(np.dot(np.c_[aa1, L, b], coef_M1_AL)).mean()
            M1_0 = sigmoid(np.dot(np.c_[aa0, L, b], coef_M1_AL)).mean()
            M2_00 = sigmoid(np.dot(np.c_[aa0, L, mm0, b], coef_M2_ALM1))
            M2_10 = sigmoid(np.dot(np.c_[aa1, L, mm0, b], coef_M2_ALM1))
            M2_01 = sigmoid(np.dot(np.c_[aa0, L, mm1, b], coef_M2_ALM1))
            M2_11 = sigmoid(np.dot(np.c_[aa1, L, mm1, b], coef_M2_ALM1))
            M3_00m2 = sigmoid(np.dot(np.c_[aa0,L,mm0,M2_00,b], coef_M3_ALM1M2))
            M3_10m2 = sigmoid(np.dot(np.c_[aa1,L,mm0,M2_10,b], coef_M3_ALM1M2))
            M3_01m2 = sigmoid(np.dot(np.c_[aa0,L,mm1,M2_01,b], coef_M3_ALM1M2))
            M3_11m2 = sigmoid(np.dot(np.c_[aa1,L,mm1,M2_11,b], coef_M3_ALM1M2))
            
            Y00 = np.dot(np.c_[aa0,L,mm0,M2_00,M3_00m2], coef_Y_ALM).mean()
            Y10 = np.dot(np.c_[aa1,L,mm0,M2_10,M3_10m2], coef_Y_ALM).mean()
            Y01 = np.dot(np.c_[aa0,L,mm1,M2_01,M3_01m2], coef_Y_ALM).mean()
            Y11 = np.dot(np.c_[aa1,L,mm1,M2_11,M3_11m2], coef_Y_ALM).mean()
            M1 = M1_1
            M0 = M1_0
            
        elif mi==1:
            M1_1 = sigmoid(np.dot(np.c_[aa1, L, b], coef_M1_AL))
            M1_0 = sigmoid(np.dot(np.c_[aa0, L, b], coef_M1_AL))
            M2_0m1 = sigmoid(np.dot(np.c_[aa0, L, M1_0, b], coef_M2_ALM1)).mean()
            M2_1m1 = sigmoid(np.dot(np.c_[aa1, L, M1_1, b], coef_M2_ALM1)).mean()
            M3_00m1 = sigmoid(np.dot(np.c_[aa0,L,M1_0,mm0,b], coef_M3_ALM1M2))
            M3_10m1 = sigmoid(np.dot(np.c_[aa1,L,M1_1,mm0,b], coef_M3_ALM1M2))
            M3_01m1 = sigmoid(np.dot(np.c_[aa0,L,M1_0,mm1,b], coef_M3_ALM1M2))
            M3_11m1 = sigmoid(np.dot(np.c_[aa1,L,M1_1,mm1,b], coef_M3_ALM1M2))
            
            Y00 = np.dot(np.c_[aa0,L,M1_0,mm0,M3_00m1], coef_Y_ALM).mean()
            Y10 = np.dot(np.c_[aa1,L,M1_1,mm0,M3_10m1], coef_Y_ALM).mean()
            Y01 = np.dot(np.c_[aa0,L,M1_0,mm1,M3_01m1], coef_Y_ALM).mean()
            Y11 = np.dot(np.c_[aa1,L,M1_1,mm1,M3_11m1], coef_Y_ALM).mean()
            M1 = M2_1m1
            M0 = M2_0m1
            
        elif mi==2:
            M1_1 = sigmoid(np.dot(np.c_[aa1, L, b], coef_M1_AL))
            M1_0 = sigmoid(np.dot(np.c_[aa0, L, b], coef_M1_AL))
            M2_0m1 = sigmoid(np.dot(np.c_[aa0, L, M1_0, b], coef_M2_ALM1))
            M2_1m1 = sigmoid(np.dot(np.c_[aa1, L, M1_1, b], coef_M2_ALM1))
            M3_0m2m3 = sigmoid(np.dot(np.c_[aa0,L,M1_0,M2_0m1,b], coef_M3_ALM1M2)).mean()
            M3_1m2m3 = sigmoid(np.dot(np.c_[aa1,L,M1_1,M2_1m1,b], coef_M3_ALM1M2)).mean()
            
            Y00 = np.dot(np.c_[aa0,L,M1_0,M2_0m1,mm0], coef_Y_ALM).mean()
            Y10 = np.dot(np.c_[aa1,L,M1_1,M2_1m1,mm0], coef_Y_ALM).mean()
            Y01 = np.dot(np.c_[aa0,L,M1_0,M2_0m1,mm1], coef_Y_ALM).mean()
            Y11 = np.dot(np.c_[aa1,L,M1_1,M2_1m1,mm1], coef_Y_ALM).mean()
            M1 = M3_1m2m3
            M0 = M3_0m2m3
        
        CDE0.append(Y10-Y00)
        CIE1.append(Y11-Y10)
        CIE0.append(Y01-Y00)
        sCIE.append(CIE1[-1]*M1-CIE0[-1]*M0)
        TEs.append(CDE0[-1]+sCIE[-1])
    avg_TE = np.mean(TEs)

    sio.savemat('simulated_data.mat',
                {'A':A, 'L':L,
                 'Y':Y, 'M':M,
                
                 'coef_A_L':coef_A_L,
                 'coef_M1_AL':coef_M1_AL,
                 'coef_M2_ALM1':coef_M2_ALM1,
                 'coef_M3_ALM1M2':coef_M3_ALM1M2,
                 'coef_Y_ALM':coef_Y_ALM,
                 
                 'CDE0':CDE0, 'sCIE':sCIE,
                 'CIE0':CIE0, 'CIE1':CIE1,
                 'TEs':TEs, 'TE':TE
                 })

