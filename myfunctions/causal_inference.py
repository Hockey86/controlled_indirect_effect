import numpy as np


def infer_mediation(model_type, model_a_l, model_m_als, model_y_alm, Y, M, A, L, random_state=None):
    """
                           +-----v
    mediation analysis for A->M->Y
                         L-^--^--^
    """
    N = len(Y)
    if model_type=='or':
        
        CDEs = []
        sCIEs = []
        CIE0s = []
        CIE1s = []
        for mi in range(len(model_m_als)):
            Mas = []
            Yams = []
            for a in [0,1]:
                # compute M(a) for each mediator
                Ma = model_m_als[mi].predict_proba(np.c_[np.zeros((N,1))+a, L])[:,1]
                Mas.append(Ma.mean())
                
                Yams.append([])
                for m in [0,1]:
                    # compute Y(a,m) for each mediator
                    M2 = np.array(M)
                    M2[:,mi] = m
                    Y_aLm = model_y_alm.predict(np.c_[np.zeros((N,1))+a, L, M2])
                    Yams[-1].append(Y_aLm.mean())
            
            CIE0 = Yams[0][1] - Yams[0][0]
            CIE1 = Yams[1][1] - Yams[1][0]
            
            CDEs.append( (Yams[1][0]-Yams[0][0]) )
            sCIEs.append( Mas[1]*CIE1-Mas[0]*CIE0 )
            CIE0s.append( CIE0 )
            CIE1s.append( CIE1 )
            
    elif model_type=='ipw':
        raise NotImplemented(model_type)
    elif model_type=='msm':
        raise NotImplemented(model_type)
        
    elif model_type=='dr':
        p_A_L = model_a_l.predict_proba(L)
        p_A_L = p_A_L[range(N), A.astype(int)]
        
        CDEs = []
        sCIEs = []
        CIE0s = []
        CIE1s = []
        for mi in range(len(model_m_als)):
            Mas = []
            Yams = []
            for a in [0,1]:
                # compute M(a) for each mediator
                # M(a) = 1/N \sum_i E[M|a,Li] + 1(Ai=a)(Mi-E[M|a,Li])/P(Ai|Li)
                p_M_aL = model_m_als[mi].predict_proba(np.c_[np.zeros((N,1))+a, L])[:,1]
                Ma = np.nanmean( p_M_aL + (A==a).astype(float)*(M[:,mi]-p_M_aL)/p_A_L )
                Mas.append(Ma)
                
                Yams.append([])
                for m in [0,1]:
                    # compute Y(a,m) for each mediator
                    # Y(a,m) = 1/N \sum_i E[Y|a,Li,m] + 1(Ai=a,Mi=m)(Yi-E[Y|a,Li,m])/P(Ai,Mi|Li)
                    # P(A,M|Li) = P(M|A,Li)P(A|Li)
                    p_M_AL = model_m_als[mi].predict_proba(np.c_[A, L])
                    p_M_AL = p_M_AL[range(N), M[:,mi].astype(int)]
                    p_AM_L = p_M_AL*p_A_L
                    M2 = np.array(M)
                    M2[:,mi] = m
                    Y_aLm = model_y_alm.predict(np.c_[np.zeros((N,1))+a, L, M2])
                    Yam = np.nanmean( Y_aLm + ((A==a)&(M[:,mi]==m)).astype(float)*(Y-Y_aLm)/p_AM_L )
                    Yams[-1].append(Yam)
            
            CIE0 = Yams[0][1] - Yams[0][0]
            CIE1 = Yams[1][1] - Yams[1][0]
            
            CDEs.append( (Yams[1][0]-Yams[0][0]) )
            sCIEs.append( Mas[1]*CIE1-Mas[0]*CIE0 )
            CIE0s.append( CIE0 )
            CIE1s.append( CIE1 )

    return CDEs, sCIEs, CIE0s, CIE1s
    
