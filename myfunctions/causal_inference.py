import numpy as np


def infer_mediation(method, model_a_l, model_m_als, model_y_alm, Y, M, A, L, random_state=None):
    """
                           +-----v
    mediation analysis for A->M->Y
                         L-^--^--^
    """
    N = len(Y)
    CDEs = []
    sCIEs = []
    CIE0s = []
    CIE1s = []
        
    if method=='or':
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
        
    elif method=='dr':
        p_A_L = model_a_l.predict_proba(L)
        p_A_L = p_A_L[range(N), A.astype(int)]
        
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
            
    else:
        raise NotImplemented(method)

    return CDEs, sCIEs, CIE0s, CIE1s


def infer_mediation_3mediator(method, model_a_l, model_m_als, model_y_alm, Y, M, A, L, random_state=None):
    """
    """
    N = len(Y)
    zeros = np.zeros(N)
    
    CDEs = []
    sCIEs = []
    CIE0s = []
    CIE1s = []
    
    M1s = []
    M2s = []
    M3s = []
    
    if method=='or':
        for a in [0,1]:
            # M1(a)
            M1 = model_m_als[0].predict_proba(np.c_[zeros+a, L])[:,1].mean()
            # M2(a) = M2(a,M1(a))
            M2 = model_m_als[1].predict_proba(np.c_[zeros+a, L, zeros+1])[:,1].mean()*M1 +\
                 model_m_als[1].predict_proba(np.c_[zeros+a, L, zeros+0])[:,1].mean()*(1-M1)
            # M3(a) = M3(a,M1(a),M2(a,M1(a)))
            M3 = model_m_als[2].predict_proba(np.c_[zeros+a, L, zeros+1, zeros+1])[:,1].mean()*M1*M2 +\
                 model_m_als[2].predict_proba(np.c_[zeros+a, L, zeros+1, zeros+0])[:,1].mean()*M1*(1-M2) +\
                 model_m_als[2].predict_proba(np.c_[zeros+a, L, zeros+0, zeros+1])[:,1].mean()*(1-M1)*M2 +\
                 model_m_als[2].predict_proba(np.c_[zeros+a, L, zeros+0, zeros+0])[:,1].mean()*(1-M1)*(1-M2)
            M1s.append(M1)
            M2s.append(M2)
            M3s.append(M3)
            
        for mi in range(len(model_m_als)):
            Yams = []
            if mi==0:
                Mas = M1s
                # compute M(a) for each mediator
                for a in [0,1]:
                    # compute Y(a,m) for each mediator
                    Yams.append([])
                    for m in [0,1]:
                        # compute Y(a,m) for each mediator
                        # Y(a,m) = Y(a, m, M2(a,m), M3(a,m,M2(a,m)))
                        M2_am = model_m_als[1].predict_proba(np.c_[zeros+a,L,zeros+m])[:,1].mean()
                        M3_am_m2 = model_m_als[2].predict_proba(np.c_[zeros+a,L,zeros+m,zeros+1])[:,1].mean()*M2_am +\
                                   model_m_als[2].predict_proba(np.c_[zeros+a,L,zeros+m,zeros+0])[:,1].mean()*(1-M2_am)
                        Yam = model_y_alm.predict(np.c_[zeros+a, L, zeros+m, zeros+1, zeros+1]).mean()*M2_am*M3_am_m2 +\
                                model_y_alm.predict(np.c_[zeros+a, L, zeros+m, zeros+1, zeros+0]).mean()*M2_am*(1-M3_am_m2) +\
                                model_y_alm.predict(np.c_[zeros+a, L, zeros+m, zeros+0, zeros+1]).mean()*(1-M2_am)*M3_am_m2 +\
                                model_y_alm.predict(np.c_[zeros+a, L, zeros+m, zeros+0, zeros+0]).mean()*(1-M2_am)*(1-M3_am_m2)
                        Yams[-1].append(Yam)
                        
            elif mi==1:
                Mas = M2s
                # compute M(a) for each mediator
                for a in [0,1]:
                    # compute Y(a,m) for each mediator
                    Yams.append([])
                    for m in [0,1]:
                        # compute Y(a,m) for each mediator
                        # Y(a,m) = Y(a, M1(a), m, M3(a,M1(a),m))
                        M3_am_m1 = model_m_als[2].predict_proba(np.c_[zeros+a,L,zeros+1,zeros+m])[:,1].mean()*M1s[a] +\
                                   model_m_als[2].predict_proba(np.c_[zeros+a,L,zeros+0,zeros+m])[:,1].mean()*(1-M1s[a])
                        Yam = model_y_alm.predict(np.c_[zeros+a, L, zeros+1, zeros+m, zeros+1]).mean()*M1s[a]*M3_am_m1 +\
                              model_y_alm.predict(np.c_[zeros+a, L, zeros+1, zeros+m, zeros+0]).mean()*M1s[a]*(1-M3_am_m1) +\
                              model_y_alm.predict(np.c_[zeros+a, L, zeros+0, zeros+m, zeros+1]).mean()*(1-M1s[a])*M3_am_m1 +\
                              model_y_alm.predict(np.c_[zeros+a, L, zeros+0, zeros+m, zeros+0]).mean()*(1-M1s[a])*(1-M3_am_m1)
                        Yams[-1].append(Yam)
                        
            elif mi==2:
                Mas = M3s
                # compute M(a) for each mediator
                for a in [0,1]:
                    # compute Y(a,m) for each mediator
                    Yams.append([])
                    for m in [0,1]:
                        # compute Y(a,m) for each mediator
                        # Y(a,m) = Y(a, M1(a), M2(a,M1(a)), m)
                        Yam = model_y_alm.predict(np.c_[zeros+a, L, zeros+1, zeros+1, zeros+m]).mean()*M1s[a]*M2s[a] +\
                              model_y_alm.predict(np.c_[zeros+a, L, zeros+1, zeros+0, zeros+m]).mean()*M1s[a]*(1-M2s[a]) +\
                              model_y_alm.predict(np.c_[zeros+a, L, zeros+0, zeros+1, zeros+m]).mean()*(1-M1s[a])*M2s[a] +\
                              model_y_alm.predict(np.c_[zeros+a, L, zeros+0, zeros+0, zeros+m]).mean()*(1-M1s[a])*(1-M2s[a])
                        Yams[-1].append(Yam)
                
            CIE0 = Yams[0][1] - Yams[0][0]
            CIE1 = Yams[1][1] - Yams[1][0]
            
            CDEs.append( (Yams[1][0]-Yams[0][0]) )
            sCIEs.append( Mas[1]*CIE1-Mas[0]*CIE0 )
            CIE0s.append( CIE0 )
            CIE1s.append( CIE1 )
            
    elif method=='dr':
        p_A_L = model_a_l.predict_proba(L)
        p_A_L = p_A_L[range(N), A.astype(int)]
        
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
                    Y_aLm = model_y_alm[mi].predict(np.c_[np.zeros((N,1))+a, L, M[:,mi]])
                    Yam = np.nanmean( Y_aLm + ((A==a)&(M[:,mi]==m)).astype(float)*(Y-Y_aLm)/p_AM_L )
                    Yams[-1].append(Yam)
            
            CIE0 = Yams[0][1] - Yams[0][0]
            CIE1 = Yams[1][1] - Yams[1][0]
            
            CDEs.append( (Yams[1][0]-Yams[0][0]) )
            sCIEs.append( Mas[1]*CIE1-Mas[0]*CIE0 )
            CIE0s.append( CIE0 )
            CIE1s.append( CIE1 )
        
    else:
        raise NotImplemented(method)

    return CDEs, sCIEs, CIE0s, CIE1s


def select_estimator(info):
    return best_pm_outcome, best_pm_exposure
    
