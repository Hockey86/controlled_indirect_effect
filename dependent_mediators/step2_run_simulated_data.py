from itertools import product
import os
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
import sys
sys.path.insert(0, '../myfunctions')
from prediction import fit_prediction_model
#from causal_inference import infer_mediation

def infer_mediation(model_type, model_a_l, model_m_als, model_y_alm, Y, M, A, L, random_state=None):
    """
    """
    N = len(Y)
    zeros = np.zeros(N)
    assert model_type=='or', 'model_type %s'%model_type
    
    p_A_L = model_a_l.predict_proba(L)
    p_A_L = p_A_L[range(N), A.astype(int)]
    
    CDEs = []
    sCIEs = []
    CIE0s = []
    CIE1s = []
    
    M1s = []
    M2s = []
    M3s = []
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

    return CDEs, sCIEs, CIE0s, CIE1s


if __name__=='__main__':
    ## load data

    matdata = sio.loadmat('simulated_data.mat')
    A = matdata['A'].flatten()
    L = matdata['L']
    Y = matdata['Y'].flatten()
    M = matdata['M']
    sids = np.arange(len(A))
    Mnames = ['M%d'%(i+1,) for i in range(M.shape[1])]
    
    print('CDE0', matdata['CDE0'][0])
    print('sCIE', matdata['sCIE'][0])
    print('CIE0', matdata['CIE0'][0])
    print('CIE1', matdata['CIE1'][0])
    print('TEs', matdata['TEs'][0])
    print('TE', matdata['TE'][0])
    
    ## set up numbers

    random_state = 2020
    prediction_methods = ['linear']
    causal_inference_methods = ['or']
    Nbt = 10#00
    np.random.seed(random_state)
    
    ## generate cv split

    cv_path = 'patients_cv_split_simulated.pickle'
    if os.path.exists(cv_path):
        with open(cv_path, 'rb') as ff:
            tr_sids, te_sids = pickle.load(ff)
    else:
        cvf = 10
        sids2 = np.array(sids)
        np.random.shuffle(sids2)
        tr_sids = []
        te_sids = []
        cv_split = np.array_split(sids2, cvf)
        for cvi in cv_split:
            te_sids.append(np.sort(cvi))
            tr_sids.append(np.sort(np.setdiff1d(sids2, cvi)))
        with open(cv_path, 'wb') as ff:
            pickle.dump([tr_sids, te_sids], ff)

    ## bootstrapping

    res = []
    
    for bti in tqdm(range(Nbt+1)):
        np.random.seed(random_state+bti)
        if bti==0:
            # use the actual data
            Abt = A
            Ybt = Y
            Lbt = L
            Mbt = M
            sidsbt = sids
        else:
            # use bootstrapped data
            btids = np.random.choice(len(Y), len(Y), replace=True)
            Abt = A[btids]
            Ybt = Y[btids]
            Lbt = L[btids]
            Mbt = M[btids]
            sidsbt = sids[btids]
        
        # outer loop cross validation
        for cvi in range(len(tr_sids)):
            trid = np.in1d(sidsbt, tr_sids[cvi])
            teid = np.in1d(sidsbt, te_sids[cvi])
            
            Atr = Abt[trid]
            Ytr = Ybt[trid]
            Ltr = Lbt[trid]
            Mtr = Mbt[trid]
            Ate = Abt[teid]
            Yte = Ybt[teid]
            Lte = Lbt[teid]
            Mte = Mbt[teid] 
            
            Lmean = Ltr.mean(axis=0)
            Lstd = Ltr.std(axis=0)
            #Ltr = (Ltr-Lmean)/Lstd
            #Lte = (Lte-Lmean)/Lstd
            
            #try:
            for pi, pm in enumerate(prediction_methods):          
                # fit A|L
                model_a_l, model_a_l_perf = fit_prediction_model(pm+':bclf', Ltr, Atr,
                                        save_path='models_simulated_data/model_a_l_cv%d_%s'%(cvi+1, pm) if bti==0 else None,
                                        random_state=random_state+pi+1000)
            
                # fit Y|A,L,M
                model_y_alm, model_y_alm_perf = fit_prediction_model(pm+':reg', np.c_[Atr, Ltr, Mtr], Ytr,
                                    save_path='models_simulated_data/model_y_alm_cv%d_%s'%(cvi+1, pm) if bti==0 else None,
                                    random_state=random_state+pi*3000)
                                        
                model_m_als = []
                model_m_al_perfs = []
                for mi, mediator_name in enumerate(Mnames):
                    if mi==0:
                        # fit M1|A,L
                        model_m_al, model_m_al_perf = fit_prediction_model(pm+':bclf', np.c_[Atr, Ltr], Mtr[:, mi],
                                    save_path='models_simulated_data/med_model_%s_cv%d_%s'%(mediator_name, cvi+1, pm) if bti==0 else None,
                                    random_state=random_state+pi*2000+mi)
                    elif mi==1:
                        # fit M2|A,L,M1
                        model_m_al, model_m_al_perf = fit_prediction_model(pm+':bclf', np.c_[Atr, Ltr, Mtr[:,0]], Mtr[:, mi],
                                    save_path='models_simulated_data/med_model_%s_cv%d_%s'%(mediator_name, cvi+1, pm) if bti==0 else None,
                                    random_state=random_state+pi*2000+mi)
                    elif mi==2:
                        # fit M3|A,L,M1,M2
                        model_m_al, model_m_al_perf = fit_prediction_model(pm+':bclf', np.c_[Atr, Ltr, Mtr[:,[0,1]]], Mtr[:, mi],
                                    save_path='models_simulated_data/med_model_%s_cv%d_%s'%(mediator_name, cvi+1, pm) if bti==0 else None,
                                    random_state=random_state+pi*2000+mi)
                    model_m_als.append(model_m_al)
                    model_m_al_perfs.append(model_m_al_perf)
                    
                # do causal inference
                for ci, cim in enumerate(causal_inference_methods):
                    cdes, scies, cies0, cies1 = infer_mediation(cim, model_a_l, model_m_als, model_y_alm, Yte, Mte, Ate, Lte, random_state=random_state+pi*4000+ci)
                    
                    # add average performance
                    cdes.append(np.mean(cdes))
                    scies.append(np.mean(scies))
                    cies0.append(np.mean(cies0))
                    cies1.append(np.mean(cies1))
                    model_m_al_perfs.append(np.nan)
                    
                    res.append([bti, cvi, pm, cim, model_a_l_perf, model_y_alm_perf] + model_m_al_perfs + cdes + scies + cies0 + cies1)
                    #print(res[-1])
            
            with open('results_simulated_data.pickle', 'wb') as ff:
                pickle.dump(res, ff, protocol=2)
            
            #except Exception as ee:
            #    print(str(ee))
    #with open('results.pickle', 'rb') as ff:
    #    res = pickle.load(ff)
        
    res = np.array(res, dtype=object)
    Nbt = res[:,0].max()
    Mnames.append('avg')
    n_mediator = len(Mnames)

    perf_A_L_cols = ['perf(A|L)']
    perf_Y_ALM_cols = ['perf(Y|A,L,M)']
    perf_M_AL_cols = ['perf(M|A,L) %s'%x for x in Mnames]
    CDE_cols = ['CDE %s'%x for x in Mnames]
    sCIE_cols = ['sCIE %s'%x for x in Mnames]
    CIE0_cols = ['CIE0 %s'%x for x in Mnames]
    CIE1_cols = ['CIE1 %s'%x for x in Mnames]
    cols = perf_A_L_cols + perf_Y_ALM_cols + perf_M_AL_cols + CDE_cols + sCIE_cols + CIE0_cols + CIE1_cols
    columns = ['bt', 'fold', 'prediction_model', 'causal_inference_model'] + cols
    res = pd.DataFrame(data=res, columns=columns)
    
    # take the average across folds
    res2 = []
    for bti, pm, cim in product(range(Nbt+1), prediction_methods, causal_inference_methods):
        ids = (res.bt==bti) & (res.prediction_model==pm) & (res.causal_inference_model==cim)
        if ids.sum()==0:
            continue
        res2.append([bti, pm, cim] + list(res[ids][cols].mean(axis=0)))
    columns = ['bt', 'prediction_model', 'causal_inference_model'] + cols
    res = pd.DataFrame(data=res2, columns=columns)
    
    # add percentages
    for m in Mnames:
        total_effect = res['CDE %s'%m].values + res['sCIE %s'%m].values
        res.insert(res.shape[1], 'TotalEffect %s'%m, total_effect)
        cols.append('TotalEffect %s'%m)
    for col in ['CDE', 'sCIE']:
        for m in Mnames:
            res_perc = res['%s %s'%(col,m)].values/res['TotalEffect %s'%m].values*100
            res.insert(res.shape[1], '%%%s %s'%(col,m), res_perc)
            cols.append('%%%s %s'%(col,m))
    
    # add confidence interval
    res2 = []
    for pm, cim in product(prediction_methods, causal_inference_methods):
        ids1 = np.where((res.bt==0) & (res.prediction_model==pm) & (res.causal_inference_model==cim))[0]
        ids2 = np.where((res.bt>0) & (res.prediction_model==pm) & (res.causal_inference_model==cim))[0]
        if len(ids1)==0 or len(ids2)==0:
            continue
        assert len(ids1)==1
        
        vals = res.iloc[ids1[0]][cols].values
        lb = np.percentile(res.iloc[ids2][cols].values, 2.5, axis=0)
        ub = np.percentile(res.iloc[ids2][cols].values, 97.5, axis=0)
        res2.append([pm, cim] +
                       ['%.3f [%.3f -- %.3f]'%(vals[ii], lb[ii], ub[ii]) for ii in range(len(vals))])
    columns = ['prediction_model', 'causal_inference_model'] + cols
    res = pd.DataFrame(data=res2, columns=columns)
    
    col_names2 = ['Mediator',
                 '%CDE', '%sCIE',
                 'CDE', 'sCIE', 'TotalEffect',
                 'CIE0', 'CIE1',
                 'perf(A|L)', 'perf(M|A,L)', 'perf(Y|A,L,M)',]
    dfs = []
    for pm, cim in product(prediction_methods, causal_inference_methods):
        ids = np.where((res.prediction_model==pm) & (res.causal_inference_model==cim))[0]
        assert len(ids)==1
        
        # remove pm and cim
        res2 = res.iloc[ids][cols]
        
        # get values that are the same for all mediators
        a_l_perf = res2['perf(A|L)'].iloc[0]
        y_alm_perf = res2['perf(Y|A,L,M)'].iloc[0]
        
        # generate dataframe with each row being a mediator
        res2 = res2.drop(columns=['perf(A|L)', 'perf(Y|A,L,M)'])
        col_names = np.array(res2.columns).reshape(-1,n_mediator)[:,0]
        col_names = ['Mediator', 'perf(A|L)', 'perf(Y|A,L,M)'] + [x.split(' ')[0] for x in col_names]
        res2 = res2.values.reshape(-1,n_mediator).T
        res2 = np.c_[Mnames, [a_l_perf]*n_mediator, [y_alm_perf]*n_mediator, res2]
        df = pd.DataFrame(data=res2, columns=col_names)
        
        # reorder the dataframe
        df = df[col_names2]
        
        # sort based on %sCIE
        ids = np.argsort([float(x.split(' ')[0]) for x in df['%sCIE'].values])[::-1]
        df = df.iloc[ids].reset_index(drop=True)

        dfs.append(df)
            
    # save
    with pd.ExcelWriter('results_simulated_data.xlsx') as writer:
        for df in dfs:
            df.to_excel(writer, sheet_name='%s+%s'%(pm, cim), index=False)
    
