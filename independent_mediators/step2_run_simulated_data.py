from itertools import product
import os
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
import sys
sys.path.insert(0, '../myfunctions')
from prediction import fit_prediction_model, MyLogisticRegression
from causal_inference import infer_mediation, select_estimator
    

if __name__=='__main__':
    ## load data

    res = sio.loadmat('simulated_data.mat')
    A = res['A'].flatten()
    L = res['L']
    Y = res['Y'].flatten()
    M = res['M']
    sids = np.arange(len(A))
    Mnames = ['M%d'%(i+1,) for i in range(M.shape[1])]
    Mnames.append('avg')
    n_mediator = len(Mnames)
    
    print('N', len(Y))
    print('CDE0_1', res['CDE0_1'][0,0])
    print('CIE1_1', res['CIE1_1'][0,0])
    print('CIE0_1', res['CIE0_1'][0,0])
    print('sCIE1', res['sCIE1'][0,0])
    print('TE1', res['TE1'][0,0])
    print('CDE0_2', res['CDE0_2'][0,0])
    print('CIE1_2', res['CIE1_2'][0,0])
    print('CIE0_2', res['CIE0_2'][0,0])
    print('sCIE2', res['sCIE2'][0,0])
    print('TE2', res['TE2'][0,0])
    print('TE', res['TE'][0,0])
    
    ## set up numbers

    random_state = 2020
    model_outcome = 'linear'
    model_exposure = 'linear'
    ci_method = 'dr'
    Nbt = 1000
    
    ## generate cv split
    
    cv_path = 'patients_cv_split_simulated.pickle'
    if os.path.exists(cv_path):
        with open(cv_path, 'rb') as ff:
            tr_sids, te_sids = pickle.load(ff)
    else:
        cvf = 5
        sids2 = np.array(sids)
        np.random.seed(random_state)
        np.random.shuffle(sids2)
        tr_sids = []
        te_sids = []
        cv_split = np.array_split(sids2, cvf)
        for cvi in cv_split:
            te_sids.append(np.sort(cvi))
            tr_sids.append(np.sort(np.setdiff1d(sids2, cvi)))
        with open(cv_path, 'wb') as ff:
            pickle.dump([tr_sids, te_sids], ff)
            
    np.random.seed(random_state)
    
    
    if model_exposure == 'linear':
        Lmean = L.mean(axis=0)
        Lstd = L.std(axis=0)
        L2 = (L-Lmean)/Lstd
        lr = MyLogisticRegression(n_neighbors=100)
        A2 = lr.get_y_cont(L, A)
        M2 = np.zeros_like(M).astype(float)
        M2[:,0] = lr.get_y_cont(np.c_[A,L], M[:,0])
        M2[:,1] = lr.get_y_cont(np.c_[A,L], M[:,1])
        
    ## bootstrapping

    #"""
    res = []
    
    for bti in tqdm(range(Nbt+1)):
        np.random.seed(random_state+bti)
        if bti==0:
            # use the actual data
            Abt = A
            Ybt = Y
            Lbt = L
            Mbt = M
            if model_exposure == 'linear':
                Abt2 = A2
                Mbt2 = M2
            sidsbt = sids
        else:
            # use bootstrapped data
            btids = np.random.choice(len(Y), len(Y), replace=True)
            Abt = A[btids]
            Ybt = Y[btids]
            Lbt = L[btids]
            Mbt = M[btids]
            sidsbt = sids[btids]
            if model_exposure == 'linear':
                Abt2 = A2[btids]
                Mbt2 = M2[btids]
        
        # outer loop cross validation
        for cvi in range(len(tr_sids)):
            #print(cvi)
            trid = np.in1d(sidsbt, tr_sids[cvi])
            teid = np.in1d(sidsbt, te_sids[cvi])
            
            Ytr = Ybt[trid]
            Ltr = Lbt[trid]
            Atr = Abt[trid]
            Mtr = Mbt[trid]
            if model_exposure == 'linear':
                Atr2 = Abt2[trid]
                Mtr2 = Mbt2[trid]
                
            Ate = Abt[teid]
            Yte = Ybt[teid]
            Lte = Lbt[teid]
            Mte = Mbt[teid] 
            
            Lmean = Ltr.mean(axis=0)
            Lstd = Ltr.std(axis=0)
            Ltr = (Ltr-Lmean)/Lstd
            Lte = (Lte-Lmean)/Lstd
                
            # fit A|L
            model_a_l, model_a_l_perf = fit_prediction_model(
                    model_exposure+':bclf', Ltr,
                    Atr, y2=Atr2, random_state=random_state+cvi+1000)
        
            # fit Y|A,L,M
            model_y_alm, model_y_alm_perf = fit_prediction_model(
                    model_outcome+':reg', np.c_[Atr, Ltr, Mtr], Ytr,
                    random_state=random_state+cvi*3000)
                                    
            model_m_als = []
            model_m_al_perfs = []
            for mi, mediator_name in enumerate(Mnames[:-1]):
                # fit Mi|A,L
                model_m_al, model_m_al_perf = fit_prediction_model(
                        model_exposure+':bclf', np.c_[Atr, Ltr],
                        Mtr[:,mi], y2=Mtr2[:,mi],
                        random_state=random_state+cvi*2000+mi)
                model_m_als.append(model_m_al)
                model_m_al_perfs.append(model_m_al_perf)
                
            # do causal inference
            cdes, scies, cies0, cies1 = infer_mediation(ci_method, model_a_l, model_m_als, model_y_alm,
                                                        Yte, Mte, Ate, Lte, random_state=random_state+cvi*4000)
            
            # add average performance
            cdes.append(np.mean(cdes))
            scies.append(np.mean(scies))
            cies0.append(np.mean(cies0))
            cies1.append(np.mean(cies1))
            model_m_al_perfs.append(np.nan)
            
            res.append([bti, cvi, model_outcome, model_exposure, ci_method, model_a_l_perf, model_y_alm_perf] + model_m_al_perfs + cdes + scies + cies0 + cies1)
            #print(res[-1])

        """
        if bti==0:
            #@misc{cui2019selective,
            #    title={Selective machine learning of doubly robust functionals},
            #    author={Yifan Cui and Eric Tchetgen Tchetgen},
            #    year={2019},
            #    eprint={1911.02029},
            #    archivePrefix={arXiv}}
            _,_,best_pm_o, best_pm_e = select_estimator(
                                            np.array([x[1] for x in res if x[0]==bti]),
                                            [(x[2],x[3]) for x in res if x[0]==bti],
                                            np.array([x[-n_mediator*4+n_mediator-1]+x[-n_mediator*3+n_mediator-1] for x in res if x[0]==bti]))
            print('best prediction model', best_pm_o, best_pm_e)
            res = [x for x in res if x[2]==best_pm_o and x[3]==best_pm_e]
        """

        with open('results_simulated_data.pickle', 'wb') as ff:
            pickle.dump(res, ff, protocol=2)
    #"""
    #with open('results_simulated_data.pickle', 'rb') as ff:
    #    res, best_pm_o, best_pm_e = pickle.load(ff)
        
    res = np.array(res, dtype=object)
    Nbt = res[:,0].max()

    perf_A_L_cols = ['perf(A|L)']
    perf_Y_ALM_cols = ['perf(Y|A,L,M)']
    perf_M_AL_cols = ['perf(M|A,L) %s'%x for x in Mnames]
    CDE_cols = ['CDE %s'%x for x in Mnames]
    sCIE_cols = ['sCIE %s'%x for x in Mnames]
    CIE0_cols = ['CIE0 %s'%x for x in Mnames]
    CIE1_cols = ['CIE1 %s'%x for x in Mnames]
    cols = perf_A_L_cols + perf_Y_ALM_cols + perf_M_AL_cols + CDE_cols + sCIE_cols + CIE0_cols + CIE1_cols
    columns = ['bt', 'fold', 'outcome prediction_model', 'exposure prediction model', 'causal_inference_model'] + cols
    res = pd.DataFrame(data=res, columns=columns)
    
    # take the average across folds
    res2 = []
    for bti in range(Nbt+1):
        ids = res.bt==bti
        if ids.sum()==0:
            continue
        res2.append([bti] + list(res[ids][cols].mean(axis=0)))
    columns = ['bt'] + cols
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
    ids1 = np.where(res.bt==0)[0]
    ids2 = np.where(res.bt>0)[0]
    assert len(ids1)==1
    
    vals = res.iloc[ids1[0]][cols].values
    lb = np.percentile(res.iloc[ids2][cols].values, 2.5, axis=0)
    ub = np.percentile(res.iloc[ids2][cols].values, 97.5, axis=0)
    res2 = np.array([['%.2f [%.2f -- %.2f]'%(vals[ii], lb[ii], ub[ii]) for ii in range(len(vals))]])
    columns = cols
    res = pd.DataFrame(data=res2, columns=columns)
    
    col_names2 = ['Mediator',
                 '%CDE', '%sCIE',
                 'CDE', 'sCIE', 'TotalEffect',
                 'CIE0', 'CIE1',
                 'perf(A|L)', 'perf(M|A,L)', 'perf(Y|A,L,M)',]
    res2 = res[cols]
    
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
    print(df)
    import pdb;pdb.set_trace()
            
    # save
    df.to_excel('results_simulated_data_%s_%s_%s.xlsx'%(model_outcome, model_exposure, ci_method), index=False)
    
