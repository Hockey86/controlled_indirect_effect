from collections import Counter
from itertools import product
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import sys
sys.path.insert(0, '../myfunctions')
from prediction import fit_prediction_model
#from causal_inference import infer_mediation_3mediator
from causal_inference import infer_mediation, select_estimator


if __name__=='__main__':
    ## load data

    res = pd.read_excel('../data/hiv-brain-age.xlsx')
    A = res['HIV'].values.astype(float)
    
    res.loc[res.Sex=='M', 'Sex'] = 1
    res.loc[res.Sex=='F', 'Sex'] = 0
    race = OneHotEncoder(sparse=False).fit_transform(res.Race.values.astype(str).reshape(-1,1))
    L = np.c_[res[['Age', 'Sex', 'Tobacco use disorder', 'Alcoholism']].values.astype(float), race]
    
    Y = res['BAI'].values.astype(float)
    
    Mnames = ['obesity', 'heart disorder', 'sleep disorder']
    M = res[Mnames].values.astype(float)
    for mm in Mnames:
        print(mm, Counter(res[mm]))
    Mnames.append('avg')
    n_mediator = len(Mnames)

    sids = np.arange(len(A))
    
    ## set up numbers

    random_state = 2020
    prediction_methods = ['linear', 'rf']#, 'xgb']
    ci_method = 'dr'
    Nbt = 1000
    np.random.seed(random_state)
    
    ## generate cv split

    """
    cv_path = 'patients_cv_split_real_data2.pickle'
    if os.path.exists(cv_path):
        with open(cv_path, 'rb') as ff:
            tr_sids, te_sids = pickle.load(ff)
    else:
        cvf = 5
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
    """
    tr_sids = [sids]
    te_sids = [sids]

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
            sidsbt = sids
            prediction_methods_outcome = prediction_methods
            prediction_methods_exposure = prediction_methods
        else:
            # use bootstrapped data
            btids = np.random.choice(len(Y), len(Y), replace=True)
            Abt = A[btids]
            Ybt = Y[btids]
            Lbt = L[btids]
            Mbt = M[btids]
            sidsbt = sids[btids]
            prediction_methods_outcome = [best_pm_o]
            prediction_methods_exposure = [best_pm_e]
        
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
            Ltr = (Ltr-Lmean)/Lstd
            Lte = (Lte-Lmean)/Lstd
            
            #try:
            for pi, pm in enumerate(product(prediction_methods_outcome, prediction_methods_exposure)):
                if bti==0:
                    print(pm)
                pm_outcome, pm_exposure = pm
                
                # fit A|L
                model_a_l, model_a_l_perf = fit_prediction_model(pm_exposure+':bclf', Ltr, Atr,
                                        random_state=random_state+pi+1000)

                # fit Y|A,L,M
                model_y_alm, model_y_alm_perf = fit_prediction_model(pm_outcome+':reg', np.c_[Atr, Ltr, Mtr], Ytr,
                                    random_state=random_state+pi*3000)

                model_m_als = []
                model_m_al_perfs = []
                for mi, mediator_name in enumerate(Mnames[:-1]):
                    """
                    if mi==0:
                        # fit M1|A,L
                        model_m_al, model_m_al_perf = fit_prediction_model(pm+':bclf', np.c_[Atr, Ltr], Mtr[:, mi],
                                    save_path='models_real_data2/med_model_%s_cv%d_%s'%(mediator_name, cvi+1, pm) if bti==0 else None,
                                    random_state=random_state+pi*2000+mi)
                    elif mi==1:
                        # fit M2|A,L,M1
                        model_m_al, model_m_al_perf = fit_prediction_model(pm+':bclf', np.c_[Atr, Ltr, Mtr[:,0]], Mtr[:, mi],
                                    save_path='models_real_data2/med_model_%s_cv%d_%s'%(mediator_name, cvi+1, pm) if bti==0 else None,
                                    random_state=random_state+pi*2000+mi)
                    elif mi==2:
                        # fit M3|A,L,M1,M2
                        model_m_al, model_m_al_perf = fit_prediction_model(pm+':bclf', np.c_[Atr, Ltr, Mtr[:,[0,1]]], Mtr[:, mi],
                                    save_path='models_real_data2/med_model_%s_cv%d_%s'%(mediator_name, cvi+1, pm) if bti==0 else None,
                                    random_state=random_state+pi*2000+mi)
                    """
                    # fit Mi|A,L
                    model_m_al, model_m_al_perf = fit_prediction_model(pm_exposure+':bclf', np.c_[Atr, Ltr], Mtr[:, mi],
                                        random_state=random_state+pi*2000+mi)
                    model_m_als.append(model_m_al)
                    model_m_al_perfs.append(model_m_al_perf)

                # do causal inference
                #cdes, scies, cies0, cies1 = infer_mediation_3mediator(cim, model_a_l, model_m_als, model_y_alms, Yte, Mte, Ate, Lte, random_state=random_state+pi*4000+ci)
                cdes, scies, cies0, cies1 = infer_mediation(ci_method, model_a_l, model_m_als, model_y_alm,
                                                            Yte, Mte, Ate, Lte, random_state=random_state+pi*4000)
                
                # add average performance
                cdes.append(np.mean(cdes))
                scies.append(np.mean(scies))
                cies0.append(np.mean(cies0))
                cies1.append(np.mean(cies1))
                model_m_al_perfs.append(np.nan)
                
                res.append([bti, cvi, pm_outcome, pm_exposure, ci_method, model_a_l_perf, model_y_alm_perf] + model_m_al_perfs + cdes + scies + cies0 + cies1)
                #print(res[-1])
            
            #except Exception as ee:
            #    print(str(ee))

        if bti==0:
            _,_,best_pm_o, best_pm_e = select_estimator(
                                            np.array([x[1] for x in res if x[0]==bti]),
                                            [(x[2],x[3]) for x in res if x[0]==bti],
                                            np.array([x[-n_mediator*4+n_mediator-1]+x[-n_mediator*3+n_mediator-1] for x in res if x[0]==bti]))
            print('best prediction model: outcome: %s; exposure: %s'%(best_pm_o, best_pm_e))
            res = [x for x in res if x[2]==best_pm_o and x[3]==best_pm_e]

        with open('results_real_data2.pickle', 'wb') as ff:
            pickle.dump([res, best_pm_o, best_pm_e], ff, protocol=2)
    #"""
    #with open('results_real_data2.pickle', 'rb') as ff:
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
    columns = ['bt', 'fold', 'outcome_prediction_model', 'exposure_prediction_model', 'causal_inference_model'] + cols
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
    res2 = np.array([['%.3f [%.3f -- %.3f]'%(vals[ii], lb[ii], ub[ii]) for ii in range(len(vals))]])
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

    # save
    import pdb;pdb.set_trace()
    df.to_excel('results_simulated_data_%s_%s_%s.xlsx'%(best_pm_o, best_pm_e, ci_method), index=False)
    
