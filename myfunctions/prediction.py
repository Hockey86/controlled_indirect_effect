import numpy as np
from scipy.stats import spearmanr
from scipy.special import expit as sigmoid
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import RandomOverSampler
#from bart import BARTClassifier


def y_2d_to_1d(y):
    if y.ndim==2:
        if y.shape[1]==2:
            y = y[:,1]
        else:
            y = y.flatten()
    return y
        

def myspearmanr(y, yp):
    return spearmanr(y,yp[:,1])[0]


def get_model(model_name, cv=5, random_state=None):
    model_name, model_type = model_name.split(':')
    n_jobs = 1
    
    if model_type.endswith('clf'):
        scorer = 'f1_weighted'
        if model_type=='bclf':
            scorer = 'roc_auc'
            
        if model_name=='linear':
            #Pipeline((
            #    ('standardization', StandardScaler()),
            model = LogisticRegression(penalty='l2', random_state=random_state, solver='lbfgs', max_iter=1000)
            model = GridSearchCV(model,
                        {'C':[0.01,0.1,1.]},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='xgb':
            model = XGBClassifier(learning_rate=0.1, verbosity=0, n_jobs=1, random_state=random_state)
            model = GridSearchCV(model,
                        {'n_estimators':[20,50],
                        'max_depth':[5,10]},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='rf':
            model = RandomForestClassifier(random_state=random_state)
            model = GridSearchCV(model,
                        {'n_estimators':[50],
                        'max_depth':[3],
                        'min_samples_split':[2],
                        'min_samples_leaf':[10,20]},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
    
    
    elif model_type=='reg':
        scorer = 'neg_root_mean_squared_error'
            
        if model_name=='linear':
            model = BayesianRidge(n_iter=1000)
            model = GridSearchCV(model,
                        {'alpha_1':[1e-6]},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='xgb':
            model = XGBRegressor(learning_rate=0.1, verbosity=0, n_jobs=1, random_state=random_state)
            model = GridSearchCV(model,
                        {'n_estimators':[20,50],
                        'max_depth':[5,10]},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='rf':
            model = RandomForestRegressor(random_state=random_state)
            model = GridSearchCV(model,
                        {'n_estimators':[50],
                        'max_depth':[3],
                        'min_samples_split':[2],
                        'min_samples_leaf':[10,20]},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
        
    return model
    
    
def fit_prediction_model(model_name, X, y, save_path=None, random_state=None):
    
    # oversample the minor class
    #resampler = RandomOverSampler(sampling_strategy='auto', random_state=random_state)
    #X, y = resampler.fit_sample(X, y)
    
    # get model
    model = get_model(model_name, random_state=random_state)
    
    # fit
    model.fit(X, y)
    best_perf = model.best_score_
    model = model.best_estimator_
    
    if save_path is not None:
        # save
        dump(model, save_path+'.joblib')
        #model = load(save_path+'.joblib')
            
    return model, best_perf

