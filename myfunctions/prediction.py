import numpy as np
from scipy.stats import spearmanr, kendalltau
from scipy.special import expit as sigmoid
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, BayesianRidge
from skbayes.linear_models import EBLogisticRegression,VBLogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import RandomOverSampler
#from bart import BARTClassifier
from learning_to_rank import LTRPairwise


def y_2d_to_1d(y):
    if y.ndim==2:
        if y.shape[1]==2:
            y = y[:,1]
        else:
            y = y.flatten()
    return y
        

def myspearmanr(y, yp):
    return spearmanr(y,yp[:,1])[0]


def get_model(model_name, model_type, cv=5, random_state=None):
    n_jobs = 1
    
    if model_type.endswith('clf'):
        scorer = 'f1_weighted'
        if model_type=='bclf':
            scorer = 'roc_auc'
            
        if model_name=='linear':
            #Pipeline((
            #    ('standardization', StandardScaler()),
            model = LogisticRegression(penalty='l2', random_state=random_state, solver='lbfgs', max_iter=1000, class_weight='balanced')
            model = GridSearchCV(model, {'C':[0.01,0.1,1,10,100]}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            #model = VBLogisticRegression(n_iter=100)
            #model = GridSearchCV(model, {'a':[1e-6]}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='xgb':
            model = XGBClassifier(learning_rate=0.1, verbosity=0, n_jobs=1, random_state=random_state, class_weight='balanced')
            model = GridSearchCV(model,
                        {'n_estimators':[20,50],
                        'max_depth':[5,10]},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='rf':
            model = RandomForestClassifier(random_state=random_state, class_weight='balanced')
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
    
    elif model_type=='ltr':
        scorer = make_scorer(lambda y,yp:kendalltau(y,yp)[0])
            
        if model_name=='linear':
            model = LTRPairwise(LogisticRegression(penalty='l2', random_state=random_state, solver='lbfgs', max_iter=1000, class_weight='balanced'))
            model = GridSearchCV(model, {'estimator__C':[0.01,0.1,1,10,100]}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
        
    return model
    
    
def fit_prediction_model(model_name, X, y, save_path=None, random_state=None):
    
    # oversample the minor class
    #resampler = RandomOverSampler(sampling_strategy='auto', random_state=random_state)
    #X, y = resampler.fit_sample(X, y)
    
    # get model
    model_name, model_type = model_name.split(':')
    model = get_model(model_name, model_type, random_state=random_state)
    
    # fit
    model.fit(X, y)
    best_perf = model.best_score_
    model = model.best_estimator_
    
    # calibrate
    if model_type.endswith('clf'):
        model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit').fit(X, y)
    
    if save_path is not None:
        # save
        dump(model, save_path+'.joblib')
        #model = load(save_path+'.joblib')
            
    return model, best_perf

