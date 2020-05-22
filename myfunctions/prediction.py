import numpy as np
from scipy.stats import spearmanr, kendalltau
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.linear_model._logistic import _logistic_loss_and_grad
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import RandomOverSampler
#from bart import BARTClassifier
from learning_to_rank import LTRPairwise


class MyLogisticRegression(LogisticRegression):
    """
    Removes regularization on intercept
    Allows bounds
    Binary only
    L2 only
    L-BFGS-B or BFGS only
    """
    def __init__(self, tol=1e-6, C=1.0, random_state=None, max_iter=1000, class_weight=None, bounds=None):
        super().__init__(penalty='l2', dual=False, tol=tol, C=C,
                 fit_intercept=True, intercept_scaling=1, class_weight=class_weight,
                 random_state=random_state, solver='lbfgs', max_iter=max_iter,
                 multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None)
        self.bounds = bounds
                 
    def fit(self, X, y, sample_weight=None):
        self.label_encoder = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder.classes_
        
        def func(w, X, y, alpha, sw):
            out, grad = _logistic_loss_and_grad(w, X, y, 0, sw)
            out_penalty = 0.5*alpha*np.sum(w[:-1]**2)
            grad_penalty = np.r_[alpha*w[:-1] ,0]
            return out+out_penalty, grad+grad_penalty
        
        y2 = np.array(y)
        y2[y2==0] = -1
        w0 = np.r_[np.random.randn(X.shape[1])/10, 0.]
        if self.bounds is None:
            method = 'BFGS'
        else:
            method = 'L-BFGS-B'
        if sample_weight is None:
            sample_weight = np.ones(len(X))
            if self.class_weight is not None:
                class_weight_ = compute_class_weight(self.class_weight,
                                                 classes=[0,1],
                                                 y=y)
                sample_weight *= class_weight_[y.astype(int)]
        sample_weight /= (sample_weight.mean()*len(X))
        self.opt_res = minimize(
            func, w0, method=method, jac=True,
            args=(X, y2, 1./self.C, sample_weight),
            bounds=self.bounds,
            options={"gtol": self.tol, "maxiter": self.max_iter}
        )
        self.coef_ = self.opt_res.x[:X.shape[1]].reshape(1,-1)
        self.intercept_ = self.opt_res.x[-1].reshape(1,)
        return self

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
            
        if model_name=='linear':
            #Pipeline((
            #    ('standardization', StandardScaler()),
            model = MyLogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')
            model = GridSearchCV(model, {'C':np.logspace(1,4,10)}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='xgb':
            model = XGBClassifier(verbosity=0, n_jobs=1, random_state=random_state, class_weight='balanced')
            model = GridSearchCV(model,
                        {'learning_rate':[0.1,0.2,0.3],
                        'max_depth':[3,5,6,10],
                        'reg_lambdas':[0.01,0.1,1],
                        'subsample':[0.5,1],},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='rf':
            model = RandomForestClassifier(random_state=random_state, class_weight='balanced')
            model = GridSearchCV(model,
                        {'n_estimators':[10,30],
                        'max_depth':[3,5],
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
            model = XGBRegressor(verbosity=0, n_jobs=1, random_state=random_state)
            model = GridSearchCV(model,
                        {'learning_rate':[0.1,0.2,0.3],
                        'max_depth':[3,5,6,10],
                        'reg_lambdas':[0.01,0.1,1],
                        'subsample':[0.5,1],},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='rf':
            model = RandomForestRegressor(random_state=random_state)
            model = GridSearchCV(model,
                        {'n_estimators':[10,50],
                        'max_depth':[3,5],
                        'min_samples_leaf':[10,20,50]},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
    
    elif model_type=='ltr':
        scorer = make_scorer(lambda y,yp:kendalltau(y,yp)[0])
            
        if model_name=='linear':
            model = LTRPairwise(MyLogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'))
            model = GridSearchCV(model, {'estimator__C':np.logspace(1,4,10)}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
        
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
        model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        model.fit(X, y)
    
    if save_path is not None:
        # save
        dump(model, save_path+'.joblib')
        #model = load(save_path+'.joblib')
            
    return model, best_perf

