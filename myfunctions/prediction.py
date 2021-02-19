import numpy as np
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import minimize
#from scipy.special import expit as sigmoid
from scipy.special import logit
from joblib import dump, load, Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, BayesianRidge, LinearRegression
from sklearn.linear_model._logistic import _logistic_loss_and_grad
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer
from sklearn.utils import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import RandomOverSampler
#from bart import BARTClassifier
from learning_to_rank import LTRPairwise, MyXGBRanker


"""
class MyLogisticRegression(LogisticRegression):
    #Removes regularization on intercept
    #Allows bounds
    #Binary only
    #L2 only
    #L-BFGS-B or BFGS only
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
"""


class MyLogisticRegression(LogisticRegression):
    """
    binary
    converts {0,1} labels to [0--1] using nearest neighbors, then do linreg(X, logit(y))
    """
    def __init__(self, dist_func='euclidean', n_neighbors=100, weight='tricubic', max_iter=1000):
        self.dist_func = dist_func
        self.n_neighbors = n_neighbors
        self.weight = weight
        self.max_iter = max_iter
        self.solver = 'lbfgs'
        self.multi_class = 'auto'

    def get_y_cont(self, X, y):
        # encode labels
        self.le = LabelEncoder().fit(y)
        self.classes_ = self.le.classes_
        y = self.le.transform(y)
        class_set = set(self.classes_)
        
        # get pairwise distance and its max
        dist = pdist(X, metric=self.dist_func)
        self.max_dist = np.percentile(dist, 95)
        
        # generate sorted ids
        dist = squareform(dist)
        dist[range(len(X)),range(len(X))] = np.inf
        sorted_ids = np.argsort(dist, axis=1)
        """ # or decide based on thresholding weight
        sorted_dists = np.sort(dist,axis=1)
        ww = (1-np.minimum(sorted_dists/self.max_dist,1)**3)**3
        n_neighbors = np.where(ww.mean(axis=0)>=0.9)[0][-1]
        """
        
        # generate close ids and weighted y
        y2 = []
        for i in range(len(dist)):
            sorted_id = sorted_ids[i]
            close_ys = y[sorted_id[:self.n_neighbors]]
            if set(close_ys)==class_set:
                # if nearest neighbor contains both classes
                close_dist = dist[i][sorted_id[:self.n_neighbors]]
            else:
                # if nearest neighbor not contain both classes
                # search further until the next missing class occurs
                missing_class = list(class_set - set(close_ys))[0]
                n_neighbors = np.where(y[sorted_id]==missing_class)[0][0]
                close_ys = y[sorted_id[:n_neighbors+1]]
                close_dist = dist[i][sorted_id[:n_neighbors+1]]
            
            if self.weight=='tricubic':
                d = np.minimum(close_dist/self.max_dist, 1)
                close_weight = (1-d**3)**3
            else:
                raise NotImplementedError(self.weight)
            #if self.weight=='softmax':
            close_weight /= close_weight.mean()
            y2.append( (close_ys*close_weight).mean() )
            
        return np.array(y2)
        
    def fit(self, X, y, y2=None, sample_weight=None):
        
        if y2 is None:
            y2 = self.get_y_cont(self, X, y)
        else:
            # encode labels
            self.le = LabelEncoder().fit(y)
            self.classes_ = self.le.classes_
            
        self._model = BayesianRidge(n_iter=self.max_iter).fit(X, logit(y2), sample_weight=sample_weight)
        #self._model = LinearRegression().fit(X, logit(y2), sample_weight=sample_weight)
        self.coef_ = self._model.coef_.reshape(1,-1)
        self.intercept_ = self._model.intercept_.reshape(1)
        
        return self


class MyDummyClassifier(BaseEstimator, ClassifierMixin):
    # always return P(y=1)
    def fit(self, X, y):
        self.label_encoder = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder.classes_

        y = self.label_encoder.transform(y)
        self.yp = np.mean(y, axis=0)
        return self

    def predict_proba(self, X):
        if len(self.classes_)==2:
            yp = np.array([self.yp]*len(X))
            return np.array([1-yp, yp]).T
        else:
            return np.array([[self.yp]*len(X)])

    def predict(self, X):
        yp = self.predict_proba(X)
        yp = np.argmax(yp, axis=1)
        yp = self.label_encoder.inverse_transform(yp)
        return yp


class LocalModelBase(BaseEstimator):
    def __init__(self, model, frac=0.2, dist_func='euclidean', weight='tricubic', n_jobs=1, verbose=False):
        self.model = model
        self.frac = frac
        self.dist_func = dist_func
        self.weight = weight
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def decide_neighbor(self, X):
        dists = cdist(X, self.X, metric=self.dist_func)#/np.sqrt(X.shape[1])
        max_dist = np.percentile(dists.flatten(), 95)
        n_neighbors = int(round(self.frac*len(X)))

        neighbor_ids = np.argsort(dists, axis=1)[:,:n_neighbors]
        neighbor_dists = np.sort(dists, axis=1)[:,:n_neighbors]
        if self.weight=='tricubic':
            d = np.minimum(neighbor_dists/max_dist, 1)
            weights = (1-d**3)**3
        else:
            raise NotImplementedError(self.weight)

        return neighbor_ids, weights


class LOWESSClassifier(LocalModelBase, ClassifierMixin):
    def __init__(self, model, frac=0.2, dist_func='euclidean', weight='tricubic', n_jobs=1, verbose=False):
        super().__init__(model, frac=frac, dist_func=dist_func, weight=weight, n_jobs=n_jobs, verbose=verbose)

    def fit(self, X, y, y2=None):
        self.X = np.array(X)

        self.le = LabelEncoder()
        self.le.fit(y)
        self.classes_ = self.le.classes_
        self.y = self.le.transform(y)
        if y2 is None:
            self.y2 = None
        else:
            self.y2 = np.array(y2)
    
        return self

    def predict_proba(self, X):
        def _inner_predict(model_, X, Xref, yref, swref, y2ref):
            model = clone(model_)
            if y2ref is None:
                local_model = model.fit(Xref, yref, sample_weight=swref)
            else:
                local_model = model.fit(Xref, yref, y2=y2ref, sample_weight=swref)
            return local_model.predict_proba(X.reshape(1,-1))[0]

        neighbors, weights = self.decide_neighbor(X)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            yp = parallel(delayed(_inner_predict)(
                self.model, X[i], self.X[neighbors[i]], self.y[neighbors[i]], weights[i], None if self.y2 is None else self.y2[neighbors[i]])
                    for i in tqdm(range(len(X)), disable=not self.verbose))

        return np.array(yp)
        
    def predict(self, X):
        yp = self.predict_proba(X)
        yp = self.le.inverse_transform(np.argmax(yp, axis=1))
        return yp


class LOWESSRegressor(LocalModelBase, RegressorMixin):
    def __init__(self, model, frac=0.2, dist_func='euclidean', weight='tricubic', n_jobs=1, verbose=False):
        super().__init__(model, frac=frac, dist_func=dist_func, weight=weight, n_jobs=n_jobs, verbose=verbose)

    def fit(self, X, y, y2=None):
        self.X = np.array(X)
        self.y = np.array(y)
        return self

    def predict(self, X):
        def _inner_predict(model_, X, Xref, yref, swref):
            model = clone(model_)
            local_model = model.fit(Xref, yref, sample_weight=swref)
            return local_model.predict(X.reshape(1,-1))[0]

        neighbors, weights = self.decide_neighbor(X)

        with Parallel(n_jobs=self.n_jobs) as parallel:
            yp = parallel(delayed(_inner_predict)(
                self.model, X[i], self.X[neighbors[i]], self.y[neighbors[i]], weights[i])
                    for i in tqdm(range(len(X)), disable=not self.verbose))

        return np.array(yp)
        

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
    n_jobs = 12
    
    if model_type.endswith('clf'):
        scorer = 'f1_weighted'
            
        if model_name=='dummy':
            model = MyDummyClassifier()

        elif model_name=='linear':
            #Pipeline((
            #    ('standardization', StandardScaler()),
            #model = MyLogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced')
            #model = GridSearchCV(model, {'C':np.logspace(1,4,10)}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            model = MyLogisticRegression(n_neighbors=100)
            #model = GridSearchCV(model, {'n_neighbors':[50,100]}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)

        elif model_name=='lowess-linear':
            model = LOWESSClassifier(MyLogisticRegression(), frac=0.2, n_jobs=n_jobs)
            #model = GridSearchCV(model, {'frac':[0.1,0.2,0.5]}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='xgb':
            model = XGBClassifier(verbosity=0, n_jobs=1, random_state=random_state, class_weight='balanced')
            model = GridSearchCV(model,
                        {'learning_rate':[0.1,0.2,0.3],
                        'max_depth':[3,5],
                        'reg_lambdas':[0.01,0.1],
                        'subsample':[0.5,1],},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='rf':
            model = RandomForestClassifier(random_state=random_state, class_weight='balanced')
            model = GridSearchCV(model,
                        {'n_estimators':[5, 10],
                        'max_depth':[3,5],
                        'min_samples_leaf':[10,20]},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
        elif model_name=='svm':
            model = LinearSVC(penalty='l2', dual=False, class_weight='balanced', random_state=random_state, max_iter=1000)
            model = GridSearchCV(model, {'C':np.logspace(-4,-1,10)}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
    
    
    elif model_type=='reg':
        scorer = 'neg_root_mean_squared_error'
            
        if model_name=='linear':
            model = BayesianRidge(n_iter=1000)

        elif model_name=='lowess-linear':
            model = LOWESSRegressor(BayesianRidge(n_iter=1000), frac=0.2, n_jobs=n_jobs)
            #model = GridSearchCV(model, {'frac':[0.1,0.2,0.5]}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='xgb':
            model = XGBRegressor(verbosity=0, n_jobs=1, random_state=random_state)
            model = GridSearchCV(model,
                        {'learning_rate':[0.1,0.2,0.3],
                        'max_depth':[3,5],
                        'reg_lambdas':[0.01,0.1],
                        'subsample':[0.5,1],},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            
        elif model_name=='rf':
            model = RandomForestRegressor(random_state=random_state)
            model = GridSearchCV(model,
                        {'n_estimators':[5, 10],
                        'max_depth':[3,5],
                        'min_samples_leaf':[10,20]},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
    
    elif model_type=='ltr':
        #scorer = make_scorer(lambda y,yp:kendalltau(y,yp)[0])
        scorer = 'f1_weighted'
            
        if model_name=='linear':
            #model = LTRPairwise(MyLogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'))
            #model = GridSearchCV(model, {'estimator__C':np.logspace(1,4,10)}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
            model = LTRPairwise(MyLogisticRegression(n_neighbors=100))
            #model = GridSearchCV(model, {'estimator__n_neighbors':[50,100]}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
        elif model_name=='svm':
            model = LTRPairwise(LinearSVC(penalty='l2', dual=False, class_weight='balanced', random_state=random_state, max_iter=1000))
            model = GridSearchCV(model, {'estimator__C':np.logspace(-4,-1,10)}, scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
        elif model_name=='xgb':
            model = MyXGBRanker(verbosity=0, n_jobs=1, random_state=random_state)
            model = GridSearchCV(model,
                        {'learning_rate':[0.1,0.2,0.3],
                        'max_depth':[3,5],
                        'reg_lambdas':[0.01,0.1],
                        'subsample':[0.5,1],},
                        scoring=scorer, n_jobs=n_jobs, refit=True, cv=cv)
        
    return model
    
    
def fit_prediction_model(model_name, X, y, y2=None, save_path=None, random_state=None):
    
    # oversample the minor class
    #resampler = RandomOverSampler(sampling_strategy='auto', random_state=random_state)
    #X, y = resampler.fit_sample(X, y)
    
    # get model
    model_name, model_type = model_name.split(':')
    model = get_model(model_name, model_type, random_state=random_state)
    
    # fit
    if y2 is None:
        model.fit(X, y)
    else:
        model.fit(X, y, y2=y2)
    if hasattr(model, 'best_score_'):
        best_perf = model.best_score_
        model = model.best_estimator_
    else:
        best_perf = model.score(X, y)
    
    # calibrate
    if model_type.endswith('clf') and model_name!='dummy':
        model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        model.fit(X, y)
    
    if save_path is not None:
        # save
        dump(model, save_path+'.joblib')
        #model = load(save_path+'.joblib')
            
    return model, best_perf

