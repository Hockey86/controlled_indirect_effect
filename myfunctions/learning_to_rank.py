from itertools import combinations
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from scipy.special import softmax
from scipy.stats import spearmanr, kendalltau
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRanker


class LTRPairwise(BaseEstimator, ClassifierMixin):
    """Learning to rank, pairwise approach
    For each pair A and B, learn a score so that A>B or A<B based on the ordering.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        It must be a classifier with a ``decision_function`` function.
    verbose : bool, optional, defaults to False
        Whether prints more information.
    """
    def __init__(self, estimator, min_level_diff=2, verbose=False):
        super().__init__()
        self.estimator = estimator
        self.verbose = verbose
        self.min_level_diff = min_level_diff

    def _generate_pairs(self, X, y):#, sample_weight):
        X2 = []
        y2 = []
        sw2 = []
        for i, j in combinations(range(len(X)), 2):
            # if there is a tie, ignore it
            if np.abs(y[i]-y[j])<self.min_level_diff:
                continue
            X2.append( X[i]-X[j] )
            y2.append( 1 if y[i]>y[j] else 0 )
            #if sample_weight is not None:
            #    sw2.append( max(sample_weight[i], sample_weight[j]) )

        #if sample_weight is None:
        #    sw2 = None
        #else:
        #    sw2 = np.array(sw2)

        return np.array(X2), np.array(y2)#, sw2

    def fit(self, X, y):
        self.label_encoder = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder.classes_

        # generate pairs
        X2, y2 = self._generate_pairs(X, y)#, sample_weight), sw2
        if self.verbose:
            print('Generated %d pairs from %d samples'%(len(X2), len(X)))

        # fit the model
        self.estimator.fit(X2, y2)#, sample_weight=sw2)

        # get the mean of z for each level of y
        z = self.predict_z(X)
        self.z_means = np.array([z[y==cl].mean() for cl in self.label_encoder.classes_])

        return self

    def predict_z(self, X):
        z = self.estimator.decision_function(X)
        return z

    def predict_proba(self, X):
        z = self.predict_z(X)
        dists = -(z.reshape(-1,1) - self.z_means)**2
        yp = softmax(dists, axis=1)
        return yp

    def predict(self, X):
        yp = self.predict_proba(X)
        yp1d = self.label_encoder.inverse_transform(np.argmax(yp, axis=1))
        return yp1d

    def score(self, X, y):
        yp = self.predict(X)
        return kendalltau(y, yp)[0]


class MyXGBRanker(XGBRanker):
    """
    + convert to probability
    """
    def fit(self, X, y):
        # fit the model
        super().fit(X, y, [len(X)])

        # get the mean of z for each level of y
        self.label_encoder = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder.classes_
        z = super().predict(X).astype(float)
        self.z_means = np.array([z[y==cl].mean() for cl in self.label_encoder.classes_])
        return self
    
    def predict_z(self, X):
        z = super().predict(X).astype(float)
        return z
        
    def predict_proba(self, X):
        z = self.predict_z(X)
        dists = -(z.reshape(-1,1) - self.z_means)**2
        yp = softmax(dists, axis=1)
        return yp
    
    def predict(self, X):
        yp = self.predict_proba(X)
        yp1d = self.label_encoder.inverse_transform(np.argmax(yp, axis=1))
        return yp1d
    
