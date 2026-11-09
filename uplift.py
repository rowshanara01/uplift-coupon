"""Uplift (causal) models: S-learner, T-learner, X-learner + naive response model (ja bhul)"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
from sklearn.ensemble import HistGradientBoostingRegressor as HGR


def _clf():
    return HGC(max_iter=150, learning_rate=0.07, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)


def _reg():
    return HGR(max_iter=120, learning_rate=0.07, max_leaf_nodes=10, l2_regularization=1.0, random_state=0)


def _p(m, X):
    return m.predict_proba(X)[:, 1]


class SLearner:
    """Ekta model, treatment ke feature hishebe dei. uplift = f(x, coupon) - f(x, no coupon)"""
    def fit(self, X, t, y):
        self.m = _clf().fit(np.column_stack([X, t]), y)
        return self

    def baseline(self, X):
        return _p(self.m, np.column_stack([X, np.zeros(len(X))]))

    def uplift(self, X):
        return _p(self.m, np.column_stack([X, np.ones(len(X))])) - self.baseline(X)


class TLearner:
    """Duto model: coupon pawa group ar na-pawa group alada shikhe. uplift = f1(x) - f0(x)"""
    def fit(self, X, t, y):
        self.m0 = _clf().fit(X[t == 0], y[t == 0])
        self.m1 = _clf().fit(X[t == 1], y[t == 1])
        return self

    def baseline(self, X):
        return _p(self.m0, X)

    def uplift(self, X):
        return _p(self.m1, X) - _p(self.m0, X)


class XLearner:
    """T-learner er upor: prottek customer er 'imputed effect' ber kore tar upor abar shikhe"""
    def __init__(self, e=0.5):
        self.e = e  # coupon dewar probability (RCT te 0.5)

    def fit(self, X, t, y):
        X0, X1, y0, y1 = X[t == 0], X[t == 1], y[t == 0], y[t == 1]
        self.m0 = _clf().fit(X0, y0)
        self.m1 = _clf().fit(X1, y1)
        d1 = y1 - _p(self.m0, X1)       # treated der jonno: ja holo - coupon chhara ja hoto
        d0 = _p(self.m1, X0) - y0       # control der jonno: coupon dile ja hoto - ja holo
        self.t1 = _reg().fit(X1, d1)
        self.t0 = _reg().fit(X0, d0)
        return self

    def baseline(self, X):
        return _p(self.m0, X)

    def uplift(self, X):
        return self.e * self.t0.predict(X) + (1 - self.e) * self.t1.predict(X)


class ResponseModel:
    """NAIVE: 'ke kinbe' predict kore, 'coupon dile ke ALADA kore kinbe' na. Prai company ei bhul kore."""
    def fit(self, X, t, y):
        self.m = _clf().fit(X[t == 1], y[t == 1])
        return self

    def baseline(self, X):
        return _p(self.m, X)

    def uplift(self, X):   # asole uplift na, shudhu kenar sombhabona (ranking er jonno)
        return _p(self.m, X)
