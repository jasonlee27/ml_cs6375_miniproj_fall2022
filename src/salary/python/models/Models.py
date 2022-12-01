# this script is to create ml models
# model list:
#   gradient boosting regressor (sklearn)
#   catboost regressor
#   xgboost regressor
#   random forest regressor (sklearn)
#   extra trees regressor (sklearn)
#   kernel ridge regressor (sklearn)
#   decision tree regressor (sklearn)
#   ada boost regressor (sklearn)
#   k neighbors regressor (sklearn)
#   svm - linear kernel (sklearn)
#   linear regressor (sklearn)

from typing import *
from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils

# ==========
# models
from sklearn.ensemble import GradientBoostingRegressor, \
    RandomForestRegressor, \
    ExtraTreesRegressor, \
    AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import KernelRidge, \
    LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from catboost import CatBoostRegressor
import xgboost as xgb


class GradientboostRegressorClass:

    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = GradientBoostingRegressor()
        else:
            num_estimators = kwargs.pop('num_estimators', 100)
            validation_fraction = kwargs.pop('validation_fraction', 0.1)
            self.model = GradientBoostingRegressor(
                n_estimators=num_estimators,
                validation_fraction=validation_fraction
            )
        # end if
        
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)
    

class XgboostRegressorClass:

    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = xgb.XGBRegressor()
        else:
            num_estimators = kwargs.pop('num_estimators', 100)
            self.model = xgb.XGBRegressor(
                n_estimators=num_estimators
            )
        # end if
        
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)


class CatboostRegressorClass:
    
    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = CatBoostRegressor()
        else:
            num_iter = kwargs.pop('num_iter', 10)
            self.model = CatBoostRegressor(
                iterations=num_iter
            )
        # end if
        
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)


class RandomforestRegressorClass:
    
    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = RandomForestRegressor()
        else:
            max_depth = kwargs.pop('max_depth', 3)
            criterion = kwargs.pop('criterion', 'squared_error')
            n_estimators = kwargs.pop('num_estimators', 100)
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                criterion=criterion
            )
        # end if
        
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)


class ExtratreesRegressorClass:
    
    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = ExtraTreesRegressor()
        else:
            num_estimators = kwargs.pop('num_estimators', 100)
            max_depth = kwargs.pop('max_depth', 3)
            criterion = kwargs.pop('criterion', 'squared_error')
            self.model = ExtraTreesRegressor(
                n_estimators=num_estimators,
                max_depth=max_depth,
                criterion=criterion
            )
        # end if
        
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)


class KernelridgeRegressorClass:
    
    def __init__(self):        
        self.model = KernelRidge()
        
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    
class DtRegressorClass:

    def __init__(self):        
        self.model = DecisionTreeRegressor()
        
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)


class AdaboostRegressorClass:

    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = AdaBoostRegressor()
        else:
            num_estimators = kwargs.pop('num_estimators', 50)
            base_estimator = kwargs.pop('base_estimator', None)
            # if base_estimator is None, then it will be decision tree
            self.model = AdaBoostRegressor(
                base_estimator=base_estimator,
                n_estimators=num_estimators
            )
        # end if
        
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)


class KnnRegressorClass:

    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = KNeighborsRegressor()
        else:
            num_neighbors = kwargs.pop('num_neighbors', 5)
            self.model = KNeighborsRegressor(
                n_neighbors=num_neighbors
            )
        # end if
        
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)


class SvmRegressorClass:
    
    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = SVC()
        else:
            # kernel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} 
            kernel = kwargs.pop('kernel', 'rbf')
            self.model = SVR(
                kernel=kernel
            )
        # end if
        
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)


class LinearRegressorClass:

    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)
