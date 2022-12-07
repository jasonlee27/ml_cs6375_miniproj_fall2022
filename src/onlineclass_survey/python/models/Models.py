# this script is to create ml models
# model list:
#   gradient boosting classifier (sklearn)
#   catboost classifier
#   xgboost classifier
#   random forest classifier (sklearn)
#   extra trees classifier (sklearn)
#   Linear discriminant analysis (sklearn)
#   ridge classifier (sklearn)
#   decision tree classifier (sklearn)
#   naive bayes classifier (sklearn)
#   ada boost classifier (sklearn)
#   k neighbors classifier (sklearn)
#   svm - linear kernel (sklearn)
#   quadratic discriminant analysis (sklearn)
#   logistic regression classifier (sklearn)

from typing import *
from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils

# ==========
# models
from sklearn.ensemble import GradientBoostingClassifier, \
    RandomForestClassifier, \
    ExtraTreesClassifier, \
    AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
    QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, \
    LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from catboost import CatBoostClassifier
import xgboost as xgb


class GradientboostClassifierClass:

    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = GradientBoostingClassifier()
        else:
            num_estimators = kwargs.pop('num_estimators', 100)
            validation_fraction = kwargs.pop('validation_fraction', 0.1)
            self.model = GradientBoostingClassifier(
                n_estimators=num_estimators,
                validation_fraction=validation_fraction
            )
        # end if
        
    def train(self, x_train, y_train, sample_weight=None):
        self.model.fit(x_train, y_train, sample_weight=sample_weight)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    

class XgboostClassifierClass:

    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = xgb.XGBClassifier()
        else:
            num_estimators = kwargs.pop('num_estimators', 100)
            self.model = xgb.XGBClassifier(
                n_estimators=num_estimators
            )
        # end if
        
    def train(self, x_train, y_train, sample_weight=None):
        self.model.fit(x_train, y_train, sample_weight=sample_weight)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    

class CatboostClassifierClass:
    
    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = CatBoostClassifier()
        else:
            num_iter = kwargs.pop('num_iter', 10)
            self.model = CatBoostClassifier(
                iterations=num_iter
            )
        # end if
        
    def train(self, x_train, y_train, sample_weight=None):
        self.model.fit(x_train, y_train, sample_weight=sample_weight)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class RandomforestClassifierClass:
    
    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = RandomForestClassifier()
        else:
            max_depth = kwargs.pop('max_depth', 15)
            criterion = kwargs.pop('criterion', 'gini')
            self.model = RandomForestClassifier(
                max_depth=max_depth,
                criterion=criterion
            )
        # end if
        
    def train(self, x_train, y_train, sample_weight=None):
        self.model.fit(x_train, y_train, sample_weight=sample_weight)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    

class ExtratreesClassifierClass:
    
    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = ExtraTreesClassifier()
        else:
            num_estimators = kwargs.pop('num_estimators', 100)
            max_depth = kwargs.pop('max_depth', 3)
            criterion = kwargs.pop('criterion', 'gini')
            self.model = ExtraTreesClassifier(
                n_estimators=num_estimators,
                max_depth=max_depth,
                criterion=criterion
            )
        # end if
        
    def train(self, x_train, y_train, sample_weight=None):
        self.model.fit(x_train, y_train, sample_weight=sample_weight)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class LdaClassifierClass:
    def __init__(self):        
        self.model = LinearDiscriminantAnalysis()
        
    def train(self, x_train, y_train, sample_weight=None):
        # This model does not accept the sample weights
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class RidgeClassifierClass:
    
    def __init__(self):        
        self.model = RidgeClassifier()
        
    def train(self, x_train, y_train, sample_weight=None):
        self.model.fit(x_train, y_train, sample_weight=sample_weight)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    # def predict_proba(self, x_test):
    #     return self.model.predict_proba(x_test)

    
class DtClassifierClass:

    def __init__(self):        
        self.model = DecisionTreeClassifier()
        
    def train(self, x_train, y_train, sample_weight=None):
        self.model.fit(x_train, y_train, sample_weight=sample_weight)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class NbClassifierClass:

    # gaussian naive bayes
    def __init__(self):        
        self.model = GaussianNB()
        
    def train(self, x_train, y_train, sample_weight=None):
        self.model.fit(x_train, y_train, sample_weight=sample_weight)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class AdaboostClassifierClass:

    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = AdaBoostClassifier()
        else:
            num_estimators = kwargs.pop('num_estimators', 100)
            base_estimator = kwargs.pop('base_estimator', None)
            # if base_estimator is None, then it will be decision tree
            self.model = AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=num_estimators
            )
        # end if
        
    def train(self, x_train, y_train, sample_weight=None):
        self.model.fit(x_train, y_train, sample_weight=sample_weight)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class KnnClassifierClass:

    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = KNeighborsClassifier()
        else:
            num_neighbors = kwargs.pop('num_neighbors', 5)
            self.model = KNeighborsClassifier(
                n_neighbors=num_neighbors
            )
        # end if
        
    def train(self, x_train, y_train, sample_weight=None):
        # This model does not accept the sample weights
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class SvmClassifierClass:
    
    def __init__(self, kwargs=None):
        if kwargs is None:
            self.model = SVC()
        else:
            # kernel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} 
            kernel = kwargs.pop('kernel', 'rbf')
            self.model = SVC(
                kernel=kernel
            )
        # end if
        
    def train(self, x_train, y_train, sample_weight=None):
        self.model.fit(x_train, y_train, sample_weight=sample_weight)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class QdaClassifierClass:
    
    def __init__(self):
        self.model = QuadraticDiscriminantAnalysis()
        
    def train(self, x_train, y_train, sample_weight=None):
        # This model does not accept the sample weights
        self.model.fit(x_train, y_train)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class LrClassifierClass:

    def __init__(self):
        self.model = LogisticRegression(max_iter=200)
        
    def train(self, x_train, y_train, sample_weight=None):
        self.model.fit(x_train, y_train, sample_weight=sample_weight)
        return

    def test(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
