# this script is to create ml models
# model list:
#   gradient boosting classifier (sklearn)
#   catboost classfier
#   light gradient boosting machine
#   xgboost
#   random forest classifier (sklearn)
#   extra trees classifier (sklearn)
#   Linear discriminant analysis (sklearn)
#   ridge classifier (sklearn)
#   decision tree classidier (sklearn)
#   naive bayes classifier (sklearn)
#   ada boost classifier (sklearn)
#   k neighbors classifier (sklearn)
#   svm - linear kernel (sklearn)
#   quadratic discriminant analysis (sklearn)
#   logistic regression

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

from catboost import CatBoostClassifier as catb
import xgboost as xgb
import lightgbm as lgb

    
class GradientboostClassifier:

    def __init__(self,
                 num_estimators=100,
                 validation_fraction=0.1):
        self.model = GradientBoostingClassifier(
            n_estimators=num_estimators,
            validation_fraction=validation_fraction
        )
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    

class XgboostClassifier:

    def __init__(self, num_estimators=100):
        self.model = xgb.XGBClassifier(
            n_estimators=num_estimators
        )
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    

class CatboostClassifier:
    
    def __init__(self, num_iter=10):
        self.model = catb.CatBoostClassifier(
            iterations=num_iter
        )
        
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)

    
class LgboostClassifier:

    def __init__(self, num_iter=10):
        self.model = lgb.LGBMClassifier()
        
    def train_models(self, x_train, y_train, validation_fraction):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test):

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    

class RandomforestClassifier:
    
    def __init__(self,
                 max_depth=3,
                 criterion='gini'):        
        self.model = RandomForestClassifier(
            max_depth=max_depth,
            criterion=criterion
        )
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
    

class ExtratreesClassifier:
    
    def __init__(self,
                 num_estimators=100,
                 max_depth=3,
                 criterion='gini'):        
        self.model = ExtraTreesClassifier(
            n_estimators=num_estimators,
            max_depth=max_depth,
            criterion=criterion
        )
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class LdaClassifier:
    def __init__(self):        
        self.model = LinearDiscriminantAnalysis()
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class RidgeClassifier:
    
    def __init__(self):        
        self.model = RidgeClassifier()
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    # def predict_proba(self, x_test):
    #     return self.model.predict_proba(x_test)

    
class DtClassifier:

    def __init__(self):        
        self.model = DecisionTreeClassifier()
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class NbClassifier:

    # gaussian naive bayes
    def __init__(self):        
        self.model = GaussianNB()
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class AdaboostClassifier:

    def __init__(self,
                 base_estimator=None,
                 num_estimators=100):
        # if base_estimator is None, then it will be decision tree
        self.model = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=num_estimators
        )
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class KnnClassifier:

    def __init__(self,
                 num_neighbors=5):
        self.model = KNeighborsClassifier(
            n_neighbors=num_neighbors
        )
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class SvmClassifier:
    
    def __init__(self,
                 kernel='rbf'):
        # kernel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} 
        self.model = SVC(
            kernel=kernel
        )
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class QdaClassifier:
    
    def __init__(self):
        self.model = QuadraticDiscriminantAnalysis()
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)


class LrClassifier:

    def __init__(self):
        self.model = LogisticRegression()
        
    def train_models(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        return

    def test_models(self, x_test, y_test):
        # returns the mean accuracy on the given test data and labels.
        return self.model.score(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def predict_proba(self, x_test):
        return self.model.predict_proba(x_test)
