# this script to train/test models from Models.py
# and it generates model evaluation by confusion matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import *
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..preprocess.Preprocess import Preprocess

from .Models import GradientboostClassifierClass,\
    XgboostClassifierClass,\
    CatboostClassifierClass, \
    RandomforestClassifierClass,\
    ExtratreesClassifierClass,\
    LdaClassifierClass,\
    RidgeClassifierClass,\
    DtClassifierClass,\
    NbClassifierClass,\
    AdaboostClassifierClass,\
    KnnClassifierClass,\
    SvmClassifierClass,\
    QdaClassifierClass,\
    LrClassifierClass

# LgboostClassifier

class RunModel:

    model_map = {
        'gdb': GradientboostClassifierClass,
        'xgb': XgboostClassifierClass,
        'catb': CatboostClassifierClass,
        # 'lgb': LgboostClassifierClass,
        'rdf': RandomforestClassifierClass,
        'extr': ExtratreesClassifierClass,
        'lda': LdaClassifierClass,
        'rdg': RidgeClassifierClass,
        'dt': DtClassifierClass,
        'nb': NbClassifierClass,
        'adab': AdaboostClassifierClass,
        'knn': KnnClassifierClass,
        'svm': SvmClassifierClass,
        'qda': QdaClassifierClass,
        'lr': LrClassifierClass
    }
    
    @classmethod
    def get_models(cls, model_name_args: Dict):
        model_dict = dict()
        for model_name, model in cls.model_map.items():
            if model_name in model_name_args.keys():
                model_obj = model(model_name_args[model_name])
            else:
                model_obj = model()
            # end if
            model_dict[model_name] = model_obj
        # end for
        return model_dict

    @classmethod
    def get_sample_weight(cls, y_train):
        return compute_sample_weight(class_weight='balanced', y=y_train)
            

    @classmethod
    def train_models(cls, model_dict: Dict, x_train, y_train, sample_weight=None):
        for model_name, model in model_dict.items():
            model.train(x_train, y_train, sample_weight=sample_weight)
            model_dict[model_name] = model
            print(f"{model_name} model trained.")
        # end for
        return

    @classmethod
    def get_model_test_accuracy(cls, model_dict: Dict, x_test, y_test):
        result_str = 'model_name,accuracy\n'
        for model_name, model in model_dict.items():
            acc = model.test(x_test, y_test)
            result_str += f"{model_name},{acc}\n"
        # end for

        # TODO: save result here?
        save_dir = Macros.result_dir / 'onlineclass_survey'
        save_dir.mkdir(parents=True, exist_ok=True)
        Utils.write_txt(result_str, save_dir / 'test_accuracy.csv')
        return

    @classmethod
    def get_confusion_matrices(cls, model_dict: Dict, x_test, y_test):
        result_str = 'model_name,tn,fp,fn,tp\n'
        for model_name, model in model_dict.items():
            # Compute the test error
            y_pred = model.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            result_str += f"{model_name},{tn},{fp},{fn},{tp}\n"
        # end for
        save_dir = Macros.result_dir / 'onlineclass_survey'
        save_dir.mkdir(parents=True, exist_ok=True)
        Utils.write_txt(result_str, save_dir / 'confusion_matrix.csv')
        return

    @classmethod
    def get_feature_importance(cls, model_dict: Dict):
        sns.set_theme()
        figs_dir = Macros.result_dir / 'onlineclass_survey'
        figs_dir.mkdir(parents=True, exist_ok=True)
        for model_name, model in model_dict.items():
            # feature importance: ndarray of shape (n_features,)
            if hasattr(model.model, 'feature_importances_'):
                feat_importances = model.model.feature_importances_
                data_lod = list()
                for f_i in range(len(feat_importances)):
                    data_lod.append({
                        'feat_id': f_i+1,
                        'feat_importance': feat_importances[f_i]
                    })
                # end for
                df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))
                
                # Plotting part
                fig: plt.Figure = plt.figure()
                ax: plt.Axes = fig.subplots()
                ax = sns.barplot(data=df,
                                 x='feat_id',
                                 y='feat_importance',
                                 ax=ax,
                                 palette='Paired')
                ax.set_xlabel('features')
                ax.set_ylabel('feat-y')
                fig.tight_layout()
                fig.savefig(figs_dir / f"feature_importance_{model_name}_barplot.eps")
            elif hasattr(model.model, 'coef_'):
                coefs = model.model.coef_[0,:]
                for c_i in range(len(coefs)):
                    data_lod.append({
                        'feat_id': c_i+1,
                        'feat_importance': coefs[c_i]
                    })
                # end for
                df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))
                
                # Plotting part
                fig: plt.Figure = plt.figure()
                ax: plt.Axes = fig.subplots()
                ax = sns.barplot(data=df,
                                 x='feat_id',
                                 y='feat_importance',
                                 ax=ax,
                                 palette='Paired')
                ax.set_xlabel('features')
                ax.set_ylabel('coef-y')
                fig.tight_layout()
                fig.savefig(figs_dir / f"coef_importance_{model_name}_barplot.eps")
            else:
                print(f"{model_name} has no feature_importances/coefs attribute")
            # end if
        # end for
        return

    @classmethod
    def run_models(cls):
        x_train, x_test, y_train, y_test = Preprocess.get_data()
        sample_weight = cls.get_sample_weight(y_train)
        
        model_config = {
            'gdb': {
                'num_estimators': 100,
                'validation_fraction': 0.1
            },
            'xgb': {
                'num_estimators': 100,
            },
            'catb': {
                'num_iter': 10
            }
        }
        
        model_dict = cls.get_models(model_config)
        cls.train_models(model_dict, x_train, y_train, sample_weight=sample_weight)
        cls.get_model_test_accuracy(model_dict, x_test, y_test)
        cls.get_confusion_matrices(model_dict, x_test, y_test)
        cls.get_feature_importance(model_dict)        
        return
    
