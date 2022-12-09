# this script to train/test models from Models.py
# and it generates model evaluation by confusion matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import *
from pathlib import Path
from sklearn.metrics import f1_score
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
    def train_models(cls,
                     model_dict: Dict,
                     x_train,
                     y_train,
                     sample_weight=None,
                     x_test=None,
                     y_test=None):
        for model_name, model in model_dict.items():
            model.train(x_train, y_train, sample_weight=sample_weight)
            if x_test is not None and y_test is not None:
                acc = model.test(x_test, y_test)
                y_pred = model.predict(x_test)
                f1 = f1_score(y_test, y_pred)
                print(f"{model_name},{acc},{f1}\n")
            # end if
            model_dict[model_name] = model
            print(f"{model_name} model trained.")
        # end for
        return

    @classmethod
    def get_model_test_accuracy(cls,
                                model_dict: Dict,
                                x_test,
                                y_test,
                                fold_i,
                                oversampling=False,
                                undersampling=False):
        result_str = 'model_name,accuracy,f1_score\n'
        for model_name, model in model_dict.items():
            acc = model.test(x_test, y_test)
            y_pred = model.predict(x_test)
            f1 = f1_score(y_test, y_pred)
            result_str += f"{model_name},{acc},{f1}\n"
        # end for

        # TODO: save result here?
        save_dir = Macros.result_dir / 'onlineclass_survey'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / 'test_accuracy.csv'
        if oversampling:
            save_file = save_dir / 'test_accuracy_os_fold{fold_i}.csv'
        elif undersampling:
            save_file = save_dir / 'test_accuracy_us_fold{fold_i}.csv'
        # end if
        Utils.write_txt(result_str, save_file)
        return

    @classmethod
    def get_confusion_matrices(cls,
                               model_dict: Dict,
                               x_test,
                               y_test,
                               fold_i,
                               oversampling=False,
                               undersampling=False):
        result_str = 'model_name,tn,fp,fn,tp\n'
        for model_name, model in model_dict.items():
            # Compute the test error
            y_pred = model.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            result_str += f"{model_name},{tn},{fp},{fn},{tp}\n"
        # end for
        save_dir = Macros.result_dir / 'onlineclass_survey'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / 'confusion_matrix.csv'
        if oversampling:
            save_file = save_dir / 'confusion_matrix_os_fold{fold_i}.csv'
        elif undersampling:
            save_file = save_dir / 'confusion_matrix_us_fold{fold_i}.csv'
        # end if
        Utils.write_txt(result_str, save_file)
        return

    @classmethod
    def get_feature_importance(cls,
                               model_dict: Dict,
                               feat_labels: List,
                               fold_i,
                               oversampling=False,
                               undersampling=False):
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
                        'feat_label': feat_labels[f_i],
                        'feat_importance': feat_importances[f_i]
                    })
                # end for
                df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))
                
                # Plotting part
                fig: plt.Figure = plt.figure()
                ax: plt.Axes = fig.subplots()
                ax = sns.barplot(data=df,
                                 x='feat_label',
                                 y='feat_importance',
                                 ax=ax,
                                 palette='Paired')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
                ax.set_xlabel('features')
                ax.set_ylabel('importance')
                fig_file = figs_dir / f"feature_importance_{model_name}_barplot_fold{fold_i}.eps"
                if oversampling:
                    fig_file = figs_dir / f"feature_importance_{model_name}_os_barplot_fold{fold_i}.eps"
                elif undersampling:
                    fig_file = figs_dir / f"feature_importance_{model_name}_us_barplot_fold{fold_i}.eps"
                # end if
                fig.tight_layout()
                fig.savefig(fig_file)
            elif hasattr(model.model, 'coef_'):
                coefs = model.model.coef_[0,:]
                for c_i in range(len(coefs)):
                    data_lod.append({
                        'feat_id': c_i+1,
                        'feat_label': feat_labels[f_i],
                        'feat_importance': coefs[c_i]
                    })
                # end for
                df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))
                
                # Plotting part
                fig: plt.Figure = plt.figure()
                ax: plt.Axes = fig.subplots()
                ax = sns.barplot(data=df,
                                 x='feat_label',
                                 y='feat_importance',
                                 ax=ax,
                                 palette='Paired')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
                ax.set_xlabel('features')
                ax.set_ylabel('coef-y')
                fig_file = figs_dir / f"coef_importance_{model_name}_barplot_fold{fold_i}.eps"
                if oversampling:
                    fig_file = figs_dir / f"coef_importance_{model_name}_os_barplot_fold{fold_i}.eps"
                elif undersampling:
                    fig_file = figs_dir / f"coef_importance_{model_name}_us_barplot_fold{fold_i}.eps"
                # end if
                fig.tight_layout()
                fig.savefig(fig_file)
            else:
                print(f"{model_name} has no feature_importances/coefs attribute")
            # end if
        # end for
        return

    @classmethod
    def run_models(cls, oversampling=False, undersampling=False):
        for fold_i, x_train, x_test, y_train, y_test, feat_labels in Preprocess.get_data(
                oversampling=oversampling,
                undersampling=undersampling):
            print(f"FOLD: {fold_i} out of {Macros.num_folds}")
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
                },
                'rdf': {
                    'max_depth': 15
                }
            }
            
            sample_weight = None
            if not oversampling and not undersampling:
                sample_weight = cls.get_sample_weight(y_train)
            # end if
            
            model_dict = cls.get_models(model_config)
            cls.train_models(model_dict,
                             x_train,
                             y_train,
                             sample_weight=sample_weight)
            cls.get_model_test_accuracy(model_dict,
                                        x_test,
                                        y_test,
                                        fold_i,
                                        oversampling=oversampling,
                                        undersampling=undersampling)
            cls.get_confusion_matrices(model_dict,
                                       x_test,
                                       y_test,
                                       fold_i,
                                       oversampling=oversampling,
                                       undersampling=undersampling)
            cls.get_feature_importance(model_dict,
                                       feat_labels,
                                       fold_i,
                                       oversampling=oversampling,
                                       undersampling=undersampling)
        # end for
        return
    
