# this script to train/test models from Models.py
# and it generates model evaluation by confusion matrix

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import *
from pathlib import Path
from sklearn.metrics import confusion_matrix

from ..utils.Macros import Macros
from ..utils.Utils import Utils
from ..preprocess.Preprocess import Preprocess

from .Models import GradientboostRegressorClass,\
    XgboostRegressorClass,\
    CatboostRegressorClass, \
    RandomforestRegressorClass,\
    ExtratreesRegressorClass,\
    KernelridgeRegressorClass,\
    DtRegressorClass,\
    AdaboostRegressorClass,\
    KnnRegressorClass,\
    SvmRegressorClass,\
    LinearRegressorClass


class RunModel:

    model_map = {
        'gdb': GradientboostRegressorClass,
        'xgb': XgboostRegressorClass,
        'catb': CatboostRegressorClass,
        'rdf': RandomforestRegressorClass,
        'extr': ExtratreesRegressorClass,
        'rdg': KernelridgeRegressorClass,
        'dt': DtRegressorClass,
        'adab': AdaboostRegressorClass,
        'knn': KnnRegressorClass,
        'svm': SvmRegressorClass,
        'lr': LinearRegressorClass
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
    def train_models(cls, model_dict: Dict, x_train, y_train):
        for model_name, model in model_dict.items():
            model.train(x_train, y_train)
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
                                res_over_folds: Dict):
        result_str = 'model_name,accuracy\n'
        for model_name, model in model_dict.items():
            acc = model.test(x_test, y_test)
            result_str += f"{model_name},{acc}\n"
                
            if model_name in res_over_folds.keys():
                res_over_folds[model_name].append(acc)
            else:
                res_over_folds[model_name] = [acc]
            # end if
        # end for

        save_dir = Macros.result_dir / 'salary'
        save_dir.mkdir(parents=True, exist_ok=True)
        Utils.write_txt(result_str, save_dir / f"test_accuracy_fold{fold_i}.csv")
        return res_over_folds

    @classmethod
    def get_scatter_plot(cls,
                         model_dict,
                         x_test,
                         y_test,
                         fold_i):
        figs_dir = Macros.result_dir / 'salary'
        figs_dir.mkdir(parents=True, exist_ok=True)
        for model_name, model in model_dict.items():
            data_lod = list()
            gts = list()
            y_preds = model.predict(x_test)
            sns.set_theme()
            for x_i in range(x_test.shape[0]):
                data_lod.append({
                    'x_ind': x_i,
                    'y_gt': y_test[x_i],
                    'y_pred': y_preds[x_i]
                })
                gts.append(y_test[x_i])
            # end for
            df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))
                
            # Plotting part
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.subplots()
            X_plot = np.linspace(0, 7, max(gts))
            Y_plot = X_plot
            p1 = sns.scatterplot(data=df,
                                 x='y_gt',
                                 y='y_pred',
                                 ax=ax,
                                 palette='Paired')
            p2 = sns.lineplot(data=df,
                              x='y_gt',
                              y='y_gt',
                              ax=ax,
                              color='g')
            ax.set_xlabel('ground truth')
            ax.set_ylabel('prediction')
            fig.tight_layout()
            fig.savefig(figs_dir / f"pred_{model_name}_scatterplot_fold{fold_i}.eps")
        # end for
        return
        
    
    @classmethod
    def get_feature_importance(cls,
                               model_dict: Dict,
                               feat_labels: List,
                               fold_i,
                               res_over_folds: Dict):
        sns.set_theme()
        figs_dir = Macros.result_dir / 'salary'
        figs_dir.mkdir(parents=True, exist_ok=True)
        for model_name, model in model_dict.items():
            # feature importance: ndarray of shape (n_features,)
            if hasattr(model.model, 'feature_importances_'):
                if model_name not in res_over_folds.keys():
                    res_over_folds[model_name] = dict()
                # end if
                feat_importances = model.model.feature_importances_
                data_lod = list()
                for f_i in range(len(feat_importances)):
                    data_lod.append({
                        'feat_id': f_i+1,
                        'feat_label': feat_labels[f_i],
                        'feat_importance': feat_importances[f_i]
                    })
                    if feat_labels[f_i] not in res_over_folds[model_name].keys():
                        res_over_folds[model_name][feat_labels[f_i]] = list()
                    # end if
                    res_over_folds[model_name][feat_labels[f_i]].append(feat_importances[f_i])
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
                fig.tight_layout()
                fig.savefig(figs_dir / f"feature_importance_{model_name}_barplot_fold{fold_i}.eps")
            elif hasattr(model.model, 'coef_'):
                if model_name not in res_over_folds.keys():
                    res_over_folds[model_name] = dict()
                # end if
                coefs = model.model.coef_
                # res = dict()
                for c_i in range(len(coefs)):
                    data_lod.append({
                        'feat_id': c_i+1,
                        'feat_label': feat_labels[c_i],
                        'feat_importance': coefs[c_i]
                    })
                    if feat_labels[c_i] not in res_over_folds[model_name].keys():
                        res_over_folds[model_name][feat_labels[c_i]] = list()
                    # end if
                    res_over_folds[model_name][feat_labels[c_i]].append(coefs[c_i])
                # end for
                
                df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))
                
                # Plotting part
                fig: plt.Figure = plt.figure()
                ax: plt.Axes = fig.subplots()
                # ax.tick_params(axis='x', rotation=45)
                ax = sns.barplot(data=df,
                                 x='feat_label',
                                 y='feat_importance',
                                 ax=ax,
                                 palette='Paired')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
                ax.set_xlabel('features')
                ax.set_ylabel('coef')
                fig.tight_layout()
                fig.savefig(figs_dir / f"coef_importance_{model_name}_barplot_fold{fold_i}.eps")
            else:
                print(f"{model_name} has no feature_importances/coefs attribute")
            # end if
        # end for
        return res_over_folds

    @classmethod
    def run_models(cls):
        # x_train, x_test, y_train, y_test, feat_labels = Preprocess.get_data()
        test_acc_over_folds = dict()
        feat_importance_over_folds = dict()
        for fold_i, x_train, x_test, y_train, y_test, feat_labels in Preprocess.get_data():
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
                }
            }
            model_dict = cls.get_models(model_config)
            cls.train_models(model_dict, x_train, y_train)
            test_acc_over_folds = cls.get_model_test_accuracy(model_dict,
                                                              x_test,
                                                              y_test,
                                                              fold_i,
                                                              test_acc_over_folds)
            # cls.get_confusion_matrices(model_dict, x_test, y_test)
            cls.get_scatter_plot(model_dict, x_test, y_test, fold_i)
            feat_importance_over_folds = cls.get_feature_importance(model_dict,
                                                                    feat_labels,
                                                                    fold_i,
                                                                    feat_importance_over_folds)
        # end for

        # Write the average results over the folds
        result_str = 'model_name,accuracy\n'
        for model_name, accs in test_acc_over_folds.items():
            acc = Utils.avg(accs)
            result_str += f"{model_name},{acc}\n"
        # end for
        save_dir = Macros.result_dir / 'salary'
        save_dir.mkdir(parents=True, exist_ok=True)
        Utils.write_txt(result_str, save_dir / f"test_accuracy_avg.csv")
        
        for model_name, feat_values in feat_importance_over_folds.items():
            data_lod = list()
            for feat_label, vals in feat_values.items():
                val = Utils.avg(vals)
                for val in vals:
                    data_lod.append({
                        'feat_label': feat_label,
                        'feat_importance': val
                    })
                # end for
            # end for

            df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))
                
            # Plotting part
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.subplots()
            # ax.tick_params(axis='x', rotation=45)
            ax = sns.barplot(data=df,
                             x='feat_label',
                             y='feat_importance',
                             estimator=np.mean,
                             ax=ax,
                             palette='Paired')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.set_xlabel('features')
            ax.set_ylabel('score')
            fig.tight_layout()
            figs_dir = Macros.result_dir / 'salary'
            figs_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(figs_dir / f"feat_importance_{model_name}_barplot_avg.eps")
        # end for
        return
    
