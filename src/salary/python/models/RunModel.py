# this script to train/test models from Models.py
# and it generates model evaluation by confusion matrix

import pandas as pd
import seaborn as sns

from typing import *
from pathlib import Path
from sklearn.metrics import confusion_matrix

from ..utils.Macros import Macros
from ..utils.Utils import Utils

from .Models import GradientboostClassifier,\
    XgboostClassifier,\
    CatboostClassifier

class RunModel:

    model_map = {
        'gdb': GradientboostClassifier,
        'xgb': XgboostClassifier,
        'catb': CatboostClassifier,
    }
    
    @classmethod
    def get_models(cls, model_name_args: Dict):
        model_dict = dict()
        for model_name, model_args in model_name_args.items():
            if model_args is None:
                model_dict[model_name]: cls.model_map[model_name]()
            else:
                model_dict[model_name]: cls.model_map[model_name](model_args)
            # end if
        # end for
        return model_dict

    @classmethod
    def train_models(cls, model_dict: Dict, x_train, y_train):
        for model_name, model in model_dict.items():
            model.train(x_train, y_train)
            model_dict[model_name] = model
        # end for
        return

    @classmethod
    def get_model_test_accuracy(cls, model_dict: Dict, x_test, y_test):
        result_str = 'model_name,accuracy\n'
        for model_name, model in model_dict.items():
            acc = model.test_models(x_train, y_train)
            result_str += f"{model_name},{acc}\n"
        # end for

        # TODO: save result here?
        save_dir = Macros.result_dir / 'salary'
        save_dir.mkdir(parents=True, exist_ok=True)
        Utils.write_txt(result_str, save_dir / 'test_accuracy.csv')
        return

    @classmethod
    def get_confusion_matrices(cls, model_dict: Dict, x_test, y_test):
        result_str = 'model_name,tn,fp,fn,tp\n'
        for model_name, model in model_dict.items():
            # Compute the test error
            y_pred = [predict_example(x, decision_tree) for x in Xtst]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            result_str += f"{model_name},{tn},{fp},{fn},{tp}\n"
        # end for
        save_dir = Macros.result_dir / 'salary'
        save_dir.mkdir(parents=True, exist_ok=True)
        Utils.write_txt(result_str, save_dir / 'confusion_matrix.csv')
        return

    @classmethod
    def get_feature_importance(cls, model_dict: Dict):
        sns.set_theme()
        for model_name, model in model_dict.items():
            # feature importance: ndarray of shape (n_features,)
            feat_importances = model.feature_importances_
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
        # end for
        return

    
