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
        result_str = ''
        for model_name, model in model_dict.items():
            acc = model.test_models(x_train, y_train)
            result_str += f"{model_name}\t{acc}\n"
        # end for

        # TODO: save result here?
        return result_str

    @classmethod
    def get_confusion_matrices(cls, model_dict: Dict, x_test, y_test):
        result_str = ''
        for model_name, model in model_dict.items():
            # Compute the test error
            y_pred = [predict_example(x, decision_tree) for x in Xtst]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            result_str += f"decision_tree::depth_{d}::confusion_matrix::tn, fp, fn, tp = {tn}, {fp}, {fn}, {tp}\n"
        # end for
        return result_str

    @classmethod
    def get_feature_importance(cls, model_dict: Dict):
        # TODO
        st = time.time()
        for model_name, model in model_dict.items():
            importances = model.feature_importances_


            df: pd.DataFrame = pd.DataFrame.from_dict(Utils.lod_to_dol(data_lod))

            # Plotting part
            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.subplots()
            ax = sns.scatterplot(data=df,
                                 x='feat_x',
                                 y='feat_y',
                                 hue=group_by,
                                 ax=ax,
                                 palette='Paired')
            ax.set_xlabel('feat-x')
            ax.set_ylabel('feat-y')
            
            forest_importances = pd.Series(importances, index=feature_names)
            
            fig, ax = plt.subplots()
            forest_importances.plot.bar(yerr=std, ax=ax)
            ax.set_title("Feature importances using MDI")
            ax.set_ylabel("Mean decrease in impurity")
            fig.tight_layout()
        # end for
        return

    
