import csv
import numpy as np
import pandas as pd
import seaborn as sns

from typing import *
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, train_test_split

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

from ..utils.Macros import Macros
from ..utils.Utils import Utils


class Preprocess:

    @classmethod
    def get_raw_data(cls):
        df = pd.read_csv(Macros.csv_file, header=0)
        # return cls.fill_nan_with(df)
        return df

    # @classmethod
    # def fill_nan_with(cls, df):
    #     qs_in_data = list(df.keys())
    #     for q_i, q in enumerate(qs_in_data):
    #         if q==Macros.FEATURES[2] or \
    #            q==Macros.FEATURES[3] or \
    #            q==Macros.FEATURES[4] or \
    #            q==Macros.FEATURES[5] or \
    #            q==Macros.FEATURES[9]:
    #             df[q] = df[q].fillna('Other')
    #         elif q==Macros.FEATURES[11]:
    #             df[q] = df[q].fillna('0')
    #         # end if
    #     # end for
    #     return df

    @classmethod
    def get_data(cls, oversampling=False, undersampling=False):
        df = cls.get_raw_data()
        # label question: 'Which learning modality do you generally prefer?'
        labels = LabelEncoder().fit_transform(df[Macros.FEATURES[-1]])
        qs_in_data = list(df.keys())
        data = list()
        feat_labels = list()
        for q_i, q in enumerate(qs_in_data):
            if q!=Macros.FEATURES[-1]:
                df_q = LabelEncoder().fit_transform(df[q])
                data.append(df_q)
                feat_labels.append(q)
            # end if
        # end for
        data = np.transpose(np.array(data)) # (#examples, #feats)
        labels = np.array(labels) # (#examples, )

        if undersampling and not oversampling:
            tl = TomekLinks()
        elif not undersampling and oversampling:
            smote = SMOTE(random_state=Macros.RAND_SEED)
            # ros = RandomOverSampler(random_state=0)
        # end if

        # implment 5-folds (test_ratio is 0.2 and we choose 5-folds)
        kFold = KFold(n_splits=Macros.num_folds,
                      random_state=Macros.RAND_SEED,
                      shuffle=True)
        fold_i = 0

        for train_index, test_index in kFold.split(data):
            fold_i += 1
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            
            # x_train, x_test, y_train, y_test = train_test_split(
            #     data,
            #     labels,
            #     test_size=Macros.test_ratio,
            #     random_state=Macros.RAND_SEED
            # )
        
            if undersampling and not oversampling:
                x_train_resampled, y_train_resampled = tl.fit_resample(x_train, y_train)
                yield fold_i, x_train_resampled, x_test, y_train_resampled, y_test, feat_labels
            elif not undersampling and oversampling:
                # x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)
                x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
                yield fold_i, x_train_resampled, x_test, y_train_resampled, y_test, feat_labels
            else:
                yield fold_i, x_train, x_test, y_train, y_test, feat_labels
            # end if
        # end for
