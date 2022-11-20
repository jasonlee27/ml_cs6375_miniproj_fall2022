import csv
import numpy as np
import pandas as pd
import seaborn as sns

from typing import *
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from ..utils.Macros import Macros
from ..utils.Utils import Utils


class Preprocess:

    @classmethod
    def get_raw_data(cls):
        df = pd.read_csv(Macros.csv_file, header=0)
        return cls.fill_nan_with(df)

    @classmethod
    def fill_nan_with(cls, df):
        qs_in_data = list(df.keys())
        for q_i, q in enumerate(qs_in_data):
            if q==Macros.QUESTIONS[2] or \
               q==Macros.QUESTIONS[3] or \
               q==Macros.QUESTIONS[4] or \
               q==Macros.QUESTIONS[5] or \
               q==Macros.QUESTIONS[9]:
                df[q] = df[q].fillna('Other')
            elif q==Macros.QUESTIONS[11]:
                df[q] = df[q].fillna('0')
            # end if
        # end for
        return df

    @classmethod
    def get_data(cls):
        df = cls.get_raw_data()
        # label question: 'Which learning modality do you generally prefer?'
        labels = LabelEncoder().fit_transform(df[Macros.QUESTIONS[-2]])
        qs_in_data = list(df.keys())
        data = list()
        for q_i, q in enumerate(qs_in_data):
            if q not in [
                    Macros.QUESTIONS[0],
                    Macros.QUESTIONS[-1],
                    Macros.QUESTIONS[-2]
            ]:
                df_q = LabelEncoder().fit_transform(df[q])                
                data.append(df_q)
            # end if
        # end for
        data = np.transpose(np.array(data)) # (#examples, #feats)
        labels = np.array(labels) # (#examples, )
        x_train, x_test, y_train, y_test = train_test_split(
            data,
            labels,
            test_size=Macros.test_ratio,
            random_state=Macros.RAND_SEED
        )
        return x_train, x_test, y_train, y_test