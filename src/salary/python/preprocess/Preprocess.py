import csv
import numpy as np
import pandas as pd
import seaborn as sns

from typing import *
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import normalize

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
    #         if q==Macros.FEATURES[12]:
    #             df[q] = df[q].fillna('na')
    #         # end if
    #     # end for
    #     return df

    @classmethod
    def get_data(cls):
        df = cls.get_raw_data()
        # label feature: 'salary'
        labels = df[Macros.FEATURES[1]]
        qs_in_data = list(df.keys())
        data = list()
        for q_i, q in enumerate(qs_in_data):
            if q!=Macros.FEATURES[1]:
                if q in [
                        Macros.FEATURES[0],
                        Macros.FEATURES[2],
                        Macros.FEATURES[3],
                        Macros.FEATURES[8],
                        Macros.FEATURES[9],
                        Macros.FEATURES[-2],
                        Macros.FEATURES[-1]]:
                    df_q = LabelEncoder().fit_transform(df[q])
                    data.append(df_q)
                else:
                    df_q = normalize([df[q]]).reshape(-1)
                    data.append(df_q)
                # end if
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
