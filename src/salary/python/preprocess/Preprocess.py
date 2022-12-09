import csv
import numpy as np
import pandas as pd
import seaborn as sns

from typing import *
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import normalize, MinMaxScaler

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
        #labels = df[Macros.FEATURES[1]]

        # Ordinal Encoding Categorical Features (Except School/Department, handled below)
        title_mapper = {"Professor":6, "Associate Professor":5, "Assistant Professor":4,
                "Non Tenure Professor":3, "Non Tenure Associate Professor":2, 
                "Non Tenure Assistant Professor":1, "Senior Lecturer I":0}

        df["title"] = df["title"].replace(title_mapper)

        gender_mapper = {"male":1, "female":0}
        df["gender"] = df["gender"].replace(gender_mapper)

        rating_mapper = {"zero":0, "poor":1, "average":2, "good":3}
        df["rating_class"] = df["rating_class"].replace(rating_mapper)

        # One Hot Encoding School Data
        df = pd.concat([df, pd.get_dummies(df["school"], prefix='school')], axis=1)

        # Trying to replace with median instead of zero, doesn't seem to work as well
        # df["citedby5y"] = df["citedby5y"].replace(0,df[df["citedby5y"] != 0]["citedby5y"].median())
        # df["hindex5y"] = df["hindex5y"].replace(0,df[df["hindex5y"] != 0]["hindex5y"].median())
        # df["i10index5y"] = df["i10index5y"].replace(0,df[df["i10index5y"] != 0]["i10index5y"].median())
        
        # Log Transform of Citation Data
        df["citedby5y"] = np.log(df["citedby5y"] + 1)
        df["hindex5y"] = np.log(df["hindex5y"] + 1)
        df["i10index5y"] = np.log(df["i10index5y"] + 1)


        # MinMax Scaling 
        columns_to_normalize = ["total_ratings","age", "overall_rating","total_courses","average_grade","percent_passing"]

        sklearn_normalizers = {}

        for name in columns_to_normalize:
            sklearn_normalizers[name] = MinMaxScaler()
            df[name] = sklearn_normalizers[name].fit_transform(df[[name]])

        # Dropping Exteraneous Features 
        # df = df[['salary',
        #          'title',
        #         'citedby5y',
        #         'hindex5y',
        #         'i10index5y',
        #         'age',
        #         'rating_class',
        #         'total_ratings',
        #         'overall_rating',
        #         'total_courses',
        #         'average_grade',
        #         'percent_passing',
        #         'gender',
        #         'department',
        #         'school_BBS',
        #         'school_JSOM',
        #         'school_NSM',
        #         'school_IS',
        #         'school_AHT',
        #         'school_EPPS',
        #         'school_ECS']]



        df = df[['salary',
                 'title',
                'citedby5y',
                'i10index5y',
                'age',
                'total_ratings',
                'overall_rating',
                'total_courses',
                'average_grade',
                'department',
                'school_JSOM',
                'percent_passing']]


        # qs_in_data = list(df.keys())[1:]
        # data = list()
        # feat_labels = list()
        # for q_i, q in enumerate(qs_in_data):
        #     if q!=Macros.FEATURES[1]:
        #         if q in [
        #                 Macros.FEATURES[0],
        #                 Macros.FEATURES[2],
        #                 Macros.FEATURES[3],
        #                 Macros.FEATURES[8],
        #                 Macros.FEATURES[9],
        #                 Macros.FEATURES[-2],
        #                 Macros.FEATURES[-1]]:
        #             df_q = LabelEncoder().fit_transform(df[q])
        #             data.append(df_q)
        #         else:
        #             df_q = normalize([df[q]]).reshape(-1)
        #             data.append(df_q)
        #         # end if
        #         feat_labels.append(q)
        #     # end if
        # # end for

        # data = np.transpose(np.array(data)) # (#examples, #feats)
        # labels = np.array(labels) # (#examples, )


        # Splitting Data
        y = df["salary"]
        X = df.drop(["salary"], axis=1)

        x_department = pd.concat([X["department"], y], axis=1)
            
        # Replaces department with average salary of department
        # Putting this at the end to avoid data leakage
        department_values = x_department.groupby(['department'])["salary"].mean().values
        department_values = [(x - min(department_values))/(max(department_values) - min(department_values)) for x in department_values]
        department_keys = x_department.groupby(['department'])["salary"].mean().keys()
        department_mapper = dict(map(lambda i,j : (i,j) , department_keys,department_values))
        X["department"] = X["department"].replace(department_mapper)
        feat_labels = list(X.columns)
        X = X.to_numpy()
        y = y.to_numpy()
        
        # implment 5-folds (test_ratio is 0.2 and we choose 5-folds)
        kFold = KFold(n_splits=Macros.num_folds,
                      random_state=Macros.RAND_SEED,
                      shuffle=True)
        fold_i = 0
        for train_index, test_index in kFold.split(X):
            fold_i += 1
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # return x_train, x_test, y_train, y_test, feat_labels
            yield fold_i, x_train, x_test, y_train, y_test, feat_labels
        # end for
        
