import csv
import numpy as np
import pandas as pd
import seaborn as sns

from typing import *
from pathlib import Path
from sklearn.model_selection import train_test_split

from ..utils.Macros import Macros
from ..utils.Utils import Utils


class Preprocess:

    @classmethod
    def get_raw_data(cls):
        data = pd.read_csv(Macros.csv_file, header=0)
        return data


    @classmethod
    def get_data(cls):
        raw_data = cls.get_raw_data()
        labels = raw_data[Macros.QUESTIONS[-2]] # 'Which learning modality do you generally prefer?'
        qs_in_data = list(raw_data.keys())
        data = list()
        for q in qs_in_data:
            if q not in [
                    Macros.QUESTIONS[0],
                    Macros.QUESTIONS[-1],
                    Macros.QUESTIONS[-2]
            ]:
                data.append(raw_data[q])
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
