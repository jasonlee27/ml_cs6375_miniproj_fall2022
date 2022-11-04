import csv
import numpy as np
import pandas as pd
import seaborn as sns

from typing import *
from pathlib import Path

from ..utils.Macros import Macros
from ..utils.Utils import Utils



class Preprocess:

    @classmethod
    def get_name_ind_from_rmf(cls, name, rmp_data):
        for d_i in range(len(rmp_data)):
            first_name = '' if pd.isna(rmp_data.Fname[d_i]) is not None else rmp_data.Fname[d_i]
            last_name =  '' if pd.isna(rmp_data.Lname[d_i]) is not None else rmp_data.Lname[d_i]
            _name = f"{first_name} {last_name}"
            if name.lower()==_name.lower():
                return d_i
            # end if
        # end for
        return

    @classmethod
    def get_gender_from_name(cls, name, ntg_data):
        for d_i in range(len(ntg_data)):
            if name.lower()==ntg_data.name[d_i].lower():
                return ntg_data.likelyGender[d_i]
            # end if
        # end for
        return
        
    @classmethod
    def combine_data(cls):
        salary_data_dir = Macros.data_dir / 'salary'
        rmp_csv_file = salary_data_dir / 'ratemyprof.csv'
        sal_csv_file = salary_data_dir / 'salary_data_cleaned.csv'
        name_to_gender_csv_file = salary_data_dir / 'names_gender.csv'
        rmp_data = pd.read_csv(rmp_csv_file, header=0)
        sal_data = pd.read_csv(sal_csv_file, header=0)
        ntg_data = pd.read_csv(name_to_gender_csv_file, header=0)
        data_lod = list()
        for d_i in range(len(sal_data)):
            name = sal_data.Name[d_i]
            gender = cls.get_gender_from_name(name, ntg_data)
            name_ind = cls.get_name_ind_from_rmf(name, rmp_data)
            print(name, gender, name_ind)
            if name_ind is not None and gender is not None:
                dept = rmp_data.Dept[name_ind]
                rate_class = rmp_data.rating_class[name_ind]
                rate_tot = rmp_data.total_Ratings[name_ind]
                rate_overall = rmp_data.overall_rating[name_ind]
                salary = sal_data.Salary[d_i]
                data.append({
                    'fname': first_name,
                    'lname': last_name,
                    'gender': gender,
                    'rmp_rate_class': rate_class,
                    'rmp_rate_tot': rate_tot,
                    'rmp_rate_overall': rate_overall,
                    'salary': salary
                })
            # end if
        # end for
        data_dol = Utils.lod_to_dol(data_lod)

        res_file = salary_data_dir / 'data_combine.csv'
        with open(res_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(data_dol.keys())
            writer.writerows(zip(*data_dol.values()))
        # end with
        return
