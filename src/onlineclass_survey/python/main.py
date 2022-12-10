
import re, os
import sys
import json
import random
import argparse

from typing import *
from pathlib import Path

from .utils.Macros import Macros
from .utils.Utils import Utils

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--run', type=str, required=True,
                    choices=['preprocess', 'run_model', 'make_figs'], help='task to be run')
parser.add_argument('--oversampling', action='store_true')
parser.add_argument('--undersampling', action='store_true')
# ==========
# TODO: arguments
args = parser.parse_args()

rand_seed_num = Macros.RAND_SEED
random.seed(rand_seed_num)

def run_preprocess():
    from .preprocess.Preprocess import Preprocess
    Preprocess.get_data()
    return

def run_models():
    from .models.RunModel import RunModel
    RunModel.run_models(oversampling=args.oversampling,
                        undersampling=args.undersampling)
    return

def run_make_figs():
    from .models.RunModel import RunModel
    RunModel.get_res_over_configs()
    return


func_map = {
    'preprocess': run_preprocess,
    'run_model': run_models,
    'make_figs': run_make_figs
}

if __name__=="__main__":
    func_map[args.run]()
