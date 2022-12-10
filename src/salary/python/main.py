
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
                    choices=['run_model', 'make_figs'], help='task to be run')
# ==========
# TODO: arguments
args = parser.parse_args()

rand_seed_num = Macros.RAND_SEED
random.seed(rand_seed_num)


def run_models():
    from .models.RunModel import RunModel
    RunModel.run_models()
    return

def run_make_figs():
    from .models.RunModel import RunModel
    RunModel.get_res_over_configs()
    return


func_map = {
    'run_model': run_models,
    'res_fig': run_make_figs
}

if __name__=="__main__":
    func_map[args.run]()
