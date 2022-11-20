from typing import *
from pathlib import Path

import re, os
import sys
import json

from .Macros import Macros


class Utils:
    
    @classmethod
    def read_txt(cls, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        #end with
        return lines

    @classmethod
    def write_txt(cls, input_str, data_file):
        with open(data_file, 'w') as f:
            lines = f.write(input_str)
        #end with
        return lines

    @classmethod
    def write_pkl(cls, results, pkl_file):
        with open(pkl_file, 'wb+') as f:
            pickle.dump(results, f)
        # end with
        return
    
    @classmethod
    def read_json(cls, json_file):
        # read cfg json file
        if os.path.exists(str(json_file)):
            with open(json_file, 'r') as f:
                return json.load(f)
            # end with
        # end if
        return
    
    @classmethod
    def write_json(cls, input_dict, json_file, pretty_format=False):
        with open(json_file, 'w') as f:
            if pretty_format:
                json.dump(input_dict, f, indent=4)
            else:
                json.dump(input_dict, f)
            # end if
        # end with
        return

    @classmethod
    def lod_to_dol(cls, list_of_dict: List[dict]) -> Dict[Any, List]:
        """
        Converts a list of dict to a dict of list.
        """
        keys = set.union(*[set(d.keys()) for d in list_of_dict])
        return {k: [d.get(k) for d in list_of_dict] for k in keys}
