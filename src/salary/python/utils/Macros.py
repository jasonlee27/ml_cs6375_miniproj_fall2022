# This script is for defining all macros used in scripts in slpproject/python/hdlp/

from typing import *
from pathlib import Path

import os

class Macros:

    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__))) # {project_root}/src/salary/python/utils
    root_dir: Path = this_dir.parent.parent.parent.parent # {project_root}
    src_dir: Path = this_dir.parent.parent.parent # {project_root}/src
    python_dir: Path = this_dir.parent # {project_root}/src/salary/python
    data_dir: Path = root_dir / 'data' / 'salary' # {project_root}/data
    result_dir: Path = root_dir / "_results" # {project_root}/_results

    csv_file: Path = data_dir / 'final_data.csv'
    test_ratio = 0.2
    num_folds = 10
    
    FMT_INT = "{:,d}"
    FMT_PER = "{:.1%}"
    FMT_FLOAT = "{:,.2f}"

    RAND_SEED = 27

    FEATURES = [
        
        'title',
        'salary', # target to be predicted
        'lastName',
        'firstName',
        'citedby5y',
        'hindex5y',
        'i10index5y',
        'age',
        'school',
        'rating_class',
	    'total_ratings',
        'overall_rating',
        'total_courses',
        'average_grade',
        'percent_passing',
        'gender',
        'department'
    ]
    
