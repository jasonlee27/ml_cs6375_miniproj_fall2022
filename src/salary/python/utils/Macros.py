# This script is for defining all macros used in scripts in slpproject/python/hdlp/

from typing import *
from pathlib import Path

import os

class Macros:

    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__))) # {project_root}/src/salary/python/utils
    root_dir: Path = this_dir.parent.parent.parent.parent # {project_root}
    src_dir: Path = this_dir.parent.parent.parent # {project_root}/src
    python_dir: Path = this_dir.parent # {project_root}/src/salary/python
    data_dir: Path = root_dir / 'data' # {project_root}/data
    result_dir: Path = root_dir / "_results" # {project_root}/_results

    FMT_INT = "{:,d}"
    FMT_PER = "{:.1%}"
    FMT_FLOAT = "{:,.2f}"

    RAND_SEED = 27
