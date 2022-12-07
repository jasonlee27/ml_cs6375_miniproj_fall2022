# This script is for defining all macros used in scripts in slpproject/python/hdlp/

from typing import *
from pathlib import Path

import os

class Macros:

    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__))) # {project_root}/src/salary/python/utils
    root_dir: Path = this_dir.parent.parent.parent.parent # {project_root}
    src_dir: Path = this_dir.parent.parent.parent # {project_root}/src
    python_dir: Path = this_dir.parent # {project_root}/src/salary/python
    data_dir: Path = root_dir / 'data' / 'onlineclass_survey' # {project_root}/data
    result_dir: Path = root_dir / "_results" # {project_root}/_results

    csv_file: Path = data_dir / 'learning-style.csv'
    test_ratio = 0.2

    FMT_INT = "{:,d}"
    FMT_PER = "{:.1%}"
    FMT_FLOAT = "{:,.2f}"

    RAND_SEED = 27

    # QUESTIONS = [
    #     'Timestamp',
    #     'What is your major?',
    #     'What is your gender identity?',
    #     'Which of the following best describes you?',
    #     'Are you a domestic or international student?',
    #     'What is your class standing?',
    #     'How many hours do you study per week outside of regular class?',
    #     'What is your primary location of study?',
    #     'What is the average length of your commute?',
    #     'What best describes your current living situation?',
    #     'What is your average bedtime',
    #     'How many student organizations are you involved with?',
    #     'Are you currently concerned with catching / spreading COVID-19?',
    #     'Which learning modality do you generally prefer?', # label
    #     'Where did you hear about this survey?'
    # ]

    FEATURES = [
        'Major',
        'Gender',
        'Race',
        'Dom',
        'Class',
        'Study Hours',
        'Study Location',
        'Residence',
        'Roommates',
        'Bedtime',
        'Student Organizations',
        'Covid-19',
        'Learning Style', # label
    ]
    
