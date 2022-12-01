#!/bin/bash
#set -e

readonly _DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
readonly PYTHON_DIR="${_DIR}/python"
readonly DL_DIR="${_DIR}/_downloads"
readonly RESULT_DIR="${_DIR}/_results"

# ==========
# Exp Results

function preprocess() {
        # write requirements in json
        (cd ${_DIR}
         python -m src.onlineclass_survey.python.main \
                --run preprocess
        )
}

function run_model() {
        # write requirements in json
        (cd ${_DIR}
         python -m src.onlineclass_survey.python.main \
                --run run_model
        )
}


# ==========
# Main

function main() {
        preprocess
        run_model
}

main
