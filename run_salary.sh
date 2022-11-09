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
         python -m src.salary.python.main \
                --run preprocess
        )
}


# ==========
# Main

function main() {
        preprocess
}

main