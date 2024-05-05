#!/bin/bash

# SCRIPT_DIR=$(realpath $(dirname $0))
CONSTANTS_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export SRC_DIR=$(realpath $(dirname ${CONSTANTS_SCRIPT_DIR}))
export PYTHON_SCRIPTS_DIR=$(realpath ${SRC_DIR}/tfm_sc2)
export BASE_MODELS_DIR=$(realpath $(dirname ${SRC_DIR})/models)
export BUFFERS_DIR="${BASE_MODELS_DIR}/buffers"
echo "The absolute path to the script is: $CONSTANTS_SCRIPT_DIR"
echo "The absolute path to the src dir is: $SRC_DIR"
echo "The absolute path to the python scripts dir is: $PYTHON_SCRIPTS_DIR"
echo "The absolute path to the models dir is: $BASE_MODELS_DIR"