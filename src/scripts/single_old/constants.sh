#!/bin/bash

SINGLE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPTS_DIR=$(realpath $(dirname ${SINGLE_DIR}))
source ${SCRIPTS_DIR}/constants.sh

export AGENT_TYPE="single"