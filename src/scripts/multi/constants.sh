#!/bin/bash

AGENT_TYPE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "AGENT_TYPE_DIR2: ${AGENT_TYPE_DIR}"
SCRIPTS_DIR=$(realpath $(dirname ${AGENT_TYPE_DIR}))
echo "SCRIPTS_DIR: ${SCRIPTS_DIR}"
source ${SCRIPTS_DIR}/constants.sh

export AGENT_TYPE="multi"