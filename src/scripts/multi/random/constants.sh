#!/bin/bash

DQN_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "DQN_DIR2: ${DQN_DIR}"
AGENT_TYPE_DIR=$(realpath $(dirname ${DQN_DIR}))
echo "AGENT_TYPE_DIR: ${AGENT_TYPE_DIR}"
source ${AGENT_TYPE_DIR}/constants.sh

export AGENT_ALGORITHM="random"