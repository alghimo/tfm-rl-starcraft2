#!/bin/bash

DQN_CONSTANTS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SINGLE_DIR=$(realpath $(dirname ${DQN_CONSTANTS_DIR}))
source ${SCRIPTS_DIR}/constants.sh

export AGENT_ALGORITHM="dqn"