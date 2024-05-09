#!/bin/bash

REWARD_METHOD_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "REWARD_METHOD_DIR2: ${REWARD_METHOD_DIR}"
DQN_DIR=$(realpath $(dirname ${REWARD_METHOD_DIR}))
echo "DQN_DIR: ${DQN_DIR}"
source ${DQN_DIR}/constants.sh

export REWARD_METHOD="score"