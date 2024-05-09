#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
REWARD_METHOD_DIR=$(realpath $(dirname ${SCRIPT_DIR}))
echo "REWARD_METHOD_DIR: ${REWARD_METHOD_DIR}"
source ${REWARD_METHOD_DIR}/constants.sh
echo "DQN_DIR: ${DQN_DIR}"


export MODEL_ID=army_recruit_manager
export SUB_AGENT_TYPE=army_recruit_manager
export MAP=BuildMarines
export MAP_ID=build_marines
export TRAIN_EPISODES=100
export LOG_SUFFIX="_01"

source ${DQN_DIR}/train_agent.sh
