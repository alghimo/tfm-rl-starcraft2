#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
REWARD_METHOD_DIR=$(realpath $(dirname ${SCRIPT_DIR}))
echo "REWARD_METHOD_DIR: ${REWARD_METHOD_DIR}"
source ${REWARD_METHOD_DIR}/constants.sh
echo "DQN_DIR: ${DQN_DIR}"


export MODEL_ID=base_manager
export SUB_AGENT_TYPE=base_manager
export MAP=CollectMineralsAndGas
export MAP_ID=collect_minerals_and_gas
export TRAIN_EPISODES=300
export EPSILON_DECAY=0.993 # 300 EP
export LOG_SUFFIX="_01"

source ${DQN_DIR}/train_agent.sh
