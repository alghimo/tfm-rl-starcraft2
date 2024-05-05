#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPTS_DIR=$(realpath $(dirname ${SCRIPT_DIR}))
source ${SCRIPTS_DIR}/constants.sh
export REWARD_METHOD="reward"
# export REWARD_METHOD="adjusted_reward"
# export REWARD_METHOD="score"
export AGENT_TYPE="single"
#export AGENT_TYPE="multi"
export AGENT_ALGORITHM="dqn"
#export AGENT_ALGORITHM="random"

export MODEL_ID=collect_minerals
export MAP=CollectMineralsAndGas
export TRAIN_EPISODES=500

# export MODEL_ID=build_marines
# export MAP=BuildMarines
# export TRAIN_EPISODES=150

# export MODEL_ID=defeat_zerglings_and_banelings
# export MAP=DefeatZerglingsAndBanelings
# export TRAIN_EPISODES=1000

#export MODEL_ID=simple64
# export MAP=Simple64
# export TRAIN_EPISODES=300

export TRAIN_EPISODES=300
export LOG_SUFFIX="_01"

source ${SCRIPT_DIR}/train_${AGENT_TYPE}_${AGENT_ALGORITHM}_agent.sh
