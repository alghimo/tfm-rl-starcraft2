#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPTS_DIR=$(realpath $(dirname ${SCRIPT_DIR}))
source ${SCRIPTS_DIR}/constants.sh

export MODEL_ID=collect_minerals
export MAP=CollectMineralsAndGas
export TRAIN_EPISODES=500

export LOG_SUFFIX="_01"

source ${SCRIPT_DIR}/train_${AGENT_TYPE}_${AGENT_ALGORITHM}_agent.sh