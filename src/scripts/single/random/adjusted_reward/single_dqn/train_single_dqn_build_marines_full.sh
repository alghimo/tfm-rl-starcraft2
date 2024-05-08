#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# export EPSILON_DECAY=0.993
export EPSILON_DECAY=0.985

source ${SCRIPT_DIR}/train_single_dqn_build_marines_reward.sh
source ${SCRIPT_DIR}/train_single_dqn_build_marines_adjusted_reward.sh
source ${SCRIPT_DIR}/train_single_dqn_build_marines_score.sh
