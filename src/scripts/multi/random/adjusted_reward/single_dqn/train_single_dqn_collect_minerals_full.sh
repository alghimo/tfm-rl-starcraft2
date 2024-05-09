#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# export EPSILON_DECAY=0.985 # 150 EP
# export EPSILON_DECAY=0.993 # 300 EP
export EPSILON_DECAY=0.995 # 500 EP
# export EPSILON_DECAY=0.997 # 1000 EP

source ${SCRIPT_DIR}/train_single_dqn_collect_minerals_reward.sh
source ${SCRIPT_DIR}/train_single_dqn_collect_minerals_adjusted_reward.sh
source ${SCRIPT_DIR}/train_single_dqn_collect_minerals_score.sh
