#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export EPSILON_DECAY=0.993

source ${SCRIPT_DIR}/train_single_dqn_collect_minerals_adjusted_reward.sh
source ${SCRIPT_DIR}/train_single_dqn_build_marines_adjusted_reward.sh
source ${SCRIPT_DIR}/train_single_dqn_defeat_zerglings_adjusted_reward.sh
source ${SCRIPT_DIR}/train_single_dqn_simple64_adjusted_reward.sh

