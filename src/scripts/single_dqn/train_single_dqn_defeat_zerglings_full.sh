#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export EPSILON_DECAY=0.997

source ${SCRIPT_DIR}/train_single_dqn_defeat_zerglings_score.sh
source ${SCRIPT_DIR}/train_single_dqn_defeat_zerglings_reward.sh
source ${SCRIPT_DIR}/train_single_dqn_defeat_zerglings_adjusted_reward.sh

