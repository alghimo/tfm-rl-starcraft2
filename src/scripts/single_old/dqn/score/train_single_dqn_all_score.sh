#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export EPSILON_DECAY=0.993

# source ${SCRIPT_DIR}/train_single_dqn_collect_minerals_score.sh
# source ${SCRIPT_DIR}/train_single_dqn_build_marines_score.sh
source ${SCRIPT_DIR}/train_single_dqn_defeat_zerglings_score.sh
source ${SCRIPT_DIR}/train_single_dqn_simple64_score.sh
source ${SCRIPT_DIR}/train_single_dqn_curriculum_score.sh

