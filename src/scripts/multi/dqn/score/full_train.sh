#!/bin/bash

SCORE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCORE_DIR}/constants.sh
# source ${SCRIPT_DIR}/train_single_dqn_collect_minerals_score.sh
# source ${SCRIPT_DIR}/train_single_dqn_build_marines_score.sh
MODELS_DIR="${BASE_MODELS_DIR}/${AGENT_ALGORITHM}/${AGENT_TYPE}/${REWARD_METHOD}"
GAME_MANAGER_MODEL_DIR=${MODELS_DIR}/game_manager
mkdir -p ${GAME_MANAGER_MODEL_DIR}

echo "Training Base Manager"
source ${SCORE_DIR}/base_manager/train.sh
cp -r ${MODELS_DIR}/base_manager ${GAME_MANAGER_MODEL_DIR}/

echo "Training Attack Manager"
source ${SCORE_DIR}/army_attack_manager/train.sh
cp -r ${MODELS_DIR}/army_attack_manager ${GAME_MANAGER_MODEL_DIR}/

echo "Training Army Recruit Manager"
source ${SCORE_DIR}/army_recruit_manager/train.sh
cp -r ${MODELS_DIR}/army_recruit_manager ${GAME_MANAGER_MODEL_DIR}/

echo "Training Game Manager"
source ${SCORE_DIR}/game_manager/train.sh
