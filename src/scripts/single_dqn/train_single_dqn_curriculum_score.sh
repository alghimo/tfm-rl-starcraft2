#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPTS_DIR=$(realpath $(dirname ${SCRIPT_DIR}))
source ${SCRIPTS_DIR}/constants.sh

# export REWARD_METHOD="reward"
# export REWARD_METHOD="adjusted_reward"
export REWARD_METHOD="score"
export AGENT_TYPE="single"
#export AGENT_TYPE="multi"
export AGENT_ALGORITHM="dqn"

export MODEL_ID=curriculum
export MAP_ID=collect_minerals
export MAP=CollectMineralsAndGas
export TRAIN_EPISODES=500

export LOG_SUFFIX="_01"

AGENT_KEY="${AGENT_TYPE}.${AGENT_ALGORITHM}"
MODELS_DIR="${BASE_MODELS_DIR}/${AGENT_ALGORITHM}/${AGENT_TYPE}/${REWARD_METHOD}"
MODEL_DIR="${MODELS_DIR}/${MODEL_ID}"

mkdir -p ${MODEL_DIR}

export MAP_ID=collect_minerals
export MAP=CollectMineralsAndGas
export TRAIN_EPISODES=500
export EPSILON_DECAY=0.995 # 500 EP

BUFFER_FILE="${BUFFERS_DIR}/${MAP_ID}_buffer.pkl"

echo "Training ${MAP}"
touch ${MODEL_DIR}/_01_${MAP_ID}_training_start_${TRAIN_EPISODES}_ep
python ${PYTHON_SCRIPTS_DIR}/runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "${MAP}" \
    --num_episodes ${TRAIN_EPISODES} \
    --log_file ${MODEL_DIR}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID} \
    --models_path ${MODELS_DIR} \
    --buffer_file ${BUFFER_FILE} \
    --epsilon_decay ${EPSILON_DECAY} \
    --reward_method ${REWARD_METHOD} 2>&1 | tee ${MODEL_DIR}/${MAP}${LOG_SUFFIX}.log
touch ${MODEL_DIR}/_02_${MAP_ID}_training_done_${TRAIN_EPISODES}_ep

export MAP_ID=build_marines
export MAP=BuildMarines
export TRAIN_EPISODES=150
export EPSILON_DECAY=0.985 # 150 EP

BUFFER_FILE="${BUFFERS_DIR}/${MAP_ID}_buffer.pkl"

echo "Training ${MAP}"
touch ${MODEL_DIR}/_03_${MAP_ID}_training_start_${TRAIN_EPISODES}_ep
python ${PYTHON_SCRIPTS_DIR}/runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "${MAP}" \
    --num_episodes ${TRAIN_EPISODES} \
    --log_file ${MODEL_DIR}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID} \
    --models_path ${MODELS_DIR} \
    --buffer_file ${BUFFER_FILE} \
    --epsilon_decay ${EPSILON_DECAY} \
    --reset_epsilon \
    --reward_method ${REWARD_METHOD} 2>&1 | tee ${MODEL_DIR}/${MAP}${LOG_SUFFIX}.log
touch ${MODEL_DIR}/_04_${MAP_ID}_training_done_${TRAIN_EPISODES}_ep

export MAP_ID=defeat_zerglings_and_banelings
export MAP=DefeatZerglingsAndBanelings
export TRAIN_EPISODES=1000
export EPSILON_DECAY=0.997 # 1000 EP

BUFFER_FILE="${BUFFERS_DIR}/${MAP_ID}_buffer.pkl"

echo "Training ${MAP}"
touch ${MODEL_DIR}/_05_${MAP_ID}_training_start_${TRAIN_EPISODES}_ep
python ${PYTHON_SCRIPTS_DIR}/runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "${MAP}" \
    --num_episodes ${TRAIN_EPISODES} \
    --log_file ${MODEL_DIR}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID} \
    --models_path ${MODELS_DIR} \
    --buffer_file ${BUFFER_FILE} \
    --epsilon_decay ${EPSILON_DECAY} \
    --reset_epsilon \
    --reward_method ${REWARD_METHOD} 2>&1 | tee ${MODEL_DIR}/${MAP}${LOG_SUFFIX}.log
touch ${MODEL_DIR}/_06_${MAP_ID}_training_done_${TRAIN_EPISODES}_ep

export MAP_ID=simple64
export MAP=Simple64
export TRAIN_EPISODES=300
export EPSILON_DECAY=0.993 # 300 EP
BUFFER_FILE="${BUFFERS_DIR}/${MAP_ID}_buffer.pkl"

echo "Training ${MAP}"
touch ${MODEL_DIR}/_07_${MAP_ID}_training_start_${TRAIN_EPISODES}_ep
python ${PYTHON_SCRIPTS_DIR}/runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "${MAP}" \
    --num_episodes ${TRAIN_EPISODES} \
    --log_file ${MODEL_DIR}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID} \
    --models_path ${MODELS_DIR} \
    --buffer_file ${BUFFER_FILE} \
    --epsilon_decay ${EPSILON_DECAY} \
    --reset_epsilon \
    --reward_method ${REWARD_METHOD} 2>&1 | tee ${MODEL_DIR}/${MAP}${LOG_SUFFIX}.log
touch ${MODEL_DIR}/_08_${MAP_ID}_training_done_${TRAIN_EPISODES}_ep
