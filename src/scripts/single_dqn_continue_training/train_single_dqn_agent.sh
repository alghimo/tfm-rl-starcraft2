#!/bin/bash

AGENT_KEY="${AGENT_TYPE}.${AGENT_ALGORITHM}"
MODELS_DIR="${BASE_MODELS_DIR}/${AGENT_ALGORITHM}/${AGENT_TYPE}/${REWARD_METHOD}"

MODEL_DIR="${MODELS_DIR}/${MODEL_ID}"
BUFFER_FILE="${MODEL_DIR}/buffer.pkl"

mkdir -p ${MODEL_DIR}

echo "Training ${MAP}"
touch ${MODEL_DIR}/_01_training_start_${TRAIN_EPISODES}_ep
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
touch ${MODEL_DIR}/_02_training_done_${TRAIN_EPISODES}_ep
