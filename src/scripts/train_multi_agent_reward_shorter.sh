#!/bin/bash

SCRIPTS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export SRC_DIR=$(realpath $(dirname ${SCRIPTS_DIR}))
export PYTHON_SCRIPTS_DIR=$(realpath ${SRC_DIR}/tfm_sc2)
export BASE_MODELS_DIR=$(realpath $(dirname ${SRC_DIR})/models)
export AGENT_TYPE="multi"
export AGENT_ALGORITHM="dqn"
export REWARD_METHOD="reward"
export MEMORY_SIZE=10000
export BURN_IN=1000
MODELS_DIR="${BASE_MODELS_DIR}/05_shorter"

# Base Manager
export AGENT_SUBTYPE="base_manager"
export BASE_MODEL_ID=multi_dqn_${AGENT_SUBTYPE}
export MAP=CollectMineralsAndGas
export TRAIN_EPISODES=100
export EPSILON_DECAY=0.98 # 300 EP
export LEARNING_RATE_MILESTONES="40 70 90"
export LOG_SUFFIX="_01"
export LEARNING_RATE_MILESTONES="40 70 90"
export LEARNING_RATE=0.001
export BASE_MANAGER_DQN_SIZE="medium" # extra_small, small, medium, large, extra_large

AGENT_KEY="${AGENT_TYPE}.${AGENT_ALGORITHM}.${AGENT_SUBTYPE}"
MODEL_ID="${BASE_MODEL_ID}_${REWARD_METHOD}"
BASE_MANAGER_MODEL_DIR="${MODELS_DIR}/${MODEL_ID}"

# mkdir -p ${BASE_MANAGER_MODEL_DIR}

# echo "Training Base Manager on ${MAP}"
# touch ${BASE_MANAGER_MODEL_DIR}/_01_training_start_${TRAIN_EPISODES}_ep

# python ${PYTHON_SCRIPTS_DIR}/runner.py \
#     --agent_key "${AGENT_KEY}" \
#     --map_name "${MAP}" \
#     --num_episodes ${TRAIN_EPISODES} \
#     --log_file ${BASE_MANAGER_MODEL_DIR}/training${LOG_SUFFIX}.log \
#     --model_id ${MODEL_ID} \
#     --models_path ${MODELS_DIR} \
#     --epsilon_decay ${EPSILON_DECAY} \
#     --lr_milestones ${LEARNING_RATE_MILESTONES} \
#     --lr ${LEARNING_RATE} \
#     --dqn_size ${BASE_MANAGER_DQN_SIZE} \
#     --memory_size ${MEMORY_SIZE} \
#     --burn_in ${BURN_IN} \
#     --reward_method ${REWARD_METHOD} 2>&1 | tee ${BASE_MANAGER_MODEL_DIR}/${MAP}${LOG_SUFFIX}.log
# touch ${BASE_MANAGER_MODEL_DIR}/_02_training_done_${TRAIN_EPISODES}_ep

# Army Recruit Manager
export AGENT_SUBTYPE="army_recruit_manager"
export BASE_MODEL_ID=multi_dqn_${AGENT_SUBTYPE}
export MAP=BuildMarines
export TRAIN_EPISODES=50
export EPSILON_DECAY=0.96 # 300 EP
export LOG_SUFFIX="_01"
export LEARNING_RATE_MILESTONES="30 40 45"
export LEARNING_RATE=0.001
export ARMY_RECRUIT_MANAGER_DQN_SIZE="small" # extra_small, small, medium, large, extra_large

AGENT_KEY="${AGENT_TYPE}.${AGENT_ALGORITHM}.${AGENT_SUBTYPE}"
MODEL_ID="${BASE_MODEL_ID}_${REWARD_METHOD}"
ARMY_RECRUIT_MANAGER_MODEL_DIR="${MODELS_DIR}/${MODEL_ID}"

# mkdir -p ${ARMY_RECRUIT_MANAGER_MODEL_DIR}

# echo "Training Army Recruit Manager on ${MAP}"
# touch ${ARMY_RECRUIT_MANAGER_MODEL_DIR}/_01_training_start_${TRAIN_EPISODES}_ep

# python ${PYTHON_SCRIPTS_DIR}/runner.py \
#     --agent_key "${AGENT_KEY}" \
#     --map_name "${MAP}" \
#     --num_episodes ${TRAIN_EPISODES} \
#     --log_file ${ARMY_RECRUIT_MANAGER_MODEL_DIR}/training${LOG_SUFFIX}.log \
#     --model_id ${MODEL_ID} \
#     --models_path ${MODELS_DIR} \
#     --epsilon_decay ${EPSILON_DECAY} \
#     --lr_milestones ${LEARNING_RATE_MILESTONES} \
#     --lr ${LEARNING_RATE} \
#     --dqn_size ${ARMY_RECRUIT_MANAGER_DQN_SIZE} \
#     --memory_size ${MEMORY_SIZE} \
#     --burn_in ${BURN_IN} \
#     --reward_method ${REWARD_METHOD} 2>&1 | tee ${ARMY_RECRUIT_MANAGER_MODEL_DIR}/${MAP}${LOG_SUFFIX}.log
# touch ${ARMY_RECRUIT_MANAGER_MODEL_DIR}/_02_training_done_${TRAIN_EPISODES}_ep

# Army Attack Manager
export AGENT_SUBTYPE="army_attack_manager"
export BASE_MODEL_ID=multi_dqn_${AGENT_SUBTYPE}
export MAP=DefeatZerglingsAndBanelings
export TRAIN_EPISODES=150
export EPSILON_DECAY=0.985 # 300 EP
export LOG_SUFFIX="_01"
export LEARNING_RATE_MILESTONES="100 150 175"
export LEARNING_RATE=0.001
export ARMY_ATTACK_MANAGER_DQN_SIZE="extra_small" # extra_small, small, medium, large, extra_large

AGENT_KEY="${AGENT_TYPE}.${AGENT_ALGORITHM}.${AGENT_SUBTYPE}"
MODEL_ID="${BASE_MODEL_ID}_${REWARD_METHOD}"
ARMY_ATTACK_MANAGER_MODEL_DIR="${MODELS_DIR}/${MODEL_ID}"

# mkdir -p ${ARMY_ATTACK_MANAGER_MODEL_DIR}

# echo "Training Army Attack Manager on ${MAP}"
# touch ${ARMY_ATTACK_MANAGER_MODEL_DIR}/_01_training_start_${TRAIN_EPISODES}_ep

# python ${PYTHON_SCRIPTS_DIR}/runner.py \
#     --agent_key "${AGENT_KEY}" \
#     --map_name "${MAP}" \
#     --num_episodes ${TRAIN_EPISODES} \
#     --log_file ${ARMY_ATTACK_MANAGER_MODEL_DIR}/training${LOG_SUFFIX}.log \
#     --model_id ${MODEL_ID} \
#     --models_path ${MODELS_DIR} \
#     --epsilon_decay ${EPSILON_DECAY} \
#     --lr_milestones ${LEARNING_RATE_MILESTONES} \
#     --lr ${LEARNING_RATE} \
#     --dqn_size ${ARMY_ATTACK_MANAGER_DQN_SIZE} \
#     --memory_size ${MEMORY_SIZE} \
#     --burn_in ${BURN_IN} \
#     --reward_method ${REWARD_METHOD} 2>&1 | tee ${ARMY_ATTACK_MANAGER_MODEL_DIR}/${MAP}${LOG_SUFFIX}.log
# touch ${ARMY_ATTACK_MANAGER_MODEL_DIR}/_02_training_done_${TRAIN_EPISODES}_ep

# Game Manager
export AGENT_SUBTYPE="game_manager"
export BASE_MODEL_ID=multi_dqn_${AGENT_SUBTYPE}
export MAP=Simple64
export TRAIN_EPISODES=50
export EPSILON_DECAY=0.93
export LOG_SUFFIX="_01"
export LEARNING_RATE_MILESTONES="38 44 48"
export LEARNING_RATE=0.001
export GAME_MANAGER_DQN_SIZE="small" # extra_small, small, medium, large, extra_large

AGENT_KEY="${AGENT_TYPE}.${AGENT_ALGORITHM}.${AGENT_SUBTYPE}"
MODEL_ID="${BASE_MODEL_ID}_${REWARD_METHOD}"
MODEL_DIR="${MODELS_DIR}/${MODEL_ID}"

mkdir -p ${MODEL_DIR}

# Copy sub-agents
cp -r ${BASE_MANAGER_MODEL_DIR} ${MODEL_DIR}/base_manager
cp -r ${ARMY_RECRUIT_MANAGER_MODEL_DIR} ${MODEL_DIR}/army_recruit_manager
cp -r ${ARMY_ATTACK_MANAGER_MODEL_DIR} ${MODEL_DIR}/army_attack_manager

echo "Training Game Manager on ${MAP}"
touch ${MODEL_DIR}/_01_training_start_${TRAIN_EPISODES}_ep

python ${PYTHON_SCRIPTS_DIR}/runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "${MAP}" \
    --num_episodes ${TRAIN_EPISODES} \
    --log_file ${MODEL_DIR}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID} \
    --models_path ${MODELS_DIR} \
    --epsilon_decay ${EPSILON_DECAY} \
    --lr_milestones ${LEARNING_RATE_MILESTONES} \
    --lr ${LEARNING_RATE} \
    --dqn_size ${GAME_MANAGER_DQN_SIZE} \
    --memory_size ${MEMORY_SIZE} \
    --burn_in ${BURN_IN} \
    --reward_method ${REWARD_METHOD} 2>&1 | tee ${MODEL_DIR}/${MAP}${LOG_SUFFIX}.log
touch ${MODEL_DIR}/_02_training_done_${TRAIN_EPISODES}_ep

echo "Training02"

echo "Moving ${MODEL_DIR} to ${MODEL_DIR}_train01"
mv ${MODEL_DIR} ${MODEL_DIR}_train01

mkdir -p ${MODEL_DIR}

# Copy sub-agents
cp -r ${BASE_MANAGER_MODEL_DIR} ${MODEL_DIR}/base_manager
cp -r ${ARMY_RECRUIT_MANAGER_MODEL_DIR} ${MODEL_DIR}/army_recruit_manager
cp -r ${ARMY_ATTACK_MANAGER_MODEL_DIR} ${MODEL_DIR}/army_attack_manager

echo "Training Game Manager on ${MAP}"
touch ${MODEL_DIR}/_01_training_start_${TRAIN_EPISODES}_ep

python ${PYTHON_SCRIPTS_DIR}/runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "${MAP}" \
    --num_episodes ${TRAIN_EPISODES} \
    --log_file ${MODEL_DIR}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID} \
    --models_path ${MODELS_DIR} \
    --epsilon_decay ${EPSILON_DECAY} \
    --lr_milestones ${LEARNING_RATE_MILESTONES} \
    --lr ${LEARNING_RATE} \
    --dqn_size ${GAME_MANAGER_DQN_SIZE} \
    --memory_size ${MEMORY_SIZE} \
    --burn_in ${BURN_IN} \
    --reward_method ${REWARD_METHOD} 2>&1 | tee ${MODEL_DIR}/${MAP}${LOG_SUFFIX}.log
touch ${MODEL_DIR}/_02_training_done_${TRAIN_EPISODES}_ep

echo "Training03"

echo "Moving ${MODEL_DIR} to ${MODEL_DIR}_train02"
mv ${MODEL_DIR} ${MODEL_DIR}_train02

mkdir -p ${MODEL_DIR}

# Copy sub-agents
cp -r ${BASE_MANAGER_MODEL_DIR} ${MODEL_DIR}/base_manager
cp -r ${ARMY_RECRUIT_MANAGER_MODEL_DIR} ${MODEL_DIR}/army_recruit_manager
cp -r ${ARMY_ATTACK_MANAGER_MODEL_DIR} ${MODEL_DIR}/army_attack_manager

echo "Training Game Manager on ${MAP}"
touch ${MODEL_DIR}/_01_training_start_${TRAIN_EPISODES}_ep

python ${PYTHON_SCRIPTS_DIR}/runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "${MAP}" \
    --num_episodes ${TRAIN_EPISODES} \
    --log_file ${MODEL_DIR}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID} \
    --models_path ${MODELS_DIR} \
    --epsilon_decay ${EPSILON_DECAY} \
    --lr_milestones ${LEARNING_RATE_MILESTONES} \
    --lr ${LEARNING_RATE} \
    --dqn_size ${GAME_MANAGER_DQN_SIZE} \
    --memory_size ${MEMORY_SIZE} \
    --burn_in ${BURN_IN} \
    --reward_method ${REWARD_METHOD} 2>&1 | tee ${MODEL_DIR}/${MAP}${LOG_SUFFIX}.log
touch ${MODEL_DIR}/_02_training_done_${TRAIN_EPISODES}_ep

echo "Training04"

echo "Moving ${MODEL_DIR} to ${MODEL_DIR}_train03"
mv ${MODEL_DIR} ${MODEL_DIR}_train03

mkdir -p ${MODEL_DIR}

# Copy sub-agents
cp -r ${BASE_MANAGER_MODEL_DIR} ${MODEL_DIR}/base_manager
cp -r ${ARMY_RECRUIT_MANAGER_MODEL_DIR} ${MODEL_DIR}/army_recruit_manager
cp -r ${ARMY_ATTACK_MANAGER_MODEL_DIR} ${MODEL_DIR}/army_attack_manager

echo "Training Game Manager on ${MAP}"
touch ${MODEL_DIR}/_01_training_start_${TRAIN_EPISODES}_ep

python ${PYTHON_SCRIPTS_DIR}/runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "${MAP}" \
    --num_episodes ${TRAIN_EPISODES} \
    --log_file ${MODEL_DIR}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID} \
    --models_path ${MODELS_DIR} \
    --epsilon_decay ${EPSILON_DECAY} \
    --lr_milestones ${LEARNING_RATE_MILESTONES} \
    --lr ${LEARNING_RATE} \
    --dqn_size ${GAME_MANAGER_DQN_SIZE} \
    --memory_size ${MEMORY_SIZE} \
    --burn_in ${BURN_IN} \
    --reward_method ${REWARD_METHOD} 2>&1 | tee ${MODEL_DIR}/${MAP}${LOG_SUFFIX}.log
touch ${MODEL_DIR}/_02_training_done_${TRAIN_EPISODES}_ep

echo "Training05"

echo "Moving ${MODEL_DIR} to ${MODEL_DIR}_train04"
mv ${MODEL_DIR} ${MODEL_DIR}_train04

mkdir -p ${MODEL_DIR}

# Copy sub-agents
cp -r ${BASE_MANAGER_MODEL_DIR} ${MODEL_DIR}/base_manager
cp -r ${ARMY_RECRUIT_MANAGER_MODEL_DIR} ${MODEL_DIR}/army_recruit_manager
cp -r ${ARMY_ATTACK_MANAGER_MODEL_DIR} ${MODEL_DIR}/army_attack_manager

echo "Training Game Manager on ${MAP}"
touch ${MODEL_DIR}/_01_training_start_${TRAIN_EPISODES}_ep

python ${PYTHON_SCRIPTS_DIR}/runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "${MAP}" \
    --num_episodes ${TRAIN_EPISODES} \
    --log_file ${MODEL_DIR}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID} \
    --models_path ${MODELS_DIR} \
    --epsilon_decay ${EPSILON_DECAY} \
    --lr_milestones ${LEARNING_RATE_MILESTONES} \
    --lr ${LEARNING_RATE} \
    --dqn_size ${GAME_MANAGER_DQN_SIZE} \
    --memory_size ${MEMORY_SIZE} \
    --burn_in ${BURN_IN} \
    --reward_method ${REWARD_METHOD} 2>&1 | tee ${MODEL_DIR}/${MAP}${LOG_SUFFIX}.log
touch ${MODEL_DIR}/_02_training_done_${TRAIN_EPISODES}_ep
