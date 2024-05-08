MODEL_ID_COLLECT_MINERALS=collect_minerals
MODEL_ID_BUILD_MARINES=build_marines
MODEL_ID_DEFEAT_ROACHES=defeat_roaches
MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS=defeat_zerglings_and_banelings
MODEL_ID_SIMPLE64=simple64

# MODELS_DIR="../../models/random/single"
MODELS_DIR="../../models/create-buffers"
MODEL_DIR_COLLECT_MINERALS="${MODELS_DIR}/${MODEL_ID_COLLECT_MINERALS}"
MODEL_DIR_BUILD_MARINES="${MODELS_DIR}/${MODEL_ID_BUILD_MARINES}"
MODEL_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS="${MODELS_DIR}/${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS}"
MODEL_DIR_SIMPLE64="${MODELS_DIR}/${MODEL_ID_SIMPLE64}"

LOG_SUFFIX="_01"
mkdir -p ${MODEL_DIR_COLLECT_MINERALS} ${MODEL_DIR_BUILD_MARINES} ${MODEL_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS} ${MODEL_DIR_SIMPLE64}
TRAIN_MINERALS_EPISODES=200
TRAIN_MARINES_EPISODES=100
TRAIN_ATTACK_EPISODES=600
TRAIN_SIMPLE64_EPISODES=100

REWARD_METHOD="score"
AGENT_KEY="single.random"
echo "Training CollectMineralsAndGas"
touch ${MODEL_DIR_COLLECT_MINERALS}/_01_training_start_${TRAIN_MINERALS_EPISODES}_ep
python runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "CollectMineralsAndGas" \
    --num_episodes ${TRAIN_MINERALS_EPISODES} \
    --log_file ${MODEL_DIR_COLLECT_MINERALS}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID_COLLECT_MINERALS} \
    --models_path ${MODELS_DIR} \
    --reward_method ${REWARD_METHOD} 2>&1 | tee ${MODEL_DIR_COLLECT_MINERALS}/CollectMineralsAndGas${LOG_SUFFIX}.log
touch ${MODEL_DIR_COLLECT_MINERALS}/_02_training_done_${TRAIN_MINERALS_EPISODES}_ep

echo "Training on BuildMarines"
touch ${MODEL_DIR_BUILD_MARINES}/_01_training_start_${TRAIN_MARINES_EPISODES}_ep
python runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "BuildMarines" \
    --num_episodes ${TRAIN_MARINES_EPISODES} \
    --log_file ${MODEL_DIR_BUILD_MARINES}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID_BUILD_MARINES} \
    --models_path ${MODELS_DIR} \
    --reward_method ${REWARD_METHOD} 2>&1| tee ${MODEL_DIR_BUILD_MARINES}/BuildMarines${LOG_SUFFIX}.log
touch ${MODEL_DIR_BUILD_MARINES}/_02_training_done_${TRAIN_MARINES_EPISODES}_ep

echo "Training on DefeatZerglingsAndBanelings"
touch ${MODEL_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/_01_training_start_${TRAIN_ATTACK_EPISODES}_ep
python runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "DefeatZerglingsAndBanelings" \
    --num_episodes ${TRAIN_ATTACK_EPISODES} \
    --log_file ${MODEL_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} \
    --models_path ${MODELS_DIR} \
    --reward_method ${REWARD_METHOD} 2>&1| tee ${MODEL_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings${LOG_SUFFIX}.log
touch ${MODEL_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/_02_training_done_${TRAIN_ATTACK_EPISODES}_ep

echo "Training on Simple64"
touch ${MODEL_DIR_SIMPLE64}/_01_training_start_${TRAIN_SIMPLE64_EPISODES}_ep
python runner.py \
    --agent_key "${AGENT_KEY}" \
    --map_name "Simple64" \
    --num_episodes ${TRAIN_SIMPLE64_EPISODES} \
    --log_file ${MODEL_DIR_SIMPLE64}/training${LOG_SUFFIX}.log \
    --model_id ${MODEL_ID_SIMPLE64} \
    --models_path ${MODELS_DIR} \
    --reward_method ${REWARD_METHOD} 2>&1| tee ${MODEL_DIR_SIMPLE64}/Simple64${LOG_SUFFIX}.log
touch ${MODEL_DIR_SIMPLE64}/_02_training_done_${TRAIN_SIMPLE64_EPISODES}_ep
