MODEL_ID_COLLECT_MINERALS=collect_minerals
MODEL_ID_BUILD_MARINES=build_marines
MODEL_ID_DEFEAT_ROACHES=defeat_roaches
MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS=defeat_zerglings_and_banelings
MODEL_ID_SIMPLE64=simple64

MODELS_DIR="../../models/dqn/single"
LOG_DIR_COLLECT_MINERALS="${MODELS_DIR}/${MODEL_ID_COLLECT_MINERALS}"
LOG_DIR_BUILD_MARINES="${MODELS_DIR}/${MODEL_ID_BUILD_MARINES}"
LOG_DIR_DEFEAT_ROACHES="${MODELS_DIR}/${MODEL_ID_DEFEAT_ROACHES}"
LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS="${MODELS_DIR}/${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS}"
LOG_DIR_SIMPLE64="${MODELS_DIR}/${MODEL_ID_SIMPLE64}"

LOG_SUFFIX="_01"
mkdir -p ${LOG_DIR_COLLECT_MINERALS} ${LOG_DIR_BUILD_MARINES} ${LOG_DIR_DEFEAT_ROACHES} ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS} ${LOG_DIR_SIMPLE64}
TRAIN_EPISODES=200
REWARD_METHOD="score"
AGENT_KEY="single.dqn"
echo "Training CollectMineralsAndGas"
touch ${LOG_DIR_COLLECT_MINERALS}/_01_training_start_${TRAIN_EPISODES}_ep
python runner.py --agent_key "${AGENT_KEY}" --map_name "CollectMineralsAndGas" --num_episodes ${TRAIN_EPISODES} --log_file ${LOG_DIR_COLLECT_MINERALS}/training${LOG_SUFFIX}.log --model_id ${MODEL_ID_COLLECT_MINERALS} --models_path ${MODELS_DIR} --reward_method ${REWARD_METHOD} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas${LOG_SUFFIX}.log
touch ${LOG_DIR_COLLECT_MINERALS}/_02_training_done_${TRAIN_EPISODES}_ep

echo "Training on BuildMarines"
touch ${LOG_DIR_BUILD_MARINES}/_01_training_start_${TRAIN_EPISODES}_ep
python runner.py --agent_key "${AGENT_KEY}" --map_name "BuildMarines" --num_episodes ${TRAIN_EPISODES} --log_file ${LOG_DIR_BUILD_MARINES}/training${LOG_SUFFIX}.log --model_id ${MODEL_ID_BUILD_MARINES} --models_path ${MODELS_DIR} --reward_method ${REWARD_METHOD} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines${LOG_SUFFIX}.log
touch ${LOG_DIR_BUILD_MARINES}/_02_training_done_${TRAIN_EPISODES}_ep

echo "Exploiting DefeatRoaches"
python runner.py --agent_key "${AGENT_KEY}" --map_name "DefeatRoaches" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_DEFEAT_ROACHES} --models_path ${MODELS_DIR} --reward_method ${REWARD_METHOD} 2>&1| tee ${LOG_DIR_DEFEAT_ROACHES}/DefeatRoaches_exploit${LOG_SUFFIX}.log
touch ${LOG_DIR_DEFEAT_ROACHES}/_04_exploit_done_${EXPLOIT_EPISODES}_ep

echo "Training on DefeatZerglingsAndBanelings"
touch ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/_01_training_start_${TRAIN_EPISODES}_ep
python runner.py --agent_key "${AGENT_KEY}" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${TRAIN_EPISODES} --log_file ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/training${LOG_SUFFIX}.log --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} --models_path ${MODELS_DIR} --reward_method ${REWARD_METHOD} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings${LOG_SUFFIX}.log
touch ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/_02_training_done_${TRAIN_EPISODES}_ep

echo "Training on Simple64"
touch ${LOG_DIR_SIMPLE64}/_01_training_start_${TRAIN_EPISODES}_ep
python runner.py --agent_key "${AGENT_KEY}" --map_name "Simple64" --num_episodes ${TRAIN_EPISODES} --log_file ${LOG_DIR_SIMPLE64}/training${LOG_SUFFIX}.log --model_id ${MODEL_ID_SIMPLE64} --models_path ${MODELS_DIR} --reward_method ${REWARD_METHOD} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64${LOG_SUFFIX}.log
touch ${LOG_DIR_SIMPLE64}/_02_training_done_${TRAIN_EPISODES}_ep
