BASE_MODEL_ID=single_drl
MODEL_ID_COLLECT_MINERALS=${BASE_MODEL_ID}_collect_minerals
MODEL_ID_BUILD_MARINES=${BASE_MODEL_ID}_build_marines
MODEL_ID_DEFEAT_ROACHES=${BASE_MODEL_ID}_defeat_roaches
MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS=${BASE_MODEL_ID}_defeat_zerglings_and_banelings
MODEL_ID_SIMPLE64=${BASE_MODEL_ID}_simple64

BASE_LOG_DIR="../../models/${BASE_MODEL_ID}"
LOG_DIR_COLLECT_MINERALS=${BASE_LOG_DIR}_collect_minerals
LOG_DIR_BUILD_MARINES=${BASE_LOG_DIR}_build_marines
LOG_DIR_DEFEAT_ROACHES=${BASE_LOG_DIR}_defeat_roaches
LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS=${BASE_LOG_DIR}_defeat_zerglings_and_banelings
LOG_DIR_SIMPLE64=${BASE_LOG_DIR}_simple64

LOG_SUFFIX="_01"
mkdir -p ${LOG_DIR_COLLECT_MINERALS} ${LOG_DIR_BUILD_MARINES} ${LOG_DIR_DEFEAT_ROACHES} ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS} ${LOG_DIR_SIMPLE64}
BURNIN_EPISODES=100
TRAIN_EPISODES=200
EXPLOIT_EPISODES=200

touch ${LOG_DIR_COLLECT_MINERALS}/_started
echo "Collecting experiences for CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_COLLECT_MINERALS} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas_random${LOG_SUFFIX}.log
touch ${LOG_DIR_COLLECT_MINERALS}/_01_random_done_${BURNIN_EPISODES}_ep

echo "Training CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_COLLECT_MINERALS} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas${LOG_SUFFIX}.log
touch ${LOG_DIR_COLLECT_MINERALS}/_02_training_done_${TRAIN_EPISODES}_ep

echo "Exploiting CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_COLLECT_MINERALS} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas${LOG_SUFFIX}.log
touch ${LOG_DIR_COLLECT_MINERALS}/_03_exploit_done_${EXPLOIT_EPISODES}_ep

touch ${LOG_DIR_BUILD_MARINES}/_started
echo "Collecting experiences for BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_BUILD_MARINES} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines_random${LOG_SUFFIX}.log
touch ${LOG_DIR_BUILD_MARINES}/_01_random_done_${BURNIN_EPISODES}_ep

echo "Training on BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_BUILD_MARINES} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines_random${LOG_SUFFIX}.log
touch ${LOG_DIR_BUILD_MARINES}/_02_train_done_${TRAIN_EPISODES}_ep

echo "Exploiting BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_BUILD_MARINES} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines_random${LOG_SUFFIX}.log
touch ${LOG_DIR_BUILD_MARINES}/_03_exploit_done_${EXPLOIT_EPISODES}_ep

touch ${LOG_DIR_DEFEAT_ROACHES}/_started
echo "Collecting experiences for DefeatRoaches"
python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_DEFEAT_ROACHES} 2>&1| tee ${LOG_DIR_DEFEAT_ROACHES}/DefeatRoaches_random${LOG_SUFFIX}.log
touch ${LOG_DIR_DEFEAT_ROACHES}/_01_random_done_${BURNIN_EPISODES}_ep
echo "Training on DefeatRoaches"
python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_DEFEAT_ROACHES} 2>&1| tee ${LOG_DIR_DEFEAT_ROACHES}/DefeatRoaches_random${LOG_SUFFIX}.log
touch ${LOG_DIR_DEFEAT_ROACHES}/_02_train_done_${TRAIN_EPISODES}_ep
echo "Exploiting DefeatRoaches"
python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_DEFEAT_ROACHES} 2>&1| tee ${LOG_DIR_DEFEAT_ROACHES}/DefeatRoaches_random${LOG_SUFFIX}.log
touch ${LOG_DIR_DEFEAT_ROACHES}/_03_exploit_done_${EXPLOIT_EPISODES}_ep

touch ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/_started
echo "Collecting experiences for DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings_random${LOG_SUFFIX}.log
touch ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/_01_random_done_${BURNIN_EPISODES}_ep
echo "Training on DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings_random${LOG_SUFFIX}.log
touch ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/_02_train_done_${TRAIN_EPISODES}_ep
echo "Exploiting DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings_random${LOG_SUFFIX}.log
touch ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/_03_exploit_done_${EXPLOIT_EPISODES}_ep

touch ${LOG_DIR_SIMPLE64}/_started
echo "CollectingExperiences for Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_SIMPLE64} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64_random${LOG_SUFFIX}.log
touch ${LOG_DIR_SIMPLE64}/_01_random_done_${BURNIN_EPISODES}_ep
echo "Training on Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_SIMPLE64} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64_random${LOG_SUFFIX}.log
touch ${LOG_DIR_SIMPLE64}/_02_train_done_${TRAIN_EPISODES}_ep
echo "Exploiting Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_SIMPLE64} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64_random${LOG_SUFFIX}.log
touch ${LOG_DIR_SIMPLE64}/_03_exploit_done_${EXPLOIT_EPISODES}_ep
