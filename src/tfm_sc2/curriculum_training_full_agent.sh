MODEL_ID=single_drl
LOG_DIR="../../models/${MODEL_ID}"

LOG_SUFFIX="_01"
mkdir -p ${LOG_DIR}
BURNIN_EPISODES=10
TRAIN_EPISODES=200
EXPLOIT_EPISODES=200

echo "Collecting experiences for CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${BURNIN_EPISODES} --load_networks_only --random_mode --model_id ${MODEL_ID_COLLECT_MINERALS} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas_random${LOG_SUFFIX}.log

echo "Training CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_COLLECT_MINERALS} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas${LOG_SUFFIX}.log

echo "Exploiting CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_COLLECT_MINERALS} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas${LOG_SUFFIX}.log

echo "Collecting experiences for BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${BURNIN_EPISODES} --load_networks_only --random_mode --model_id ${MODEL_ID_BUILD_MARINES} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines_random${LOG_SUFFIX}.log

echo "Training on BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_BUILD_MARINES} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines_random${LOG_SUFFIX}.log

echo "Exploiting BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_BUILD_MARINES} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines_random${LOG_SUFFIX}.log


# echo "Collecting experiences for DefeatRoaches"
# python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${BURNIN_EPISODES} --load_networks_only --random_mode --model_id ${MODEL_ID_DEFEAT_ROACHES} 2>&1| tee ${LOG_DIR_DEFEAT_ROACHES}/DefeatRoaches_random${LOG_SUFFIX}.log
# echo "Training on DefeatRoaches"
# python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_DEFEAT_ROACHES} 2>&1| tee ${LOG_DIR_DEFEAT_ROACHES}/DefeatRoaches_random${LOG_SUFFIX}.log
# echo "Exploiting DefeatRoaches"
# python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_DEFEAT_ROACHES} 2>&1| tee ${LOG_DIR_DEFEAT_ROACHES}/DefeatRoaches_random${LOG_SUFFIX}.log

echo "Collecting experiences for DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${BURNIN_EPISODES} --load_networks_only --random_mode --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings_random${LOG_SUFFIX}.log
echo "Training on DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings_random${LOG_SUFFIX}.log
echo "Exploiting DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings_random${LOG_SUFFIX}.log

echo "CollectingExperiences for Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${BURNIN_EPISODES} --load_networks_only --random_mode --model_id ${MODEL_ID_SIMPLE64} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64_random${LOG_SUFFIX}.log
echo "Training on Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_SIMPLE64} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64_random${LOG_SUFFIX}.log
echo "Exploiting Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_SIMPLE64} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64_random${LOG_SUFFIX}.log
