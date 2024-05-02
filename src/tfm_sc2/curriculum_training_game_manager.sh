MODEL_ID=drl_game_manager
MODEL_ID_BASE_MANAGER=drl_base_manager
MODEL_ID_ARMY_MANAGER=drl_army_manager
MODEL_ID_ATTACK_MANAGER=drl_attack_manager

MODELS_DIR
LOG_DIR="${MODELS_DIR}/${MODEL_ID}"
LOG_DIR_BASE_MANAGER="${MODELS_DIR}/${MODEL_ID_BASE_MANAGER}"
LOG_DIR_ARMY_MANAGER="${MODELS_DIR}/${MODEL_ID_ARMY_MANAGER}"
LOG_DIR_ATTACK_MANAGER="${MODELS_DIR}/${MODEL_ID_ATTACK_MANAGER}"

LOG_SUFFIX="_01"
mkdir -p ${LOG_DIR} ${LOG_DIR_BASE_MANAGER} ${LOG_DIR_ARMY_MANAGER} ${LOG_DIR_ATTACK_MANAGER}
BURNIN_EPISODES=10
TRAIN_EPISODES=200
EXPLOIT_EPISODES=200

echo "Collecting experiences for CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_COLLECT_MINERALS} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas_random${LOG_SUFFIX}.log

echo "Training CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_COLLECT_MINERALS} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas${LOG_SUFFIX}.log

echo "Exploiting CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_COLLECT_MINERALS} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas${LOG_SUFFIX}.log

echo "Collecting experiences for BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_BUILD_MARINES} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines_random${LOG_SUFFIX}.log

echo "Training on BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_BUILD_MARINES} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines_random${LOG_SUFFIX}.log

echo "Exploiting BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_BUILD_MARINES} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines_random${LOG_SUFFIX}.log


echo "Collecting experiences for DefeatRoaches"
python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_DEFEAT_ROACHES} 2>&1| tee ${LOG_DIR_DEFEAT_ROACHES}/DefeatRoaches_random${LOG_SUFFIX}.log
echo "Training on DefeatRoaches"
python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_DEFEAT_ROACHES} 2>&1| tee ${LOG_DIR_DEFEAT_ROACHES}/DefeatRoaches_random${LOG_SUFFIX}.log
echo "Exploiting DefeatRoaches"
python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_DEFEAT_ROACHES} 2>&1| tee ${LOG_DIR_DEFEAT_ROACHES}/DefeatRoaches_random${LOG_SUFFIX}.log

echo "Collecting experiences for DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings_random${LOG_SUFFIX}.log
echo "Training on DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings_random${LOG_SUFFIX}.log
echo "Exploiting DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings_random${LOG_SUFFIX}.log

echo "CollectingExperiences for Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_SIMPLE64} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64_random${LOG_SUFFIX}.log
echo "Training on Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_SIMPLE64} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64_random${LOG_SUFFIX}.log
echo "Exploiting Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_SIMPLE64} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64_random${LOG_SUFFIX}.log
