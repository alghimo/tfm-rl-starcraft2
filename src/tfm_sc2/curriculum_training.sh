MODEL_ID=single_drl
MODEL_ID_RANDOM=random_agent
MODEL_ID_COLLECT_MINERALS=${MODEL_ID}_collect_minerals
MODEL_ID_BUILD_MARINES=${MODEL_ID}_build_marines
MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS=${MODEL_ID}_defeat_zerglings_and_banelings
MODEL_ID_DEFEAT_ROACHES=${MODEL_ID}_defeat_roaches
MODEL_ID_SIMPLE64=${MODEL_ID}_simple64
LOG_DIR="../../models/${MODEL_ID}"
LOG_DIR_RANDOM="../../models/${MODEL_ID_RANDOM}"
LOG_DIR_COLLECT_MINERALS="../../models/${MODEL_ID_COLLECT_MINERALS}"
LOG_DIR_BUILD_MARINES="../../models/${MODEL_ID_BUILD_MARINES}"
LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS="../../models/${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS}"
LOG_DIR_DEFEAT_ROACHES="../../models/${MODEL_ID_DEFEAT_ROACHES}"
LOG_DIR_SIMPLE64="../../models/${MODEL_ID_SIMPLE64}"
LOG_SUFFIX="_02"
mkdir -p ${LOG_DIR} ${LOG_DIR_RANDOM} ${LOG_DIR_COLLECT_MINERALS} ${LOG_DIR_BUILD_MARINES} ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS} ${LOG_DIR_SIMPLE64}
BURNIN_EPISODES=10
TRAIN_EPISODES=100
TRAIN_EPISODES_SIMPLE64=50
EXPLOIT_EPISODES=50
EXPLOIT_EPISODES_SIMPLE64=25

#Quick iterations
# BURNIN_EPISODES=1
# TRAIN_EPISODES=3
# TRAIN_EPISODES_SIMPLE64=1
# EXPLOIT_EPISODES=3
# EXPLOIT_EPISODES_SIMPLE64=1

echo "Collecting experiences for CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID} 2>&1 | tee ${LOG_DIR}/CollectMineralsAndGas_random${LOG_SUFFIX}.log

echo "Training on CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID} 2>&1 | tee ${LOG_DIR}/CollectMineralsAndGas${LOG_SUFFIX}.log


echo "Collecting experiences for BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/BuildMarines_random${LOG_SUFFIX}.log
echo "Training on BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/BuildMarines${LOG_SUFFIX}.log


# echo "Collecting experiences for DefeatRoaches"
# python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/DefeatRoaches_random${LOG_SUFFIX}.log
# echo "Training on DefeatRoaches"
# python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/DefeatRoaches${LOG_SUFFIX}.log

echo "Collecting experiences for DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/DefeatZerglingsAndBanelings_random${LOG_SUFFIX}.log
echo "Training on DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/DefeatZerglingsAndBanelings${LOG_SUFFIX}.log


echo "Collecting experiences for Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/Simple64_random${LOG_SUFFIX}.log

echo "Training on Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${TRAIN_EPISODES_SIMPLE64} --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/Simple64${LOG_SUFFIX}.log



# Collect experiences in random mode to ensure we collect for all scenarios
echo "Standalone - Collecting experiences for CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_COLLECT_MINERALS} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas_random${LOG_SUFFIX}.log

echo "Training on CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_COLLECT_MINERALS} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas${LOG_SUFFIX}.log


echo "Collecting experiences for BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_BUILD_MARINES} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines_random${LOG_SUFFIX}.log
echo "Training on BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_BUILD_MARINES} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines${LOG_SUFFIX}.log


# echo "Collecting experiences for DefeatRoaches"
# python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_DEFEAT_ROACHES} 2>&1| tee ${LOG_DIR_DEFEAT_ROACHES}/DefeatRoaches_random${LOG_SUFFIX}.log
# echo "Training on DefeatRoaches"
# python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_DEFEAT_ROACHES} 2>&1| tee ${LOG_DIR_DEFEAT_ROACHES}/DefeatRoaches${LOG_SUFFIX}.log

echo "Collecting experiences for DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings_random${LOG_SUFFIX}.log
echo "Training on DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${TRAIN_EPISODES} --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings${LOG_SUFFIX}.log

echo "Collecting experiences for Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${BURNIN_EPISODES} --random_mode --model_id ${MODEL_ID_SIMPLE64} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64_random${LOG_SUFFIX}.log

echo "Training on Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${TRAIN_EPISODES_SIMPLE64} --model_id ${MODEL_ID_SIMPLE64} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64${LOG_SUFFIX}.log

# Exploit full agent

# Exploit
echo "Exploiting on CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID} 2>&1 | tee ${LOG_DIR}/CollectMineralsAndGas_exploit${LOG_SUFFIX}.log
echo "Exploiting on BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/BuildMarines_exploit${LOG_SUFFIX}.log
# echo "Exploiting on DefeatRoaches"
# python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/DefeatRoaches_exploit${LOG_SUFFIX}.log
echo "Exploiting on DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/DefeatZerglingsAndBanelings_exploit${LOG_SUFFIX}.log

echo "Exploiting on Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${EXPLOIT_EPISODES_SIMPLE64} --exploit --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/Simple64_exploit${LOG_SUFFIX}.log

# Exploit single agents
echo "Exploiting on CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_COLLECT_MINERALS} 2>&1 | tee ${LOG_DIR_COLLECT_MINERALS}/CollectMineralsAndGas_exploit${LOG_SUFFIX}.log
echo "Exploiting on BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_BUILD_MARINES} 2>&1| tee ${LOG_DIR_BUILD_MARINES}/BuildMarines_exploit${LOG_SUFFIX}.log
# echo "Exploiting on DefeatRoaches"
# python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatRoaches_exploit${LOG_SUFFIX}.log
echo "Exploiting on DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.dqn" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${EXPLOIT_EPISODES} --exploit --model_id ${MODEL_ID_DEFEAT_ZERGLINGS_AND_BANELINGS} 2>&1| tee ${LOG_DIR_DEFEAT_ZERGLINGS_AND_BANELINGS}/DefeatZerglingsAndBanelings_exploit${LOG_SUFFIX}.log

echo "Exploiting on Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes ${EXPLOIT_EPISODES_SIMPLE64} --exploit --model_id ${MODEL_ID_SIMPLE64} 2>&1| tee ${LOG_DIR_SIMPLE64}/Simple64_exploit${LOG_SUFFIX}.log

# Exploit - Random agent
echo "Exploiting Random Agent on CollectMineralsAndGas"
python runner.py --agent_key "single.random" --map_name "CollectMineralsAndGas" --num_episodes ${EXPLOIT_EPISODES} --random_mode --exploit --model_id ${MODEL_ID_RANDOM} 2>&1 | tee ${LOG_DIR_RANDOM}/CollectMineralsAndGas_exploit_random${LOG_SUFFIX}.log
echo "Exploiting Random Agent on BuildMarines"
python runner.py --agent_key "single.random" --map_name "BuildMarines" --num_episodes ${EXPLOIT_EPISODES} --random_mode --exploit --model_id ${MODEL_ID_RANDOM} 2>&1| tee ${LOG_DIR_RANDOM}/BuildMarines_exploit_random${LOG_SUFFIX}.log
# echo "Exploiting Random Agent on DefeatRoaches"
# python runner.py --agent_key "single.random" --map_name "DefeatRoaches" --num_episodes ${EXPLOIT_EPISODES} --random_mode --exploit --model_id ${MODEL_ID_RANDOM} 2>&1| tee ${LOG_DIR_RANDOM}/DefeatRoaches_exploit_random${LOG_SUFFIX}.log
echo "Exploiting Random Agent on DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.random" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${EXPLOIT_EPISODES} --random_mode --exploit --model_id ${MODEL_ID_RANDOM} 2>&1| tee ${LOG_DIR_RANDOM}/DefeatZerglingsAndBanelings_exploit_random${LOG_SUFFIX}.log

echo "Exploiting Random Agent on Simple64"
python runner.py --agent_key "single.random" --map_name "Simple64" --num_episodes ${EXPLOIT_EPISODES_SIMPLE64} --random_mode --exploit --model_id ${MODEL_ID_RANDOM} 2>&1| tee ${LOG_DIR_RANDOM}/Simple64_exploit_random${LOG_SUFFIX}.log