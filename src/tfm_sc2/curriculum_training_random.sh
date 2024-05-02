MODEL_ID=random_agent
LOG_DIR="../../models/${MODEL_ID}"
LOG_SUFFIX="_01"
mkdir -p ${LOG_DIR}
EXPLOIT_EPISODES=200

echo "Exploiting CollectMineralsAndGas"
python runner.py --agent_key "single.random" --map_name "CollectMineralsAndGas" --num_episodes ${EXPLOIT_EPISODES} --model_id ${MODEL_ID} 2>&1 | tee ${LOG_DIR}/CollectMineralsAndGas${LOG_SUFFIX}.log

echo "Exploiting BuildMarines"
python runner.py --agent_key "single.random" --map_name "BuildMarines" --num_episodes ${EXPLOIT_EPISODES} --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/BuildMarines_random${LOG_SUFFIX}.log

echo "Exploiting DefeatRoaches"
python runner.py --agent_key "single.random" --map_name "DefeatRoaches" --num_episodes ${EXPLOIT_EPISODES} --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/DefeatRoaches_random${LOG_SUFFIX}.log

echo "Exploiting DefeatZerglingsAndBanelings"
python runner.py --agent_key "single.random" --map_name "DefeatZerglingsAndBanelings" --num_episodes ${EXPLOIT_EPISODES} --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/DefeatZerglingsAndBanelings_random${LOG_SUFFIX}.log

echo "Exploiting Simple64"
python runner.py --agent_key "single.random" --map_name "Simple64" --num_episodes ${EXPLOIT_EPISODES} --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/Simple64_random${LOG_SUFFIX}.log
