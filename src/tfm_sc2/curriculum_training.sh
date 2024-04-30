MODEL_ID=single_drl_simplified_actions
LOG_DIR="../../models/${MODEL_ID}"
LOG_SUFFIX="_02"
mkdir -p ${LOG_DIR}
# Collect experiences in random mode to ensure we collect for all scenarios
echo "Collecting experiences for CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes 10 --random_mode --model_id ${MODEL_ID} 2>&1 | tee ${LOG_DIR}/CollectMineralsAndGas_random${LOG_SUFFIX}.log
echo "Collecting experiences for BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes 10 --random_mode --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/BuildMarines_random${LOG_SUFFIX}.log
echo "Collecting experiences for DefeatRoaches"
python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes 10 --random_mode --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/DefeatRoaches_random${LOG_SUFFIX}.log
echo "Collecting experiences for Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes 10 --random_mode --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/Simple64_random${LOG_SUFFIX}.log

# Train
echo "Training on CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes 50 --model_id ${MODEL_ID} 2>&1 | tee ${LOG_DIR}/CollectMineralsAndGas${LOG_SUFFIX}.log
echo "Training on BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes 50 --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/BuildMarines${LOG_SUFFIX}.log
echo "Training on DefeatRoaches"
python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes 50 --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/DefeatRoaches${LOG_SUFFIX}.log
echo "Training on Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes 25 --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/Simple64${LOG_SUFFIX}.log

# Exploit
echo "Exploiting on CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes 50 --exploit --model_id ${MODEL_ID} 2>&1 | tee ${LOG_DIR}/CollectMineralsAndGas_exploit${LOG_SUFFIX}.log
echo "Exploiting on BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes 50 --exploit --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/BuildMarines_exploit${LOG_SUFFIX}.log
echo "Exploiting on DefeatRoaches"
python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes 50 --exploit --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/DefeatRoaches_exploit${LOG_SUFFIX}.log
echo "Exploiting on Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes 25 --exploit --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/Simple64_exploit${LOG_SUFFIX}.log

# Exploit - Random agent
echo "Exploiting Random Agent on CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes 50 --random_mode --exploit --model_id ${MODEL_ID} 2>&1 | tee ${LOG_DIR}/CollectMineralsAndGas_exploit${LOG_SUFFIX}.log
echo "Exploiting Random Agent on BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes 50 --random_mode --exploit --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/BuildMarines_exploit${LOG_SUFFIX}.log
echo "Exploiting Random Agent on DefeatRoaches"
python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes 50 --random_mode --exploit --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/DefeatRoaches_exploit${LOG_SUFFIX}.log
echo "Exploiting Random Agent on Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes 25 --random_mode --exploit --model_id ${MODEL_ID} 2>&1| tee ${LOG_DIR}/Simple64_exploit${LOG_SUFFIX}.log