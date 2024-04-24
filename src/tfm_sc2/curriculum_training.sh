echo "Training on CollectMineralsAndGas"
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes 100 --model_id single_drl 2>&1 | tee CollectMineralsAndGas.log
echo "Training on BuildMarines"
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes 100 --model_id single_drl 2>&1| tee BuildMarines.log
echo "Training on DefeatRoaches"
python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes 100 --model_id single_drl 2>&1| tee DefeatRoaches.log
echo "Training on Simple64"
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes 10 --model_id single_drl 2>&1| tee Simple64.log

python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes 10 --exploit --model_id single_drl 2>&1| tee Simple64_exploit.log