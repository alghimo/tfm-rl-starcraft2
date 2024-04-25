TODO

- Move save, tracker, stats, etc to base agent

- Keep track of what workers are harvesting or find a way to check it
- Add action to move worker harvesting minerals to gas
- Add action to move worker harvesting gas to minerals
- "Free" positions for command centers, barracks, supply depots when one is destroyed, so they can be built again
- Add penalties / rewards for losing units or killing enemy units
- Allow SCVs to attack?
- Manage stats per map



```bash
python runner.py --agent_key "single.dqn" --map_name "CollectMineralsAndGas" --num_episodes 20 --model_id single_drl| tee CollectMineralsAndGas.log
python runner.py --agent_key "single.dqn" --map_name "BuildMarines" --num_episodes 20 --model_id single_drl| tee BuildMarines.log
python runner.py --agent_key "single.dqn" --map_name "DefeatRoaches" --num_episodes 20 --model_id single_drl| tee DefeatRoaches.log
python runner.py --agent_key "single.dqn" --map_name "Simple64" --num_episodes 20 --model_id single_drl| tee Simple64.log
```