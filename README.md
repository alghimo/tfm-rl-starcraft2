# tfm-rl-starcraft2

Master thesis - RL - Starcraft II

## Setup

First off, this assumes that you have already setup Starcraft two, as stated in the [PySC2 instructions](https://github.com/google-deepmind/pysc2/tree/master?tab=readme-ov-file#get-starcraft-ii). The work for this project was done under Linux (Ubuntu 22.04) and used the latest [Starcraft II Linux package](https://github.com/Blizzard/s2client-proto#linux-packages) available at the time (4.10).

As for the environment, we've setup Python 3.8, since with newer versions the installation seemed to be broken (I couldn't get pysc2 to work with pygame>=2, and I could only install pygame<2 up until python 3.8).

```bash
conda create -n tfm python=3.8
conda activate tfm
# Go to the project root
cd tfm-rl-starcraft2
# Installing the package will also install pysc2==4.0.0, pygame==1.9.6 and protobuf 3.19.6
pip install -e src/
```

You can now test runnign a random agent with this command:

```bash
python -m pysc2.bin.agent --map CollectMineralShards --feature_screen_size=256 --feature_minimap_size=128
```

This should write a replay, you can look for a line like this in your output:

```bash
I0204 12:00:02.995138 140405932926784 sc2_env.py:736] Wrote replay to: /home/albert/StarCraftII/Replays/RandomAgent/CollectMineralShards_2024-02-04-11-00-02.SC2Replay
```

Now, test that you can indeed show the replay:

```bash
python -m pysc2.bin.play --rgb_screen_size=1600,1200 --replay /home/albert/StarCraftII/Replays/RandomAgent/CollectMineralShards_2024-02-04-11-00-02.SC2Replay
```

## Maps

### List maps

```bash
python -m pysc2.bin.map_list
```

### List mini-games

```bash
python -m pysc2.bin.map_list|grep "mini_games"
```



## Agents

### Random agent

```bash
python -m pysc2.bin.agent --map CollectMineralShards --feature_screen_size=256 --feature_minimap_size=128
```

### Play as a human

```bash
python -m pysc2.bin.play --map CollectMineralShards --feature_screen_size=256 --feature_minimap_size=128
```

### Test agent

```bash
python -m pysc2.bin.agent --map CollectMineralShards --agent tfm_sc2.rl.agents.test_agent.TestAgent
```

```bash
python -m pysc2.bin.agent --map DefeatRoaches --agent tfm_sc2.rl.agents.test_agent.TestAgent
```

```bash
python -m pysc2.bin.agent --map DefeatRoaches --agent tfm_sc2.rl.agents.test_agent.TestAgent --use_feature_units --use_raw_units
```




## Troubleshooting

### `AttributeError: module 'pysc2.lib.replay' has no attribute 'get_replay_version'`

If you get an error like `AttributeError: module 'pysc2.lib.replay' has no attribute 'get_replay_version'`, then you can fix it by copying the contents of `replay.py` into the `replay/__init__.py``. Steps:

- Locate the location of the `pysc2` package in the conda environment

```bash
PYSC2_PKG_PATH=$(python -c "import pysc2; from pathlib import Path; print(Path(pysc2.__file__).parent)")
echo $PYSC2_PKG_PATH
```

- Within that folder copy the contents of `lib/replay.py` into `lib/replay/__init__.py`.

```bash
cat $PYSC2_PKG_PATH/lib/replay.py >> $PYSC2_PKG_PATH/lib/replay/__init__.py
# Show the final contents of the init file
cat $PYSC2_PKG_PATH/lib/replay/__init__.py
```

The replay should work as expected now:

```bash
python -m pysc2.bin.play --rgb_screen_size=1600,1200 --replay /home/albert/StarCraftII/Replays/RandomAgent/CollectMineralShards_2024-02-04-11-00-02.SC2Replay
```



$ python -m pysc2.bin.agent --map CollectMineralShards --agent tfm_sc2.rl.agents.test_agent.TestAgent



python -m pysc2.bin.agent --map CollectMineralShards --feature_screen_size=256 --feature_minimap_size=128

python -m pysc2.bin.play --map CollectMineralShards --rgb_screen_size=1080

python -m pysc2.bin.play -rgb_screen_size=1600,1200 --replay /home/albert/StarCraftII/Replays/RandomAgent/CollectMineralShards_2024-02-04-09-58-26.SC2Replay
