#!/bin/bash

# ./src/scripts/single/dqn/score/simple64/train.sh 2>&1 | tee single_dqn_score_simple64.log
# ./src/scripts/single/dqn/reward/simple64/train.sh 2>&1 | tee single_dqn_reward_simple64.log

# ./src/scripts/multi/dqn/score/full_train.sh 2>&1 | tee multi_dqn_score_simple64.log
# ./src/scripts/multi/dqn/reward/full_train.sh 2>&1 | tee multi_dqn_reward_simple64.log

# ./src/scripts/single/dqn/reward/simple64/train.sh 2>&1 | tee single_dqn_reward_simple64.log

echo "Training multi-agent reward scheme"
./src/scripts/train_multi_agent_reward.sh 2>&1 | tee train_multi_agent_reward.log
echo "Training multi-agent score scheme"
./src/scripts/train_multi_agent_score.sh 2>&1 | tee train_multi_agent_score.log
echo "Training single agent reward scheme"
./src/scripts/train_single_agent_reward.sh 2>&1 | tee train_single_agent_reward.log
echo "Training single agent score scheme"
./src/scripts/train_single_agent_score.sh 2>&1 | tee train_single_agent_score.log
