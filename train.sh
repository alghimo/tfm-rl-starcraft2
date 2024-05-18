#!/bin/bash

echo "Training multi-agent reward scheme"
./src/scripts/train_multi_agent_reward.sh 2>&1 | tee train_multi_agent_reward.log
echo "Training single agent reward scheme"
./src/scripts/train_single_agent_reward.sh 2>&1 | tee train_single_agent_reward.log
echo "Training multi-agent score scheme"
./src/scripts/train_multi_agent_score.sh 2>&1 | tee train_multi_agent_score.log
echo "Training single agent score scheme"
./src/scripts/train_single_agent_score.sh 2>&1 | tee train_single_agent_score.log
