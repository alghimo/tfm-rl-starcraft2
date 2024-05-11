#!/bin/bash

ADJUSTED_REWARDS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DQN_DIR=$(realpath $(dirname ${DQN_CONSTANTS_DIR}))
source ${DQN_DIR}/constants.sh

export REWARD_METHOD="adjusted_reward"