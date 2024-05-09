#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export EPSILON_DECAY=0.993

source ${SCRIPT_DIR}/adjusted_reward/collect_minerals/train.sh
source ${SCRIPT_DIR}/adjusted_reward/build_marines/train.sh
source ${SCRIPT_DIR}/adjusted_reward/defeat_zerglings_and_banelings/train.sh
source ${SCRIPT_DIR}/adjusted_reward/simple64/train.sh

