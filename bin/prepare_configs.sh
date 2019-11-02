#!/usr/bin/env bash

if [[ -z "$LOGDIR" ]]
then
      LOGDIR="."
fi


date=$(date +%y%m%d-%H%M)
mkdir -p ${LOGDIR}/logs
mkdir -p ${LOGDIR}/${date}-mongodb
mkdir -p ${LOGDIR}/${date}-mongodb-ppo


if [[ "$(uname)" == "Darwin" ]]; then
    sed -i ".bak" "s/dbPath: .*/dbPath: ${LOGDIR//\//\\/}\/$date-mongodb/g" ./configs/_mongod.conf
    sed -i ".bak" "s/path: .*/path: ${LOGDIR//\//\\/}\/$date-mongo.log/g" ./configs/_mongod.conf
    sed -i ".bak" "s/dbPath: .*/dbPath: ${LOGDIR//\//\\/}\/$date-mongodb-ppo/g" ./configs/mongod_ppo.conf
    sed -i ".bak" "s/path: .*/path: ${LOGDIR//\//\\/}\/$date-mongo-ppo.log/g" ./configs/mongod_ppo.conf

    agent=ddpg
    sed -i ".bak" "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/$date-$agent/g" ./configs/"$agent".yml
    agent=sac
    sed -i ".bak" "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/$date-$agent/g" ./configs/"$agent".yml
    agent=td3
    sed -i ".bak" "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/$date-$agent/g" ./configs/"$agent".yml
    agent=ppo
    sed -i ".bak" "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/$date-$agent/g" ./configs/"$agent".yml
elif [[ "$(expr substr $(uname -s) 1 5)" == "Linux" ]]; then
    sed -i "s/dbPath: .*/dbPath: ${LOGDIR//\//\\/}\/$date-mongodb/g" ./configs/_mongod.conf
    sed -i "s/path: .*/path: ${LOGDIR//\//\\/}\/$date-mongo.log/g" ./configs/_mongod.conf
    sed -i "s/dbPath: .*/dbPath: ${LOGDIR//\//\\/}\/$date-mongodb-ppo/g" ./configs/mongod_ppo.conf
    sed -i "s/path: .*/path: ${LOGDIR//\//\\/}\/$date-mongo-ppo.log/g" ./configs/mongod_ppo.conf

    agent=ddpg
    sed -i "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/$date-$agent/g" ./configs/"$agent".yml
    agent=sac
    sed -i "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/$date-$agent/g" ./configs/"$agent".yml
    agent=td3
    sed -i "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/$date-$agent/g" ./configs/"$agent".yml
    agent=ppo
    sed -i "s/logdir: .*/logdir: ${LOGDIR//\//\\/}\/logs\/$date-$agent/g" ./configs/"$agent".yml
fi
