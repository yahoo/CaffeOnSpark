#!/bin/bash

: ${HADOOP_PREFIX:=/usr/local/hadoop}

$HADOOP_PREFIX/etc/hadoop/hadoop-env.sh

rm /tmp/*.pid

# installing libraries if any - (resource urls added comma separated to the ACP system variable)
cd $HADOOP_PREFIX/share/hadoop/common ; for cp in ${ACP//,/ }; do  echo == $cp; curl -LO $cp ; done; cd -

# adding necessary paths to environment variables (FIXME: These are already in Dockerfile, but does not work. So giving them explicitly.)
export PATH=$PATH:$SPARK_HOME/bin
export PATH=$PATH:$HADOOP_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CAFFE_ON_SPARK/caffe-public/distribute/lib:$CAFFE_ON_SPARK/caffe-distri/distribute/lib

service ssh start
$HADOOP_PREFIX/sbin/start-dfs.sh
$HADOOP_PREFIX/sbin/start-yarn.sh

if [[ $1 == "-d" ]]; then
  while true; do sleep 1000; done
fi

if [[ $1 == "-bash" ]]; then
  /bin/bash
fi
