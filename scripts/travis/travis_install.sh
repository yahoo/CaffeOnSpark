#!/bin/bash
# This script must be run with sudo.
# for libopencv-dev-2.4
add-apt-repository -y ppa:kubuntu-ppa/backports
# for proto2.5
add-apt-repository -y  ppa:chris-lea/protobuf
apt-get -y --force-yes update
# remove CONDA directory
rm -rf $CONDA_DIR
# invoke BVLC caffe scripts
./caffe-public/scripts/travis/travis_install.sh
export CAFFE_ON_SPARK=$(pwd)
${CAFFE_ON_SPARK}/scripts/local-setup-hadoop.sh 2>&1 > /dev/null
export HADOOP_HOME=$(pwd)/hadoop-2.6.4
export PATH=${HADOOP_HOME}/bin:${PATH}
${CAFFE_ON_SPARK}/scripts/local-setup-spark.sh 2>&1 > /dev/null
export SPARK_HOME=$(pwd)/spark-1.6.0-bin-hadoop2.6
export PATH=${SPARK_HOME}/bin:${PATH}
