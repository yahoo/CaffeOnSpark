#!/bin/bash
# This script must be run with sudo.
# for libopencv-dev-2.4
add-apt-repository -y ppa:kubuntu-ppa/backports
# for proto2.5
add-apt-repository -y ppa:chris-lea/protobuf
apt-get -y update
./caffe-public/scripts/travis/travis_install.sh