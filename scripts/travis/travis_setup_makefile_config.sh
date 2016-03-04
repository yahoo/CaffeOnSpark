#!/bin/bash

cd ./caffe-public
./scripts/travis/travis_setup_makefile_config.sh
sed -i -e '/WITH_PYTHON_LAYER/d' Makefile.config
cd ../