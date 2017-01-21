# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
#
# This file is the dockerfile to setup caffeonspark cpu standalone version.

FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        vim \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy \
        maven \
        unzip \
        zip \
        unzip \
        libopenblas-dev \
        openssh-server \
        openssh-client \
        libopenblas-dev \
        libboost-all-dev \
        openjdk-8-jdk

RUN rm -rf /var/lib/apt/lists/*


# Passwordless SSH
RUN ssh-keygen -y -q -N "" -t dsa -f /etc/ssh/ssh_host_dsa_key
RUN ssh-keygen -y -q -N "" -t rsa -f /etc/ssh/ssh_host_rsa_key
RUN ssh-keygen -q -N "" -t rsa -f /root/.ssh/id_rsa
RUN cp /root/.ssh/id_rsa.pub ~/.ssh/authorized_keys


# Apache Hadoop and Spark section
RUN wget http://apache.mirrors.tds.net/hadoop/common/hadoop-2.6.4/hadoop-2.6.4.tar.gz
RUN wget http://archive.apache.org/dist/spark/spark-1.6.0/spark-1.6.0-bin-hadoop2.6.tgz

RUN gunzip hadoop-2.6.4.tar.gz
RUN gunzip spark-1.6.0-bin-hadoop2.6.tgz
RUN tar -xf hadoop-2.6.4.tar
RUN tar -xf spark-1.6.0-bin-hadoop2.6.tar

RUN sudo cp -r hadoop-2.6.4 /usr/local/hadoop
RUN sudo cp -r spark-1.6.0-bin-hadoop2.6 /usr/local/spark

RUN rm hadoop-2.6.4.tar spark-1.6.0-bin-hadoop2.6.tar
RUN rm -rf hadoop-2.6.4/ spark-1.6.0-bin-hadoop2.6/

RUN sudo mkdir -p /usr/local/hadoop/hadoop_data/hdfs/namenode
RUN sudo mkdir -p /usr/local/hadoop/hadoop_data/hdfs/datanode

# Environment variables
ENV JAVA_HOME /usr/lib/jvm/java-1.8.0-openjdk-amd64
ENV HADOOP_HOME=/usr/local/hadoop
ENV SPARK_HOME=/usr/local/spark
ENV PATH $PATH:$JAVA_HOME/bin
ENV PATH $PATH:$HADOOP_HOME/bin
ENV PATH $PATH:$HADOOP_HOME/sbin
ENV PATH $PATH:$SPARK_HOME/bin
ENV PATH $PATH:$SPARK_HOME/sbin
ENV HADOOP_MAPRED_HOME /usr/local/hadoop
ENV HADOOP_COMMON_HOME /usr/local/hadoop
ENV HADOOP_HDFS_HOME /usr/local/hadoop
ENV HADOOP_CONF_DIR /usr/local/hadoop/etc/hadoop
ENV YARN_CONF_DIR /usr/local/hadoop/etc/hadoop
ENV YARN_HOME /usr/local/hadoop
ENV HADOOP_COMMON_LIB_NATIVE_DIR /usr/local/hadoop/lib/native
ENV HADOOP_OPTS "-Djava.library.path=$HADOOP_HOME/lib"

# Clone CaffeOnSpark
ENV CAFFE_ON_SPARK=/opt/CaffeOnSpark
WORKDIR $CAFFE_ON_SPARK
RUN git clone https://github.com/yahoo/CaffeOnSpark.git . --recursive

# Some of the Hadoop part extracted from "https://hub.docker.com/r/sequenceiq/hadoop-docker/~/dockerfile/"
RUN mkdir $HADOOP_HOME/input
RUN cp $HADOOP_HOME/etc/hadoop/*.xml $HADOOP_HOME/input
RUN cd /usr/local/hadoop/input

# Copy .xml files.
RUN cp ${CAFFE_ON_SPARK}/scripts/*.xml  ${HADOOP_HOME}/etc/hadoop

# Format namenode and finish hadoop, spark installations.
RUN $HADOOP_HOME/bin/hdfs namenode -format

RUN ls /root/.ssh/
ADD config/ssh_config /root/.ssh/config
RUN chmod 600 /root/.ssh/config
RUN chown root:root /root/.ssh/config

ADD config/bootstrap.sh /etc/bootstrap.sh
RUN chown root:root /etc/bootstrap.sh
RUN chmod 700 /etc/bootstrap.sh

ENV BOOTSTRAP /etc/bootstrap.sh

RUN sed -i '/^export JAVA_HOME/ s:.*:export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64\nexport HADOOP_HOME=/usr/local/hadoop\n:' $HADOOP_HOME/etc/hadoop/hadoop-env.sh
RUN sed -i '/^export HADOOP_CONF_DIR/ s:.*:export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop/:' $HADOOP_HOME/etc/hadoop/hadoop-env.sh

# workingaround docker.io build error
RUN ls -la /usr/local/hadoop/etc/hadoop/*-env.sh
RUN chmod +x /usr/local/hadoop/etc/hadoop/*-env.sh
RUN ls -la /usr/local/hadoop/etc/hadoop/*-env.sh

# fix the 254 error code
RUN sed  -i "/^[^#]*UsePAM/ s/.*/#&/"  /etc/ssh/sshd_config
RUN echo "UsePAM no" >> /etc/ssh/sshd_config
RUN echo "Port 2122" >> /etc/ssh/sshd_config

RUN service ssh start && $HADOOP_HOME/etc/hadoop/hadoop-env.sh && $HADOOP_HOME/sbin/start-dfs.sh && $HADOOP_HOME/bin/hdfs dfs -mkdir -p /user/root
RUN service ssh start && $HADOOP_HOME/etc/hadoop/hadoop-env.sh && $HADOOP_HOME/sbin/start-dfs.sh && $HADOOP_HOME/bin/hdfs dfs -put $HADOOP_HOME/etc/hadoop/ input

CMD ["/etc/bootstrap.sh", "-bash"]

# Hdfs ports
EXPOSE 50010 50020 50070 50075 50090 8020 9000
# Mapred ports
EXPOSE 10020 19888
#Yarn ports
EXPOSE 8030 8031 8032 8033 8040 8042 8088
#Other ports
EXPOSE 49707 2122

# Continue with CaffeOnSpark build.
# ENV CAFFE_ON_SPARK=/opt/CaffeOnSpark
WORKDIR $CAFFE_ON_SPARK
# RUN git clone https://github.com/yahoo/CaffeOnSpark.git . --recursive
RUN cp caffe-public/Makefile.config.example caffe-public/Makefile.config
RUN echo "INCLUDE_DIRS += ${JAVA_HOME}/include" >> caffe-public/Makefile.config
#RUN sed -i "s/# USE_CUDNN := 1/USE_CUDNN := 1/g" caffe-public/Makefile.config
RUN sed -i "s|BLAS := atlas|BLAS := open|g" caffe-public/Makefile.config

RUN make build

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$CAFFE_ON_SPARK/caffe-public/distribute/lib:$CAFFE_ON_SPARK/caffe-distri/distribute/lib

WORKDIR /root
