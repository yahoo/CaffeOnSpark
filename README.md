<!--
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
-->
### Note: we're lovingly marking this project as Archived since we're no longer supporting it. You are welcome to read the code and fork your own version of it and continue to use this code under the terms of the project license.

# CaffeOnSpark

## What's CaffeOnSpark?

CaffeOnSpark brings deep learning to Hadoop and Spark clusters.  By
combining salient features from deep learning framework
[Caffe](https://github.com/BVLC/caffe) and big-data frameworks [Apache
Spark](http://spark.apache.org/) and [Apache Hadoop](http://hadoop.apache.org/), CaffeOnSpark enables distributed
deep learning on a cluster of GPU and CPU servers.

As a distributed extension of Caffe, CaffeOnSpark supports neural
network model training, testing, and feature extraction.  Caffe users
can now perform distributed learning using their existing LMDB data
files and minorly adjusted network configuration (as
[illustrated](../master/data/lenet_memory_train_test.prototxt#L10-L12)).

CaffeOnSpark is a Spark package for deep learning. It is complementary
to non-deep learning libraries MLlib and Spark SQL.
CaffeOnSpark's Scala API provides Spark applications with an easy
mechanism to invoke deep learning (see
[sample](../master/caffe-grid/src/main/scala/com/yahoo/ml/caffe/examples/MyMLPipeline.scala))
over distributed datasets.

CaffeOnSpark was developed by Yahoo for [large-scale distributed deep
learning on our Hadoop
clusters](http://yahoohadoop.tumblr.com/post/129872361846/large-scale-distributed-deep-learning-on-hadoop)
in Yahoo's private cloud.  It's been in use by Yahoo for image search,
content classification and several other use cases.

## Why CaffeOnSpark?

CaffeOnSpark provides some important benefits (see [our blog](http://yahoohadoop.tumblr.com/post/139916563586/caffeonspark-open-sourced-for-distributed-deep)) over alternative deep learning solutions.  

* It enables model training, test and feature extraction directly on Hadoop datasets stored in HDFS on Hadoop clusters.
* It turns your Hadoop or Spark cluster(s) into a powerful platform for deep learning, without the need to set up a new dedicated cluster for deep learning separately.
* Server-to-server direct communication (Ethernet or InfiniBand) achieves faster learning and eliminates scalability bottleneck. 
* Caffe users' existing datasets (e.g. LMDB) and configurations could be applied for distributed learning without any conversion needed.
* High-level API empowers Spark applications to easily conduct deep learning. 
* Incremental learning is supported to leverage previously trained models or snapshots. 
* Additional data formats and network interfaces could be easily added.
* It can be easily deployed on public cloud (ex. AWS EC2) or a private cloud.

## Using CaffeOnSpark

Please check CaffeOnSpark [wiki site](../../wiki) for detailed
documentations such as [building instruction](../../wiki/build), [API
reference](http://yahoo.github.io/CaffeOnSpark/scala_doc/#com.yahoo.ml.caffe.package)
and getting started guides for [standalone
cluster](../../wiki/GetStarted_local) and [AWS EC2
cluster](../../wiki/GetStarted_EC2).


* Batch sizes specified in prototxt files are per device.
* Memory layers should not be shared among GPUs, and thus "share_in_parallel: false" is required for layer configuration.

## Building for Spark 2.X

CaffeOnSpark supports both Spark 1.x and 2.x. For Spark 2.0, our default settings are:
  - spark-2.0.0
  - hadoop-2.7.1
  - scala-2.11.7
You may want to adjust them in caffe-grid/pom.xml.

 
## Mailing List

Please join [CaffeOnSpark user
group](https://groups.google.com/forum/#!forum/caffeonspark-users) for
discussions and questions.


## License

The use and distribution terms for this software are covered by the
Apache 2.0 license. See [LICENSE](LICENSE.txt) file for terms.
