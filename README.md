<!--
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
-->
# CaffeOnSpark

## What's CaffeOnSpark?

CaffeOnSpark brings deep learning onto Hadoop/Spark clusters.  By
combining salient features from deep learning framework
[Caffe](https://github.com/BVLC/caffe) and big-data framework [Apache
Spark](http://spark.apache.org/), CaffeOnSpark enables distributed
deep learning on a cluster of GPU and CPU servers.

As a distributed extension of Caffe, CaffeOnSpark supports neural
network model training, testing and feature extraction.  Caffe users
could now perform distributed learning using their existing LMDB data
files other format) and minorly adjusted network configuration (as
[illustrated](../master/data/lenet_memory_train_test.prototxt#L10-L12)).

CaffeOnSpark is a Spark package for deep learning. It is complementary
to non-deep learning libraries MLlib and and Spark SQL.
CaffeOnSpark's Scala API provides Spark applications with an easy
mechanism to invoke deep learning (see
[sample](../master/caffe-grid/src/main/scala/com/yahoo/ml/caffe/examples/MyMLPipeline.scala))
over distributed datasets.

CaffeOnSpark was developed by Yahoo for [large-scale distributed deep
learning on our Hadoop
clusters](http://yahoohadoop.tumblr.com/post/129872361846/large-scale-distributed-deep-learning-on-hadoop)
in Yahoo's private cloud.  It's been used by Yahoo for photo search,
content classification and so on.

## Why CaffeOnSpark?

CaffeOnSpark provides some important benefits over alternative deep learning solutions.

* It enables model training, test and feature extraction directly upon Hadoop datasets.
* It turns your Hadoop or Spark clusters into a powerful platform for deep learning, without setting up new clusters.
* Server-to-server direct communication (ethernet or infiniband) achieves speedy learning, and eliminates scalability bottleneck. 
* Caffe users' existing datasets (ex. LMDB) and configurations could be applied for distributed learning without conversions.
* High-level API empowers Spark applications to easily conduct deep learnings. 
* Incremental learning is supported to leverage previously trained model or snapshots. 
* Additional data formats and network interfaces could be easily added.
* It's easily deployabe at public clouds (ex. AWS EC2) and private cloud.

## Documentations

CaffeOnSpark [wiki site](../../wiki) for detailed documentations including:
* [Build instruction](../../wiki/build)
* Get Started guides
 * [Start CaffeOnSpark on standalone cluster](../wiki/GetStarted_local)
 * [Start CaffeOnSpark on Amazon EC2](../../wiki/GetStarted_EC2) using a pre-built Amazon machine image (AMI). 
* References
 * [API Reference](http://yahoo.github.io/CaffeOnSpark/scala_doc/)
 * [CLI Reference](../../wiki/CLI)
* [Create your own CaffeOnSpark AMI](../../wiki/Create_AMI)

## Remarks

* Batch sizes specified in prototxt files are per device.
* Memory layers should not be shared among GPUs, and thus "share_in_parallel: false" is required for layer configuration.

## Mailing List

Please join [CaffeOnSpark user
group](https://groups.google.com/forum/#!forum/caffeonspark-users) for
discussions and questions.


## License

The use and distribution terms for this software are covered by the
Apache 2.0 license. See [LICENSE](LICENSE.txt) file for terms.
