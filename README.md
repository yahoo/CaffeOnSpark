# CaffeOnSpark

## Run caffe on grid:

To run caffe on Jet-Blue grid, one needs to install the ycaffe package on the gateway machine, set up the environmental variables and launch jobs with spark-submit. We detail the steps in the below.

#####(1) Install ycaffe package

    kinit ${USER}@Y.CORP.YAHOO.COM
    export YROOT=~/y
    export PATH=${SPARK_HOME}/bin:${PATH}
    yinst install ycaffe -r ${YROOT} -nosudo -br nightly

#####(2) build all the required libraries into a tarball. This step is needed every time a new version is intalled.

    mkdir -p ${HOME}/tmp 
    rm -f ${HOME}/tmp/caffe_on_grid_archive.tar
    pushd ${YROOT}
    tar -cpzf ${HOME}/tmp/caffe_on_grid_archive.tgz lib64
    popd  
    
#####(3) Generate a sequence file containing all the images, needed for next step.

    pushd ${YROOT}/share/caffe
    hadoop fs -mkdir images
    hadoop fs -put images/*.jpg images
    hadoop fs -put images/labels.txt images
    export QUEUE=default
    hadoop fs -rm -r -f image_sequence_file
    spark-submit --master yarn --deploy-mode cluster --queue ${QUEUE} \
        --class com.yahoo.ml.caffe.tools.Binary2Sequence  \
          ${YROOT}/share/caffe/lib/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
            -output image_sequence_file  \
            -imageRoot hdfs:///user/${USER}/images/ \
            -labelFile hdfs:///user/${USER}/images/labels.txt
    hadoop fs -ls image_sequence_file
    hadoop fs -rm -r images
    popd
    
#####(4) Train image dataset with multiple GPUs.

    pushd ${YROOT}/share/caffe
    export QUEUE=gpu
    hadoop fs -rm sample_images.model*
    spark-submit --master yarn --deploy-mode cluster --queue ${QUEUE} \
        --files caffenet_train_solver_GPU.prototxt,caffenet_train_net.prototxt \
        --num-executors 2  \
        --executor-memory 38g --conf spark.yarn.executor.memoryOverhead=16384 \
        --archives ${HOME}/tmp/caffe_on_grid_archive.tgz \
        --conf spark.task.maxFailures=0 \
        --conf spark.speculation=false \
        --conf spark.scheduler.maxRegisteredResourcesWaitingTime=10m \
        --conf spark.driver.extraLibraryPath="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --class com.yahoo.ml.caffe.CaffeOnSpark  \
            ${YROOT}/share/caffe/lib/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
            -train -persistent \
            -conf caffenet_train_solver_GPU.prototxt \
            -model hdfs:///user/${USER}/sample_images.model \
            -devices 2
    hadoop fs -ls sample_images.model*
    popd
    
#####(5) Test image dataset with multiple GPUs.

    pushd ${YROOT}/share/caffe
    export QUEUE=gpu
    hadoop fs -rm image_test_result
    spark-submit --master yarn --deploy-mode cluster --queue ${QUEUE} \
        --files caffenet_train_solver_GPU.prototxt,caffenet_train_net.prototxt \
        --num-executors 2  \
        --executor-memory 38g --conf spark.yarn.executor.memoryOverhead=16384 \
        --archives ${HOME}/tmp/caffe_on_grid_archive.tgz \
        --conf spark.task.maxFailures=0 \
        --conf spark.speculation=false \
        --conf spark.scheduler.maxRegisteredResourcesWaitingTime=10m \
        --conf spark.driver.extraLibraryPath="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --class com.yahoo.ml.caffe.CaffeOnSpark  \
            ${YROOT}/share/caffe/lib/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
            -test \
            -conf caffenet_train_solver_GPU.prototxt \
            -model hdfs:///user/${USER}/sample_images.model \
            -output hdfs:///user/${USER}/image_test_result \
            -devices 2
    hadoop fs -cat image_test_result
    popd

#####(6) Feature extraction with multiple GPUs.

    pushd ${YROOT}/share/caffe
    export QUEUE=gpu
    hadoop fs -rm -r -f image_feature_result
    spark-submit --master yarn --deploy-mode cluster --queue ${QUEUE} \
        --files caffenet_train_solver_GPU.prototxt,caffenet_train_net.prototxt \
        --num-executors 2  \
        --executor-memory 38g --conf spark.yarn.executor.memoryOverhead=16384 \
        --archives ${HOME}/tmp/caffe_on_grid_archive.tgz \
        --conf spark.task.maxFailures=0 \
        --conf spark.speculation=false \
        --conf spark.scheduler.maxRegisteredResourcesWaitingTime=10m \
        --conf spark.driver.extraLibraryPath="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --class com.yahoo.ml.caffe.CaffeOnSpark  \
            ${YROOT}/share/caffe/lib/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
            -features fc7,fc8 \
            -conf caffenet_train_solver_GPU.prototxt \
            -model hdfs:///user/${USER}/sample_images.model \
            -output hdfs:///user/${USER}/image_feature_result \
            -outputFormat json \
            -devices 2
    hadoop fs -cat image_feature_result/*
    popd

#####(7) DL training + feature extraction + MLlib training in a single program

    pushd ${YROOT}/share/caffe
    export QUEUE=gpu
    hadoop fs -rm sample_images.model*
    hadoop fs -rm -r -f image_classifier_model
    spark-submit --master yarn --deploy-mode cluster --queue ${QUEUE} \
        --files caffenet_train_solver_GPU.prototxt,caffenet_train_net.prototxt \
        --num-executors 2  \
        --executor-memory 38g --conf spark.yarn.executor.memoryOverhead=16384 \
        --archives ${HOME}/tmp/caffe_on_grid_archive.tgz \
        --conf spark.task.maxFailures=0 \
        --conf spark.speculation=false \
        --conf spark.scheduler.maxRegisteredResourcesWaitingTime=10m \
        --conf spark.driver.extraLibraryPath="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --class com.yahoo.ml.caffe.examples.MyMLPipeline  \
            ${YROOT}/share/caffe/lib/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
            -persistent \
            -features fc8 \
            -label label \
            -conf caffenet_train_solver_GPU.prototxt \
            -model hdfs:///user/${USER}/sample_images.model  \
            -output hdfs:///user/${USER}/image_classifier_model \
            -devices 2
    hadoop fs -ls sample_images.model*
    hadoop fs -ls image_classifier_model/*
    popd

#####(8) Train VW dataset with multiple GPUs.

    pushd ${YROOT}/share/caffe/vw
    hadoop fs -mkdir txt
    hadoop fs -put txt/*.vw txt
    export QUEUE=gpu
    hadoop fs -rm vw.model*
    spark-submit --master yarn --deploy-mode cluster --queue ${QUEUE} \
        --files VW_java_solver_GPU.prototxt,VW_java_net_GPU.prototxt \
        --archives ${HOME}/tmp/caffe_on_grid_archive.tgz \
        --conf spark.task.maxFailures=0 \
        --conf spark.speculation=false \
        --num-executors 2  \
         --executor-memory 38g --conf spark.yarn.executor.memoryOverhead=16384 \
        --conf spark.driver.extraLibraryPath="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --class com.yahoo.ml.caffe.CaffeOnSpark  \
          ${YROOT}/share/caffe/lib/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
            -train \
            -conf VW_java_solver_GPU.prototxt \
            -devices 2 \
            -model hdfs:///user/${USER}/vw.model
    hadoop fs -ls vw.model*
    popd

#####(9) Train MNIST with multiple GPUs.

    pushd ${YROOT}/share/caffe
    hadoop fs -rm mnist_lenet*
    export QUEUE=gpu
    spark-submit --master yarn --deploy-mode cluster --queue ${QUEUE} \
        --files lenet_memory_train_test.prototxt,lenet_memory_solver.prototxt \
        --archives ${HOME}/tmp/caffe_on_grid_archive.tgz \
        --conf spark.task.maxFailures=0 \
        --conf spark.speculation=false \
        --num-executors 2  \
         --executor-memory 38g --conf spark.yarn.executor.memoryOverhead=16384 \
        --conf spark.driver.extraLibraryPath="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --class com.yahoo.ml.caffe.CaffeOnSpark  \
          ${YROOT}/share/caffe/lib/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
            -train \
            -conf lenet_memory_solver.prototxt \
            -devices 2 \
            -model hdfs:///user/${USER}/mnist_lenet.model
    hadoop fs -ls mnist_lenet*
    popd

#####(10) Test MNIST with multiple GPUs.

    pushd ${YROOT}/share/caffe
    hadoop fs -rm lenet_accuracy_result
    export QUEUE=gpu
    spark-submit --master yarn --deploy-mode cluster --queue ${QUEUE} \
        --files lenet_memory_train_test.prototxt,lenet_memory_solver.prototxt \
        --archives ${HOME}/tmp/caffe_on_grid_archive.tgz \
        --conf spark.task.maxFailures=0 \
        --conf spark.speculation=false \
        --num-executors 2  \
         --executor-memory 38g --conf spark.yarn.executor.memoryOverhead=16384 \
        --conf spark.driver.extraLibraryPath="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda-7.0/lib64:/usr/local/mkl/lib/intel64/:./caffe_on_grid_archive.tgz/lib64/caffe:./caffe_on_grid_archive.tgz/lib64" \
        --class com.yahoo.ml.caffe.CaffeOnSpark  \
          ${YROOT}/share/caffe/lib/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
            -test \
            -conf lenet_memory_solver.prototxt \
            -devices 2 \
            -model hdfs:///user/${USER}/mnist_lenet.model \
            -output hdfs:///user/${USER}/lenet_accuracy_result
    hadoop fs -cat lenet_accuracy_result
    popd



## CaffeOnGrid command line options:

* -train: training mode
* -test: test mode
* -features: feature extraction mode, followed with a list of features separated by comma. This function uses test phase defined in the network prototxt file, where the batch size should divide total number of samples. Normally, non-aggregated features should be used. If you use aggregated features, individual sample will get the same batch-aggregated values, and we recommend you change batch size to 1.
* -outputFormat: feature output format, either json or parquet, json by default.
* -conf: solver prototxt file
* -devices: number of GPUs per node
* -model: output model file name after training.
* -snapshot: input state file for resuming training. "-weights" is also required together with this option.
* -weights: input model file as initial weights for training, either used with "-snapshot" or used alone.
* -resize: resize input images. The height and width are specified in memory data layer of the prototxt file.

## Important updates:

* Batch sizes specified in prototxt files are per device.
* Memory layers are shared among GPUs by default due to BVLC caffe update. Please add "share_in_parallel: false" in your Memory layer prototxt file to disable it, as shown in examples in data/ directory.
* Inner product layer that takes sparse matrices is re-named to SparseInnerProductLayer, please update your net prototxt file if you are using it. 

## License

The use and distribution terms for this software are covered by the Apache 2.0 license.
See LICENSE file for terms.

