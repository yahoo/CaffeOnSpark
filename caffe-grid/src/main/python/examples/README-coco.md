Steps to run the COCO dataset for Image Captioning
==================================================
##### (1) Env setup
    export CAFFE_ON_SPARK=/Users/mridul/bigml/CaffeOnSpark
    export DYLD_LIBRARY_PATH=${CAFFE_ON_SPARK}/caffe-public/distribute/lib:${CAFFE_ON_SPARK}/caffe-distri/distribute/lib
    export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:/usr/local/cuda/lib:/usr/local/mkl/lib/intel64/
    export LD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}
    export SPARK_HOME=/Users/mridul/bigml/spark-1.6.0-bin-hadoop2.6
    export PATH=${SPARK_HOME}/bin:${PATH}
    export PYSPARK_PYTHON=Python2.7.10/bin/python
    export PYTHONPATH=$PYTHONPATH:caffeonsparkpythonapi.zip:caffe_on_grid_archive/lib64:/usr/local/cuda-7.5/lib64
    export LD_LIBRARY_PATH=Python2.7.10/lib:/usr/local/cuda/lib:caffe_on_grid_archive/lib64/mkl/intel64/:${LD_LIBRARY_PATH}
    export DYLD_LIBRARY_PATH=Python2.7.10/lib:/usr/local/cuda/lib:caffe_on_grid_archive/lib64/mkl/intel64/:${LD_LIBRARY_PATH}
    export IPYTHON_ROOT=~/Python2.7.10
    unset SPARK_CONF_DIR

##### (2) Download the coco dataset if required

    mkdir -p /tmp/coco
    pushd /tmp/coco
    wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
    wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
    wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
    wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
    unzip *.zip
    popd

##### (3) Create the input dataframe from cocodataset
    #-output the root directory for producing all the outputs
    #-imageRoot the root input directory for all the images (required)
    #-captionFile the input json which contains the image details and captions in coco format (check on mscoco.org)
    #-outputFormat the format of the output file to produce the dataframe
    #-imageCaptionDFDir the dataframe output dir name for images and their captions under -output, in json
    #-vocabDir the vocabulary for the dataframe under -output in desired outputFormat
    #-embeddingDFDir the dataframe output dir name for embedded images and their captions under -output in desired outputFormat
   
    pushd ${CAFFE_ON_SPARK}/data/
    spark-submit --master ${MASTER_URL} --deploy-mode client \
        --conf spark.executor.extraClassPath=${CAFFE_ON_SPARK}/caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
	--conf --driver-class-path=${CAFFE_ON_SPARK}/caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
        --class com.yahoo.ml.caffe.tools.CocoDataSetConverter  \
        ${CAFFE_ON_SPARK}/caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
        -output  /tmp/coco/parquet/ \
        -imageRoot /tmp/coco/images/train2014/ \
        -captionFile /tmp/coco/annotations/captions_train2014.json \
        -outputFormat parquet \
        -imageCaptionDFDir df_image_caption_train2014 \
        -vocabDir vocab \
	-vocabSize 8800 \
        -embeddingDFDir df_embedded_train2014
    popd

##### (4) Train the image model
    pushd ${CAFFE_ON_SPARK}/data/
    spark-submit --master ${MASTER_URL} \
        --files train_val.prototxt,solver.prototxt \
        --conf spark.cores.max=${TOTAL_CORES} \
        --conf spark.task.cpus=${CORES_PER_WORKER} \
    	--conf spark.driver.extraLibraryPath="${DYLD_LIBRARY_PATH}" \
	--conf spark.executorEnv.DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}" \
        --class com.yahoo.ml.caffe.CaffeOnSpark  \
	${CAFFE_ON_SPARK}/caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
            -train \
            -conf solver.prototxt \
            -model /tmp/coco/bvlc_reference_caffenet.caffemodel \
            -devices 1
    hadoop fs -ls /tmp/coco/bvlc_reference_caffenet.caffemodel
    popd
##### (5) Train the lstm
    pushd ${CAFFE_ON_SPARK}/data/
    spark-submit --master ${MASTER_URL} \
        --files lrcn_cos.prototxt,lrcn_solver.prototxt \
	--conf spark.cores.max=${TOTAL_CORES} \
        --conf spark.task.cpus=${CORES_PER_WORKER} \
        --conf spark.driver.extraLibraryPath="${DYLD_LIBRARY_PATH}" \
        --conf spark.executorEnv.DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}" \       
        --class com.yahoo.ml.caffe.CaffeOnSpark  \
  	  ${CAFFE_ON_SPARK}/caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar \
	     -train \
             -conf lrcn_solver.prototxt \
             -devices 1 \
             -resize \
             -weights /tmp/coco/bvlc_reference_caffenet.caffemodel \
             -model /tmp/coco/parquet/lrcn_coco.model
    popd

##### (6) Submit the data for inference
    Note that the below files also need to be shipped as shown
    #-model the image-lstm pretrained model to ship 
    #-imagenet the image network definition
    #-lstmnet the lstm network definition
    #-vocab the vocabulary file (produced from above) for the given train set
    #-input the input embedding produced above
    #-output the path where to write the desired output

    pushd ${CAFFE_ON_SPARK}/data/
    ln -s ~/Python2.7.10 Python2.7.10
    unzip ${CAFFE_ON_SPARK}/caffe-grid/target/caffeonsparkpythonapi.zip
    cat /tmp/coco/parquet/vocab/part* > vocab.txt
    rm -rf /tmp/coco/parquet/df_caption_results_train2014
    spark-submit --master ${MASTER_URL} \
    		 --conf spark.cores.max=${TOTAL_CORES} \
    		 --conf spark.task.cpus=${CORES_PER_WORKER} \    
    		 --conf spark.driver.extraLibraryPath="${DYLD_LIBRARY_PATH}:Python2.7.10/lib" \
    		 --conf spark.executorEnv.LD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:Python2.7.10/lib" \
    		 --conf spark.pythonargs="-model /tmp/coco/parquet/lrcn_coco.model -imagenet deploy.prototxt -lstmnet lrcn_word_to_preds.deploy.prototxt -vocab vocab.txt -input /tmp/coco/parquet/df_embedded_train2014 -output /tmp/coco/parquet/df_caption_results_train2014" examples/ImageCaption.py
    popd
##### (7) Launch IPython Notebook
    export IPYTHON_OPTS="notebook --no-browser --ip=127.0.0.1"
    pushd ${CAFFE_ON_SPARK}/data/
    ln -s ~/Python2.7.10 Python2.7.10
    unzip ${CAFFE_ON_SPARK}/caffe-grid/target/caffeonsparkpythonapi.zip
    cat /tmp/coco/parquet/vocab/part* > vocab.txt
    pyspark --master ${MASTER_URL} --deploy-mode client \    
    	    --conf spark.driver.extraLibraryPath="${DYLD_LIBRARY_PATH}:Python2.7.10/lib" \
	    --conf spark.executorEnv.LD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:Python2.7.10/lib" \
	    --files "${CAFFE_ON_SPARK}/data/deploy.prototxt,${CAFFE_ON_SPARK}/data/vocab.txt,${CAFFE_ON_SPARK}/data/lrcn_word_to_preds.deploy.prototxt,${CAFFE_ON_SPARK}/data/caffe/_caffe.so" \
	    --py-files "${CAFFE_ON_SPARK}/caffe-grid/target/caffeonsparkpythonapi.zip" \
	    --jars "${CAFFE_ON_SPARK}/caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar" \
	    --driver-library-path "${CAFFE_ON_SPARK}/caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar" \
	    --driver-class-path "${CAFFE_ON_SPARK}/caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar"    