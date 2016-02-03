screwdriver: build package

HOME ?=/home/${USER}

build: update
	pushd caffe-public; make proto; make -j4 -e distribute; popd
	export LD_LIBRARY_PATH="$(shell pwd)/caffe-public/distribute/lib:$(shell pwd)/caffe-distri/distribute/lib:/home/y/lib64:/home/y/lib64/mkl/intel64:/usr/lib64:/lib64"; mvn package

update:
	git submodule init
	git submodule update --force
	git submodule foreach --recursive git clean -dfx

package:
	pushd pkg; yinst_create --clean -r; popd

doc: 
	rm -rf scala_doc
	scaladoc -cp caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar:${HOME}/.m2/repository/org/apache/spark/spark-core_2.10/1.6.0/spark-core_2.10-1.6.0.jar:${HOME}/.m2/repository/org/apache/spark/spark-sql_2.10/1.6.0/spark-sql_2.10-1.6.0.jar:${HOME}/.m2/repository/org/apache/spark/spark-mllib_2.10/1.6.0/spark-mllib_2.10-1.6.0.jar:${HOME}/.m2/repository/org/apache/spark/spark-catalyst_2.10/1.6.0/spark-catalyst_2.10-1.6.0.jar:$(shell hadoop classpath)  -d scala_doc caffe-grid/src/main/scala/com/yahoo/ml/caffe/*.scala

clean: 
	pushd caffe-public; make clean; popd
	pushd caffe-distri; make clean; popd
	mvn clean 

ALL: build
