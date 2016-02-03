// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include "caffe/caffe.hpp"
#include "CaffeNet.hpp"
#include "common.hpp"
#include "jni/com_yahoo_ml_jcaffe_CaffeNet.h"

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    allocate
 * Signature: (Ljava/lang/String;Ljava/lang/String;IIIZII)Z
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_allocate
(JNIEnv *env, jobject object, jstring solver_conf_file, jstring model_file, jstring state_file,
 jint num_local_devices, jint cluster_size, jint myRank, jboolean isTraining,
 jint connection_type, jint start_device_id) {
    /* create a native CaffeNet object */
    CaffeNet<float>* native_ptr = NULL;

    jboolean isCopy_solver = false;
    const char* solver_conf_file_chars = env->GetStringUTFChars(solver_conf_file, &isCopy_solver);
    if (solver_conf_file_chars == NULL) {
      LOG(ERROR) << "solver_conf_file_chars == NULL";
      return false;
    }
    jboolean isCopy_model = false;
    const char* model_file_chars = env->GetStringUTFChars(model_file, &isCopy_model);
    if (model_file_chars == NULL) {
      LOG(ERROR) << "model_file_chars == NULL";
      return false;
    }
    jboolean isCopy_state = false;
    const char* state_file_chars = env->GetStringUTFChars(state_file, &isCopy_state);
    if (state_file_chars == NULL) {
      LOG(ERROR) << "state_file_chars == NUL";
      return false;
    }
    if (cluster_size ==1)
        native_ptr = new LocalCaffeNet<float>(solver_conf_file_chars,
                                              model_file_chars,
                                              state_file_chars,
                                              num_local_devices, isTraining, start_device_id);
    else {
      switch (connection_type) {
#ifdef INFINIBAND
      case com_yahoo_ml_jcaffe_CaffeNet_RDMA:
	    native_ptr = new RDMACaffeNet<float>(solver_conf_file_chars,
					    model_file_chars,
					    state_file_chars,
					    num_local_devices, cluster_size, myRank, isTraining,
					    start_device_id);
	    break;
#endif
      case com_yahoo_ml_jcaffe_CaffeNet_SOCKET:
        native_ptr = new SocketCaffeNet<float>(solver_conf_file_chars,
					    model_file_chars,
					    state_file_chars,
					    num_local_devices, cluster_size, myRank, isTraining,
					    start_device_id);
	    break;
      }
    }
    if (isCopy_solver)
        env->ReleaseStringUTFChars(solver_conf_file, solver_conf_file_chars);
    if (isCopy_model)
        env->ReleaseStringUTFChars(model_file, model_file_chars);
    if (isCopy_state)
        env->ReleaseStringUTFChars(state_file, state_file_chars);

    /* associate native object with JVM object */
    return SetNativeAddress(env, object, native_ptr);
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    deallocate
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_deallocate
(JNIEnv *env, jobject object, jlong address) {
   delete (CaffeNet<float>*) address;
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    serverAddr
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_localAddresses
(JNIEnv *env, jobject object) {
    CaffeNet<float>* native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);

    vector<string> addrs;
    native_ptr->localAddresses(addrs);

    // Get a class reference for com.yahoo.ml.jcaffe..FloatBlob
    jclass classString = env->FindClass("java/lang/String");

    // Allocate a jobjectArray of com.yahoo.ml.jcaffe.FloatBlob
    jsize len = addrs.size();
    jobjectArray outJNIArray = env->NewObjectArray(len, classString, NULL);
    if (outJNIArray == NULL) {
      LOG(ERROR) << "Unable to create a new array";
      return NULL;
    }
    //construct a set of JVM String object
    int i;
    for (i=0; i<len; i++) {
        LOG(INFO) << i << "-th local addr: " << addrs[i].c_str();
        jstring str = env->NewStringUTF(addrs[i].c_str());
        if (str == NULL) {
            LOG(ERROR) << "Unable to create new String";
            return NULL;
        }
        env->SetObjectArrayElement(outJNIArray, i, str);
        if (env->ExceptionOccurred()) {
            LOG(ERROR) << "Unable to set Array Elements";
            return NULL;
        }
    }

    return outJNIArray;
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    sync
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_sync
(JNIEnv *env, jobject object) {
    CaffeNet<float>* native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
    native_ptr->sync();
    return true;
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    connect
 * Signature: ([Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_connect
(JNIEnv *env, jobject object, jobjectArray address_array) {
    CaffeNet<float>* native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);

    jsize length = (address_array==NULL? 0 : env->GetArrayLength(address_array));
    vector<const char*>  addresses(length);
    if(!GetStringVector(addresses, env, address_array, length)) {
      LOG(ERROR) << "Unable to retrieve StringVector";
      return false;
    }

    native_ptr->connect(addresses);
    for (int i = 0; i < length; i++) {
      if(addresses[i] != NULL){
            jstring addr = (jstring)env->GetObjectArrayElement(address_array, i);
            env->ReleaseStringUTFChars(addr, addresses[i]);
        }
    }
    return true;
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    deviceID
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_deviceID
(JNIEnv *env, jobject object, jint solver_index) {

    CaffeNet<float>* native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
    return native_ptr->deviceID(solver_index);
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    init
 * Signature: (IZ)Z
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_init
(JNIEnv *env, jobject object, jint solver_index, jboolean enableNN) {
    CaffeNet<float>* native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);

    return native_ptr->init(solver_index, enableNN);
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    predict
 * Signature: (I[Lcom/yahoo/ml/jcaffe/FloatBlob;[Ljava/lang/String;)[Lcom/yahoo/ml/jcaffe/FloatBlob;
 */
JNIEXPORT jobjectArray JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_predict
(JNIEnv *env, jobject object, jint solver_index, jobjectArray input_data, jobject input_labels, jobjectArray output_blobnames) {
    CaffeNet<float>* native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);

    size_t length = env->GetArrayLength(input_data);
    vector< Blob<float>* > data_vec(length);
    
    if(!GetFloatBlobVector(data_vec, env, input_data, length)) {
      LOG(ERROR) << "Could not get FoatBlob vector";
      return NULL;
    }

    length = env->GetArrayLength(output_blobnames);
    if (length==0) return NULL;
    vector<const char*>  output_blobnames_chars(length);
    
    if(!GetStringVector(output_blobnames_chars, env, output_blobnames, length)){
      LOG(ERROR) << "Could not get String vector";
      return NULL;
    }

    /* Get a reference to JVM object class */
    jclass claz = env->GetObjectClass(input_labels);
    if (claz == NULL) {
      LOG(ERROR) << "unable to get input_label's class (FloatArray)";
      return 0;
    }
    /* Getting the field id in the class */
    jfieldID fieldId = env->GetFieldID(claz, "arrayAddress", "J");
    if (fieldId == NULL) {
      LOG(ERROR) << "could not locate field 'arrayAddress'";
      return 0;
    }

    jfloat* labels = (jfloat*) env->GetLongField(input_labels, fieldId);
    if (labels==NULL) {
      LOG(ERROR) << "labels are NULL";
      return NULL;
    }
    vector<Blob<float>* > results(length);
    native_ptr->predict(solver_index, data_vec, labels, output_blobnames_chars, results);

    // Get a class reference for com.yahoo.ml.jcaffe.FloatBlob
    jclass classFloatBlob = env->FindClass("com/yahoo/ml/jcaffe/FloatBlob");
    if (env->ExceptionOccurred()) {
      LOG(ERROR) << "Unable to find class FloatBlob";
      return NULL;
    }
    jmethodID midFloatBlobInit = env->GetMethodID(classFloatBlob, "<init>", "(JZ)V");
    if (midFloatBlobInit == NULL) {
      LOG(ERROR) << "Unable to locate method init";
      return NULL;
    }
    // Allocate a jobjectArray of com.yahoo.ml.jcaffe.FloatBlob
    jobjectArray outJNIArray = env->NewObjectArray(length, classFloatBlob, NULL);
    if (outJNIArray == NULL) {
      LOG(ERROR) << "Unable to allocate a new array";
      return NULL;
    }
    //construct a set of JVM FloatBlob object from native Blob<float>
    for (int i=0; i<length; i++) {
        //FloatBlob object created here should not release native blob<float> object
        jobject obj = env->NewObject(classFloatBlob, midFloatBlobInit, results[i], false);
        if (obj == NULL) {
            LOG(ERROR) << "Unable to construct new object";
            return NULL;
        }
        env->SetObjectArrayElement(outJNIArray, i, obj);
        if (env->ExceptionOccurred()) {
            LOG(ERROR) << "Unable to set Array Elements";
            return NULL;
        }
    }

    //release JNI objects
    for (int i = 0; i < length; i++) {
      if (output_blobnames_chars[i] != NULL) {
            jstring output_blobname = (jstring)env->GetObjectArrayElement(output_blobnames, i);
            env->ReleaseStringUTFChars(output_blobname, output_blobnames_chars[i]);
        }
    }
    return outJNIArray;
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    train
 * Signature: (I[Lcom/yahoo/ml/jcaffe/FloatBlob;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_train
(JNIEnv *env, jobject object, jint solver_index, jobjectArray input_data, jobject input_labels) {
    CaffeNet<float>* native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);

    size_t length = (input_data != NULL? env->GetArrayLength(input_data) : 0);
    vector< Blob<float>* > data_vec(length);
    /* Get a reference to JVM object class */
    jclass claz = env->GetObjectClass(input_labels);
    if (claz == NULL) {
      LOG(ERROR) << "unable to get input_label's class (FloatArray)";
      return 0;
    }
    /* Getting the field id in the class */
    jfieldID fieldId = env->GetFieldID(claz, "arrayAddress", "J");
    if (fieldId == NULL) {
      LOG(ERROR) << "could not locate field 'arrayAddress'";
      return 0;
    }

    jfloat* labels = (jfloat*) env->GetLongField(input_labels, fieldId);
    if (labels==NULL) {
      LOG(ERROR) << "labels are NULL";
      return false;
    }

    if(!GetFloatBlobVector(data_vec, env, input_data, length)) {
      LOG(ERROR) << "Could not retrieve FloatBlobVector";
      return false;
    }

    native_ptr->train(solver_index, data_vec, labels);
    
    return true;
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    getInitIter
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_getInitIter
(JNIEnv *env, jobject object, jint solver_index) {
    CaffeNet<float>* native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);

    return native_ptr->getInitIter(solver_index);
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    getMaxIter
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_getMaxIter
(JNIEnv *env, jobject object, jint solver_index) {
    CaffeNet<float>* native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);

    return native_ptr->getMaxIter(solver_index);
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    snapshot
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_snapshot
(JNIEnv *env, jobject object) {

    CaffeNet<float>* native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);

    return native_ptr->snapshot();
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    getTestOutputBlobNames
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_getTestOutputBlobNames
  (JNIEnv *env, jobject object) {
    CaffeNet<float>* native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
    string blob_names = native_ptr->getTestOutputBlobNames();
    return env->NewStringUTF(blob_names.c_str());
 }
