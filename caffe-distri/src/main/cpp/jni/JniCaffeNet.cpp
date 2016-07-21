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
 * Signature: (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;IIIZIII)Z
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_allocate
(JNIEnv *env, jobject object, jstring solver_conf_file, jstring model_file, jstring state_file,
 jint num_local_devices, jint cluster_size, jint myRank, jboolean isTraining,
 jint connection_type, jint start_device_id, jint validation_net_id) {
  /* create a native CaffeNet object */
  CaffeNet<float>* native_ptr = NULL;
  
  jboolean isCopy_solver = false;
  const char* solver_conf_file_chars = env->GetStringUTFChars(solver_conf_file, &isCopy_solver);
  if (solver_conf_file_chars == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "solver_conf_file_chars == NULL";
    return false;
  }
  jboolean isCopy_model = false;
  const char* model_file_chars = env->GetStringUTFChars(model_file, &isCopy_model);
  if (model_file_chars == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "model_file_chars == NULL";
    return false;
  }
  jboolean isCopy_state = false;
  const char* state_file_chars = env->GetStringUTFChars(state_file, &isCopy_state);
  if (state_file_chars == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "state_file_chars == NUL";
    return false;
  }

  try {
    if (cluster_size ==1)
      native_ptr = new LocalCaffeNet<float>(solver_conf_file_chars,
                                            model_file_chars,
                                            state_file_chars,
                                            num_local_devices, isTraining, start_device_id, validation_net_id);
    else {
      switch (connection_type) {
#ifdef INFINIBAND
        case com_yahoo_ml_jcaffe_CaffeNet_RDMA:
          native_ptr = new RDMACaffeNet<float>(solver_conf_file_chars,
                                               model_file_chars,
                                               state_file_chars,
                                               num_local_devices, cluster_size, myRank, isTraining,
                                               start_device_id, validation_net_id);
          break;
#endif
        case com_yahoo_ml_jcaffe_CaffeNet_SOCKET:
          native_ptr = new SocketCaffeNet<float>(solver_conf_file_chars,
                                                 model_file_chars,
                                                 state_file_chars,
                                                 num_local_devices, cluster_size, myRank, isTraining,
                                                 start_device_id, validation_net_id);
          break;
      }
    }
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return false;
  }

  if (native_ptr == NULL) {
    LOG(ERROR) << "unable to create CaffeNet object";
    return false;
  }
  
  if (isCopy_solver)
    env->ReleaseStringUTFChars(solver_conf_file, solver_conf_file_chars);
  if (isCopy_model)
    env->ReleaseStringUTFChars(model_file, model_file_chars);
  if (isCopy_state)
    env->ReleaseStringUTFChars(state_file, state_file_chars);
  
  if (env->ExceptionCheck()) {
    LOG(ERROR) << "ReleaseStringUTFChars failed";
    return false;
  }
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
  if (object == NULL) {
    LOG(ERROR) << "localAddresses object is NULL";
    return NULL;
  }

  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  
  vector<string> addrs;
  try {
    native_ptr->localAddresses(addrs);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  
  // Get a class reference for com.yahoo.ml.jcaffe..FloatBlob
  jclass classString = env->FindClass("java/lang/String");
  if (env->ExceptionCheck()) {
    LOG(ERROR) << "FindClass failed";
    return NULL;
  }
  // Allocate a jobjectArray of com.yahoo.ml.jcaffe.FloatBlob
  jsize len = addrs.size();
  jobjectArray outJNIArray = env->NewObjectArray(len, classString, NULL);
  if (outJNIArray == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "Unable to create a new array";
    return NULL;
  }
  //construct a set of JVM String object
  int i;
  for (i=0; i<len; i++) {
    LOG(INFO) << i << "-th local addr: " << addrs[i].c_str();
    jstring str = env->NewStringUTF(addrs[i].c_str());
    if (str == NULL || env->ExceptionCheck()) {
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
  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
    native_ptr->sync();
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return false;
  }
  return true;
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    connect
 * Signature: ([Ljava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_connect
(JNIEnv *env, jobject object, jobjectArray address_array) {
  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return false;
  }

  /*  if (address_array == NULL) {
    LOG(ERROR) << "Address is NULL";
    return false;
    }*/
  jsize length = (address_array==NULL? 0 : env->GetArrayLength(address_array));
  vector<const char*>  addresses(length);
  if(!GetStringVector(addresses, env, address_array, length)) {
    LOG(ERROR) << "Unable to retrieve StringVector";
    return false;
  }
  
  try {
    if (!native_ptr->connect(addresses))
      return false;
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return false;
  }

  for (int i = 0; i < length; i++) {
    if(addresses[i] != NULL){
      jstring addr = (jstring)env->GetObjectArrayElement(address_array, i);
      if (env->ExceptionCheck()) {
        LOG(ERROR) << "GetObjectArrayElement failed";
        return false;
      }
      env->ReleaseStringUTFChars(addr, addresses[i]);
      if (env->ExceptionCheck()) {
        LOG(ERROR) << "ReleaseStringUTFChar failed";
        return false;
      }
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
  if (solver_index < 0) {
    LOG(ERROR) << "Solver index invalid";
    return -1;
  }
  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
    return native_ptr->deviceID(solver_index);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    init
 * Signature: (IZ)Z
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_init
(JNIEnv *env, jobject object, jint solver_index, jboolean enableNN) {
  CaffeNet<float>* native_ptr = NULL;
  if (solver_index < 0) {
    LOG(ERROR) << "Solver index invalid";
    return false;
  }
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
    return native_ptr->init(solver_index, enableNN);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return false;
  }
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    predict
 * Signature: (I[Lcom/yahoo/ml/jcaffe/FloatBlob;[Ljava/lang/String;)[Lcom/yahoo/ml/jcaffe/FloatBlob;
 */
JNIEXPORT jobjectArray JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_predict
(JNIEnv *env, jobject object, jint solver_index, jobjectArray input_data, jobjectArray output_blobnames) {
  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  
  if (input_data == NULL) {
    LOG(ERROR) << "data is NULL";
    ThrowCosJavaException((char*)"data is NULL", env);
    return NULL;
  }

  size_t length = env->GetArrayLength(input_data);
  if (env->ExceptionCheck()) {
    LOG(ERROR) << "GetArrayLength failed";
    return NULL;
  }
  vector< Blob<float>* > data_vec(length);
  
  if(!GetFloatBlobVector(data_vec, env, input_data, length)) {
    LOG(ERROR) << "Could not get FoatBlob vector";
    return NULL;
  }
  
  length = env->GetArrayLength(output_blobnames);
  if (env->ExceptionCheck()) {
    LOG(ERROR) << "GetArrayLength failed";
    return NULL;
  }
  if (length==0) return NULL;
  vector<const char*>  output_blobnames_chars(length);
  
  if(!GetStringVector(output_blobnames_chars, env, output_blobnames, length)){
    LOG(ERROR) << "Could not get String vector";
    return NULL;
  }

  vector<Blob<float>* > results(length);
  try {
    native_ptr->predict(solver_index, data_vec, output_blobnames_chars, results);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  
  // Get a class reference for com.yahoo.ml.jcaffe.FloatBlob
  jclass classFloatBlob = env->FindClass("com/yahoo/ml/jcaffe/FloatBlob");
  if (env->ExceptionOccurred()) {
    LOG(ERROR) << "Unable to find class FloatBlob";
    return NULL;
  }
  jmethodID midFloatBlobInit = env->GetMethodID(classFloatBlob, "<init>", "(JZ)V");
  if (midFloatBlobInit == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "Unable to locate method init";
    return NULL;
  }
  // Allocate a jobjectArray of com.yahoo.ml.jcaffe.FloatBlob
  jobjectArray outJNIArray = env->NewObjectArray(length, classFloatBlob, NULL);
  if (outJNIArray == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "Unable to allocate a new array";
    return NULL;
  }
  //construct a set of JVM FloatBlob object from native Blob<float>
  for (int i=0; i<length; i++) {
    //FloatBlob object created here should not release native blob<float> object
    jobject obj = env->NewObject(classFloatBlob, midFloatBlobInit, results[i], false);
    if (obj == NULL || env->ExceptionCheck()) {
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
      if (env->ExceptionCheck()) {
        LOG(ERROR) << "GetObjectArrayElement failed";
        return NULL;
      }
      
      env->ReleaseStringUTFChars(output_blobname, output_blobnames_chars[i]);
      if (env->ExceptionCheck()) {
        LOG(ERROR) << "ReleaseStringUTFChars failed";
        return NULL;
      }
      
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
(JNIEnv *env, jobject object, jint solver_index, jobjectArray input_data) {
  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return false;
  }

  if (input_data == NULL) {
    LOG(ERROR) << "data is NULL";
    ThrowCosJavaException((char*)"data is NULL", env);
    return false;
  }
  
  size_t length = (input_data != NULL? env->GetArrayLength(input_data) : 0);
  vector< Blob<float>* > data_vec(length);
  if(!GetFloatBlobVector(data_vec, env, input_data, length)) {
    LOG(ERROR) << "Could not retrieve FloatBlobVector";
    return false;
  }

  try {
    native_ptr->train(solver_index, data_vec);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return false;
  }
  return true;
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    validation
 * Signature: ([Lcom/yahoo/ml/jcaffe/FloatBlob;)V
 */
JNIEXPORT void JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_validation
(JNIEnv *env, jobject object, jobjectArray test_input_data) {
  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return ;
  }
  
  size_t test_length = (test_input_data != NULL? env->GetArrayLength(test_input_data) : 0);
  vector< Blob<float>* > test_data_vec(test_length);
  
  if(!GetFloatBlobVector(test_data_vec, env, test_input_data, test_length)) {
    LOG(ERROR) << "Could not retrieve FloatBlobVector";
    return ;
  }

  try {
    native_ptr->validation(test_data_vec);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return;
  }

  return;
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    aggregateValidationOutputs
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_aggregateValidationOutputs
(JNIEnv *env, jobject object) {
  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return;
  }

  try {
    native_ptr->aggregateValidationOutputs();
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return;
  }
  return;
}



/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    getInitIter
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_getInitIter
(JNIEnv *env, jobject object, jint solver_index) {
  if (solver_index < 0) {
    LOG(ERROR) << "Solver index invalid";
    return -1;
  }

  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
    return native_ptr->getInitIter(solver_index);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    getTestIter
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_getTestIter
(JNIEnv *env, jobject object, jint solver_index) {

  if (solver_index < 0) {
    LOG(ERROR) << "Solver index invalid";
    return -1;
  }
  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
    return native_ptr->getTestIter(solver_index);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }

}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    getTestInterval
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_getTestInterval
(JNIEnv *env, jobject object){

  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
    return native_ptr->getTestInterval();
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }


}


/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    getMaxIter
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_getMaxIter
(JNIEnv *env, jobject object, jint solver_index) {
  if (solver_index < 0) {
    LOG(ERROR) << "Solver index invalid";
    return -1;
  }
  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
    return native_ptr->getMaxIter(solver_index);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    snapshot
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_snapshot
(JNIEnv *env, jobject object) {
  if (object == NULL) {
    LOG(ERROR) << "Snapshot object is NULL";
    return -1;
  }
  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
    return native_ptr->snapshot();
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    getValidationOutputBlobNames
 * Signature: ()[Ljava/lang/String;
*/
JNIEXPORT jobjectArray JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_getValidationOutputBlobNames
(JNIEnv *env, jobject object) {
  if (object == NULL) {
    LOG(ERROR) << "NULL object for OutputBlobNames";
    return NULL;
  }
  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  vector<string> blob_names;
  try {
    blob_names = native_ptr->getValidationOutputBlobNames();
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  jobjectArray ret = (jobjectArray)env->NewObjectArray(blob_names.size(),
                     env->FindClass("java/lang/String"),
                     env->NewStringUTF(""));
  
  for(int i=0;i<blob_names.size();i++) {
    env->SetObjectArrayElement(ret,i,env->NewStringUTF(blob_names[i].c_str()));
  }
  return ret;
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    getValidationOutputBlobs
 * Signature: (I)[Lcom/yahoo/ml/jcaffe/FloatBlob;
*/
JNIEXPORT jobjectArray JNICALL Java_com_yahoo_ml_jcaffe_CaffeNet_getValidationOutputBlobs
(JNIEnv *env, jobject object, jint length) {
  CaffeNet<float>* native_ptr = NULL;
  try {
    native_ptr = (CaffeNet<float>*) GetNativeAddress(env, object);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }

  vector<Blob<float>* > results(length);
  try {
    results = native_ptr->getValidationOutputBlobs(length);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  
  // Get a class reference for com.yahoo.ml.jcaffe.FloatBlob
  jclass classFloatBlob = env->FindClass("com/yahoo/ml/jcaffe/FloatBlob");
  if (env->ExceptionOccurred()) {
    LOG(ERROR) << "Unable to find class FloatBlob";
    return NULL;
  }
  jmethodID midFloatBlobInit = env->GetMethodID(classFloatBlob, "<init>", "(JZ)V");
  if (midFloatBlobInit == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "Unable to locate method init";
    return NULL;
  }
  // Allocate a jobjectArray of com.yahoo.ml.jcaffe.FloatBlob
  jobjectArray outJNIArray = env->NewObjectArray(length, classFloatBlob, NULL);
  if (outJNIArray == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "Unable to allocate a new array";
    return NULL;
  }
  //construct a set of JVM FloatBlob object from native Blob<float>
  for (int i=0; i<length; i++) {
    //FloatBlob object created here should not release native blob<float> object
    jobject obj = env->NewObject(classFloatBlob, midFloatBlobInit, results[i], false);
    if (obj == NULL || env->ExceptionCheck()) {
      LOG(ERROR) << "Unable to construct new object";
      return NULL;
    }
    env->SetObjectArrayElement(outJNIArray, i, obj);
    if (env->ExceptionOccurred()) {
      LOG(ERROR) << "Unable to set Array Elements";
      return NULL;
    }
  }
  
  return outJNIArray;
}
