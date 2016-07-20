// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include <vector>
#include <opencv2/core/core.hpp>
#include <glog/logging.h>

#include "common.hpp"
#include "jni/com_yahoo_ml_jcaffe_MatVector.h"
/*
 * Class:     com_yahoo_ml_jcaffe_MatVector
 * Method:    allocate
 * Signature: (I)Z
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_MatVector_allocate
  (JNIEnv *env, jobject object, jint size) {

  /* create a native vector<cv::Mat*> object */
  vector<cv::Mat>* native_ptr = NULL;
  if (size < 0) {
    LOG(ERROR) << "Negative MatVector size specified";
    ThrowCosJavaException((char*)"Negative MatVector size specified", env);
    return false;
  }
    
  try {
    native_ptr = new  vector<cv::Mat>(size);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return false;
  }

  if (native_ptr == NULL) {
    LOG(ERROR) << "unable to allocate memory for vector of Mats";
    return false;
  }
  /* associate native object with JVM object */
  return SetNativeAddress(env, object, native_ptr);
}

/*
 * Class:     com_yahoo_ml_jcaffe_MatVector
 * Method:    deallocate
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_yahoo_ml_jcaffe_MatVector_deallocateVec(JNIEnv *env, jobject object, jlong address) {

  vector<cv::Mat>* native_ptr = (vector<cv::Mat>*) address;
    
  delete native_ptr;
}

/*
 * Class:     com_yahoo_ml_jcaffe_MatVector
 * Method:    put
 * Signature: (ILcom/yahoo/ml/jcaffe/Mat;)V
 */
JNIEXPORT void JNICALL Java_com_yahoo_ml_jcaffe_MatVector_putnative
  (JNIEnv *env, jobject object, jint pos, jobject mat) {

  vector<cv::Mat> *native_ptr = NULL;
  try {
    native_ptr = (vector<cv::Mat>*) GetNativeAddress(env, object);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return;
  }

  
  cv::Mat* mat_ptr = NULL;
  try {
    mat_ptr = (cv::Mat*) GetNativeAddress(env, mat);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return;
  }

  if (mat_ptr==NULL) {
    LOG(ERROR) << "invalid native address of Mat";
    ThrowCosJavaException((char*)"invalid native address of Mat", env);
    return;
  }
  
  if (pos < 0 || pos >= native_ptr->size()) {
    LOG(ERROR) << "invalid index in MatVector";
    ThrowCosJavaException((char*)"invalid index in MatVector", env);
    return;
  }
  
  (*native_ptr)[pos] = *mat_ptr;
}

/*
 * Class:     com_yahoo_ml_jcaffe_MatVector
 * Method:    data
 * Signature: (I)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_yahoo_ml_jcaffe_MatVector_data
    (JNIEnv *env, jobject object, jint pos) {
    
  vector<cv::Mat> *native_ptr = NULL;
  try {
    native_ptr = (vector<cv::Mat>*) GetNativeAddress(env, object);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  
  if (pos < 0 || pos > native_ptr->size()) {
    LOG(ERROR) << "Invalid Mat index in MatVector";
    return NULL;
  }

  cv::Mat mat = (cv::Mat)(*native_ptr)[pos];
  int size = 0;
  try {
    size = mat.total() * mat.elemSize();
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  
  jbyteArray dataarray = env->NewByteArray(size);
  if(dataarray == NULL || env->ExceptionCheck()){
    LOG(ERROR) << "Out of memory while allocating array for Mat data" ;
    return NULL;
  }
  env->SetByteArrayRegion(dataarray,0, size, (jbyte*)mat.data);
  if (env->ExceptionCheck()) {
    LOG(ERROR) << "SetByteArrayRegion failed";
    return NULL;
  }
  return dataarray;
}

/*
 * Class:     com_yahoo_ml_jcaffe_MatVector
 * Method:    height
 * Signature: (I)I
 */

JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_MatVector_height
    (JNIEnv *env, jobject object, jint pos) {
  
  vector<cv::Mat> *native_ptr = NULL;
  try {
    native_ptr = (vector<cv::Mat>*) GetNativeAddress(env, object);
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }
  if (pos < 0 || pos > native_ptr->size()) {
    LOG(ERROR) << "Invalid Mat index in MatVector";
    return -1;
  }
  cv::Mat mat = (cv::Mat)(*native_ptr)[pos];
  try {
    return mat.rows;
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }
}

/*
 * Class:     com_yahoo_ml_jcaffe_MatVector
 * Method:    width
 * Signature: (I)I
 */

JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_MatVector_width
    (JNIEnv *env, jobject object, jint pos) {
  
  vector<cv::Mat> *native_ptr = NULL;
  try {
    native_ptr = (vector<cv::Mat>*) GetNativeAddress(env, object);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }
  if (pos < 0 || pos > native_ptr->size()) {
    LOG(ERROR) << "Invalid Mat index in MatVector";
    return -1;
  }
  cv::Mat mat = (cv::Mat)(*native_ptr)[pos];
  try {
    return mat.cols;
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }
}

/*
 * Class:     com_yahoo_ml_jcaffe_MatVector
 * Method:    channels
 * Signature: (I)I
 */

JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_MatVector_channels
(JNIEnv *env, jobject object, jint pos) {

  vector<cv::Mat> *native_ptr = NULL;
  try {
    native_ptr = (vector<cv::Mat>*) GetNativeAddress(env, object);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }
  if (pos < 0 || pos > native_ptr->size()) {
    LOG(ERROR) << "Invalid Mat index in MatVector";
    return -1;
  }
  cv::Mat mat = (cv::Mat)(*native_ptr)[pos];
  try {
    return mat.channels();
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }
}
