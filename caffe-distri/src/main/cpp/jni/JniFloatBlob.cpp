// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include <glog/logging.h>
#include "caffe/caffe.hpp"
#include "common.hpp"
#include "jni/com_yahoo_ml_jcaffe_FloatBlob.h"

/*
 * Class:     com_yahoo_ml_jcaffe_FloatBlob
 * Method:    allocate
 * Signature: ()V
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_FloatBlob_allocate(JNIEnv *env, jobject object) {
  /* create a native FloatBlob object */
  Blob<float>* native_ptr = NULL;
  try {
    native_ptr = new Blob<float>();
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return false;
  }

  if (native_ptr == NULL) {
    LOG(ERROR) << "Unable to allocate memory for Blob";
    return false;
  }
  /* associate native object with JVM object */
  return SetNativeAddress(env, object, native_ptr);
}

/*
 * Class:     com_yahoo_ml_jcaffe_FloatBlob
 * Method:    deallocate1
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_yahoo_ml_jcaffe_FloatBlob_deallocate1(JNIEnv *env, jobject object, jlong native_ptr, jlong dataaddress) {
   delete (Blob<float>*) native_ptr;
   if(dataaddress){
      delete (jbyte*) dataaddress;
   }
}

JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_FloatBlob_count
  (JNIEnv *env, jobject object) {
  Blob<float>* native_ptr = NULL;
  try {
    native_ptr = (Blob<float>*) GetNativeAddress(env, object);
    return native_ptr->count();
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return -1;
  }
}

/*
 * Class:     com_yahoo_ml_jcaffe_FloatBlob
 * Method:    reshape
 * Signature: ([I)V
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_FloatBlob_reshape
  (JNIEnv *env, jobject object, jintArray shape) {
  Blob<float>* native_ptr = NULL;
  try {
    native_ptr = (Blob<float>*) GetNativeAddress(env, object);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return false;
  }
  
  size_t size = env->GetArrayLength(shape);
  if (env->ExceptionCheck()) {
    LOG(ERROR) << "GetArrayLength failed";
    return false;
  }
  jint *vals = env->GetIntArrayElements(shape, NULL);
  if (vals == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "vals == NULL";
    return false;
  }
  
  vector<jint> shap_vec(size);
  for (int i=0; i<size; i++) {
    if (vals[i] < 1)  {
      LOG(ERROR) << "Invalid val for reshape dimension";
      return false;
    }
    shap_vec[i] = vals[i];
  }
  
  try {
    native_ptr->Reshape(shap_vec);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return false;
  }
  //release JNI objects
  env->ReleaseIntArrayElements(shape, vals, JNI_ABORT);
  if (env->ExceptionOccurred()) {
    LOG(ERROR) << "Unable to release Array Elements";
    return false;
  }
  
  return true;
}

/*
 * Class:     com_yahoo_ml_jcaffe_FloatBlob
 * Method:    CopyFrom
 * Signature: (Lcom/yahoo/ml/jcaffe/FloatBlob;)V
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_FloatBlob_copyFrom(JNIEnv *env, jobject object, jobject source) {
  Blob<float>* native_ptr = NULL;
  Blob<float>* source_ptr = NULL;
  if (source == NULL) {
    LOG(ERROR) << "source is NULL";
    return false;
  }
  try {
    native_ptr = (Blob<float>*) GetNativeAddress(env, object);
    source_ptr = (Blob<float>*) GetNativeAddress(env, source);
    //perform operation
    native_ptr->CopyFrom(*source_ptr);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return false;
  }
  return true;
}

/*
 * Class:     com_yahoo_ml_jcaffe_FloatBlob
 * Method:    cpu_data
 * Signature: ()[F
 */
JNIEXPORT jobject JNICALL Java_com_yahoo_ml_jcaffe_FloatBlob_cpu_1data(JNIEnv *env, jobject object) {
  Blob<float>* native_ptr = NULL;
  try {
    native_ptr = (Blob<float>*) GetNativeAddress(env, object);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  jfloat* cpu_data = NULL;
  try {
    //retrieve cpu_data()
    cpu_data = (jfloat*) native_ptr->mutable_cpu_data();
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  if (cpu_data == NULL) {
    LOG(ERROR) << "cpu_data == NULL";
    return NULL;
  }
  
  jclass claz = env->FindClass("com/yahoo/ml/jcaffe/FloatArray");
  jmethodID constructorId = env->GetMethodID( claz, "<init>", "(J)V");
  jobject objectFloatArray = env->NewObject(claz,constructorId,(long)cpu_data);
  if (env->ExceptionCheck()) {
    LOG(ERROR) << "FloatArray object creation failed";
    return NULL;
  }    
  return objectFloatArray;
}

/*
 * Class:     com_yahoo_ml_jcaffe_FloatBlob
 * Method:    set_cpu_data
 * Signature: ([F)V
 */
JNIEXPORT jlong JNICALL Java_com_yahoo_ml_jcaffe_FloatBlob_set_1cpu_1data(JNIEnv *env, jobject object, jfloatArray array, jlong dataaddress) {

  Blob<float>* native_ptr = NULL;
  try {
    native_ptr = (Blob<float>*) GetNativeAddress(env, object);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return 0;
  }
  
  jboolean copied = false;
  if (array == NULL) {
    LOG(ERROR) << "input array is NULL";
    return 0;
  }
  float* data = env->GetFloatArrayElements(array, &copied);
  if (data == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "GetFloatArrayElements() == NULL";
    return 0;
  }
  
  if (!copied) {
    size_t len = 0;
    try {
      len = native_ptr->count();
    } catch (const std::exception& ex) {
      ThrowJavaException(ex, env);
      return 0;
    }
    float* new_data = new float[len];
    if (new_data == NULL) {
      LOG(ERROR) << "fail to float[] for new data";
      return 0;
    }
    
    memcpy(new_data, data, len * sizeof(float));
    //set new data
    data = new_data;
  }
  
  try {
    native_ptr->set_cpu_data(data);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return 0;
  }
  
  if(dataaddress)
    delete (jbyte*) dataaddress;
  
  return (long) data;
}

/*
 * Class:     com_yahoo_ml_jcaffe_FloatBlob
 * Method:    gpu_data
 * Signature: ()[F
 */
JNIEXPORT jobject JNICALL Java_com_yahoo_ml_jcaffe_FloatBlob_gpu_1data(JNIEnv *env, jobject object) {
  Blob<float>* native_ptr = NULL;
  try {
    native_ptr = (Blob<float>*) GetNativeAddress(env, object);
  } catch (const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  jfloat* gpu_data = NULL;
  try {
    //retrieve gpu_data()
    gpu_data = (jfloat*)native_ptr->mutable_gpu_data();
  } catch(const std::exception& ex) {
    ThrowJavaException(ex, env);
    return NULL;
  }
  if (gpu_data == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "gpu_data == NULL";
    return NULL;
  }
  
  jclass claz = env->FindClass("com/yahoo/ml/jcaffe/FloatArray");
  jmethodID constructorId = env->GetMethodID( claz, "<init>", "(J)V");
  jobject objectFloatArray = env->NewObject(claz,constructorId,(long)gpu_data);
  if (env->ExceptionCheck()) {
    LOG(ERROR) << "FloatArray object creation failed";
    return NULL;
  }

  return objectFloatArray;
}
