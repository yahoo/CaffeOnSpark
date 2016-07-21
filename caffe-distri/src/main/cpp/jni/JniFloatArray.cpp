// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include <glog/logging.h>

#include "caffe/caffe.hpp"
#include "common.hpp"
#include "jni/com_yahoo_ml_jcaffe_FloatArray.h"

/*
* Class:     com_yahoo_ml_jcaffe_FloatArray
* Method:    get
* Signature: (I)F
 */
JNIEXPORT jfloat JNICALL Java_com_yahoo_ml_jcaffe_FloatArray_get(JNIEnv *env, jobject object, jint index){
  /* Get a reference to JVM object class */
  jclass claz = env->GetObjectClass(object);
  if (claz == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "unable to get object's class";
    return 0;
  }
  /* Getting the field id in the class */
  jfieldID fieldId = env->GetFieldID(claz, "arrayAddress", "J");
  if (fieldId == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "could not locate field 'arrayAddress'";
    return 0;
  }
  
  jfloat* float_array_ptr = (jfloat*) env->GetLongField(object, fieldId);
  if (env->ExceptionCheck()) {
    LOG(ERROR) << "GetLongField failed";
    return 0;
  }
  if (index < 0) {
    LOG(ERROR) << "negative index";
    return 0;
  }
  return float_array_ptr[index];
}

/*
* Class:     com_yahoo_ml_jcaffe_FloatArray
* Method:    set
* Signature: (IF)V
*/
JNIEXPORT void JNICALL Java_com_yahoo_ml_jcaffe_FloatArray_set(JNIEnv *env, jobject object, jint index, jfloat data){
   /* Get a reference to JVM object class */
  jclass claz = env->GetObjectClass(object);
  if (claz == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "unable to get object's class";
    ThrowCosJavaException((char*)"unable to get object's class", env);
    return;
  }
  /* Getting the field id in the class */
  jfieldID fieldId = env->GetFieldID(claz, "arrayAddress", "J");
  if (fieldId == NULL || env->ExceptionCheck()) {
    LOG(ERROR) << "could not locate field 'arrayAddress'";
    ThrowCosJavaException((char*)"could not locate field 'arrayAddress'", env);
    return;
  }
  
  jfloat* float_array_ptr = (jfloat*) env->GetLongField(object, fieldId);
  if (env->ExceptionCheck()) {
    LOG(ERROR) << "GetLongField failed";
    ThrowCosJavaException((char*)"GetLongField failed", env);
    return;
  }
  if (index < 0) {
    LOG(ERROR) << "Invalid index for floatarray";
    ThrowCosJavaException((char*)"Invalid index for floatarray" , env);
    return;
  }

  float_array_ptr[index] = data;
}

