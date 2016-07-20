// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifndef CAFFE_DISTRI_COMMON_HPP_
#define CAFFE_DISTRI_COMMON_HPP_

#include <jni.h>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"

using  namespace caffe;

#ifdef __cplusplus
extern "C" {
#endif
  
  bool SetNativeAddress(JNIEnv *env, jobject object, void* address);
  void* GetNativeAddress(JNIEnv *env, jobject object);
  
  bool GetStringVector(vector<const char*>& vec, JNIEnv *env, jobjectArray array, int length);
  bool GetFloatBlobVector(vector< Blob<float>* >& vec, JNIEnv *env, jobjectArray array, int length);
  void ThrowJavaException(const std::exception& ex, JNIEnv* env);
  void ThrowCosJavaException(char* message, JNIEnv* env);
#ifdef __cplusplus
}
#endif
#endif
