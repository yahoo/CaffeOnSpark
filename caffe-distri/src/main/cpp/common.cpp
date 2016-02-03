// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include "common.hpp"

#include <glog/logging.h>

bool SetNativeAddress(JNIEnv *env, jobject object, void* address) {

    if (object == NULL) {
      LOG(ERROR) << "object is NULL";
      return false;
    }
    /* Get a reference to JVM object class */
    jclass claz = env->GetObjectClass(object);
    if (claz == NULL) {
      LOG(ERROR) << "unable to get object's class";
      return false;
    }
    /* Locate init(long) method */
    jmethodID methodId = env->GetMethodID(claz, "init", "(J)V");
    if (methodId == NULL) {
        LOG(ERROR) << "could not locate init() method";
        return false;
    }

    /* associate native object with JVM object */
    env->CallVoidMethod(object, methodId, (long)address);
    if (env->ExceptionCheck()) {
      LOG(ERROR) << "CallVoidMethod failed";
      return false;
    }
    return true;
}

void* GetNativeAddress(JNIEnv *env, jobject object) {
    if (object == NULL) {
      LOG(ERROR) << "object is NULL";
      return 0;
    }
    /* Get a reference to JVM object class */
    jclass claz = env->GetObjectClass(object);
    if (claz == NULL) {
      LOG(ERROR) << "unable to get object's class";
      return 0;
    }
    /* Getting the field id in the class */
    jfieldID fieldId = env->GetFieldID(claz, "address", "J");
    if (fieldId == NULL) {
        LOG(ERROR) << "could not locate field 'address'";
        return 0;
    }

    return (void*) env->GetLongField(object, fieldId);
}

bool GetStringVector(vector<const char*>& vec, JNIEnv *env, jobjectArray array, int length) {
    for (int i = 0; i < length; i++) {
        jstring addr = (jstring)env->GetObjectArrayElement(array, i);
        if (addr == NULL) {
            LOG(INFO) << i << "-th string is NULL";
            vec[i] = NULL;
        } else {
            const char *cStr = env->GetStringUTFChars(addr, NULL);
            vec[i] = cStr;
            //Too many local refs could get created due to the loop, so delete them
            //CHECKME:cStr is also a local ref called in loop, but it's not clear if deleting it via DeleteLocalRef deletes the memory pointed by it too
            env->DeleteLocalRef(addr);
        }
    }
    
    return true;
}

bool GetFloatBlobVector(vector< Blob<float>* >& vec, JNIEnv *env, jobjectArray array, int length) {
   if (array == NULL) {
      LOG(ERROR) << "array is NULL";
      return false;
   }

  for (int i = 0; i < length; i++) {
        //get i-th FloatBlob object (JVM)
        jobject object = env->GetObjectArrayElement(array, i);
        if (object == NULL) {
            LOG(WARNING)  << i << "-th FloatBlob is NULL";
            vec[i] = NULL;
        } else {
            //find the native Blob<float> object
            vec[i] = (Blob<float>*) GetNativeAddress(env, object);
        }
	//Too many local refs could get created due to the loop, so delete them
	env->DeleteLocalRef(object);
    }

   return true;
}

