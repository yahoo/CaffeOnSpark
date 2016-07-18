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
    Blob<float>* native_ptr = new Blob<float>();

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
    Blob<float>* native_ptr = (Blob<float>*) GetNativeAddress(env, object);

    return native_ptr->count();
}

/*
 * Class:     com_yahoo_ml_jcaffe_FloatBlob
 * Method:    reshape
 * Signature: ([I)V
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_FloatBlob_reshape
  (JNIEnv *env, jobject object, jintArray shape) {
    Blob<float>* native_ptr = (Blob<float>*) GetNativeAddress(env, object);

    size_t size = env->GetArrayLength(shape);
    jint *vals = env->GetIntArrayElements(shape, NULL);
    if (vals == NULL) {
      LOG(ERROR) << "vals == NULL";
      return false;
    }

    vector<jint> shap_vec(size);
    for (int i=0; i<size; i++)
        shap_vec[i] = vals[i];

    native_ptr->Reshape(shap_vec);

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
    Blob<float>* native_ptr = (Blob<float>*) GetNativeAddress(env, object);

    Blob<float>* source_ptr = (Blob<float>*) GetNativeAddress(env, source);

    //perform operation
    native_ptr->CopyFrom(*source_ptr);

    return true;
}

/*
 * Class:     com_yahoo_ml_jcaffe_FloatBlob
 * Method:    cpu_data
 * Signature: ()[F
 */
JNIEXPORT jobject JNICALL Java_com_yahoo_ml_jcaffe_FloatBlob_cpu_1data(JNIEnv *env, jobject object) {
    Blob<float>* native_ptr = (Blob<float>*) GetNativeAddress(env, object);

    //retrieve cpu_data()
    jfloat* cpu_data = (jfloat*) native_ptr->mutable_cpu_data();
    if (cpu_data == NULL) {
        LOG(ERROR) << "cpu_data == NULL";
        return NULL;
    }

    jclass claz = env->FindClass("com/yahoo/ml/jcaffe/FloatArray");
    jmethodID constructorId = env->GetMethodID( claz, "<init>", "(J)V");
    jobject objectFloatArray = env->NewObject(claz,constructorId,(long)cpu_data);
    
    return objectFloatArray;
}

/*
 * Class:     com_yahoo_ml_jcaffe_FloatBlob
 * Method:    set_cpu_data
 * Signature: ([F)V
 */
JNIEXPORT jlong JNICALL Java_com_yahoo_ml_jcaffe_FloatBlob_set_1cpu_1data(JNIEnv *env, jobject object, jfloatArray array, jlong dataaddress) {

    Blob<float>* native_ptr = (Blob<float>*) GetNativeAddress(env, object);

    jboolean copied = false;
    float* data = env->GetFloatArrayElements(array, &copied);
    if (data == NULL) {
        LOG(ERROR) << "GetFloatArrayElements() == NULL";
        return 0;
    }

    if (!copied) {
        size_t len = native_ptr->count();
        float* new_data = new float[len];
        if (new_data == NULL) {
            LOG(ERROR) << "fail to float[] for new data";
            return 0;
        }

        memcpy(new_data, data, len * sizeof(float));
        //set new data
        data = new_data;
    }

    native_ptr->set_cpu_data(data);

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
    Blob<float>* native_ptr = (Blob<float>*) GetNativeAddress(env, object);

    //retrieve gpu_data()
    jfloat* gpu_data = (jfloat*)native_ptr->mutable_gpu_data();
    if (gpu_data == NULL) {
        LOG(ERROR) << "gpu_data == NULL";
        return NULL;
    }

    jclass claz = env->FindClass("com/yahoo/ml/jcaffe/FloatArray");
    jmethodID constructorId = env->GetMethodID( claz, "<init>", "(J)V");
    jobject objectFloatArray = env->NewObject(claz,constructorId,(long)gpu_data);

    return objectFloatArray;
}
