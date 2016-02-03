// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/data_transformer.hpp"
#include "common.hpp"
#include "jni/com_yahoo_ml_jcaffe_FloatDataTransformer.h"

using  namespace caffe;

/*
 * Class:     com_yahoo_ml_jcaffe_FloatDataTransformer
 * Method:    allocate
 * Signature: ([BZ)Z
 */
JNIEXPORT jboolean JNICALL Java_com_yahoo_ml_jcaffe_FloatDataTransformer_allocate
  (JNIEnv *env, jobject object, jstring xform_param_str, jboolean isTrain) {

    TransformationParameter param;
    jboolean isCopy = false;
    const char* xform_chars = env->GetStringUTFChars(xform_param_str, &isCopy);
    google::protobuf::TextFormat::ParseFromString(string(xform_chars), &param);

    DataTransformer<float>* xformer = NULL;
    if (isTrain)
        xformer = new DataTransformer<float>(param, TRAIN);
    else
        xformer = new DataTransformer<float>(param, TEST);

    //initialize randomizer
    xformer->InitRand();

    if (isCopy){
        env->ReleaseStringUTFChars(xform_param_str, xform_chars);
	if (env->ExceptionOccurred()) {
	  LOG(ERROR) << "Unable to release String";
	  return false;
	}
    }
    /* associate native object with JVM object */
    return SetNativeAddress(env, object, xformer);
}

/*
 * Class:     com_yahoo_ml_jcaffe_FloatDataTransformer
 * Method:    deallocate
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_yahoo_ml_jcaffe_FloatDataTransformer_deallocate
  (JNIEnv *env, jobject object, jlong native_ptr) {
   delete (DataTransformer<float>*) native_ptr;
}

/*
 * Class:     com_yahoo_ml_jcaffe_FloatDataTransformer
 * Method:    transform
 * Signature: (Lcom/yahoo/ml/jcaffe/MatVector;Lcom/yahoo/ml/jcaffe/FloatBlob;)V
 */
JNIEXPORT void JNICALL Java_com_yahoo_ml_jcaffe_FloatDataTransformer_transform
  (JNIEnv *env, jobject object, jobject matVec, jobject transformed_blob) {

    DataTransformer<float>* xformer = (DataTransformer<float>*) GetNativeAddress(env, object);

    vector<cv::Mat>* mat_vector_ptr = (vector<cv::Mat>*) GetNativeAddress(env, matVec);

    Blob<float>* blob_ptr = (Blob<float>*) GetNativeAddress(env, transformed_blob);

    xformer->Transform((* mat_vector_ptr), blob_ptr);
}
