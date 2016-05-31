// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <glog/logging.h>

#include "common.hpp"
#include "jni/com_yahoo_ml_jcaffe_Mat.h"
/*
 * Class:     com_yahoo_ml_jcaffe_Mat
 * Method:    allocate
 * Signature: (III[)Z
 */
JNIEXPORT jlong JNICALL Java_com_yahoo_ml_jcaffe_Mat_allocate
  (JNIEnv *env, jobject object, jint channels, jint height, jint width,  jbyteArray array) {

    jboolean isCopy = false;
    jbyte *data = env->GetByteArrayElements(array, &isCopy);
    
    if (data==NULL) {
      LOG(ERROR) << "invalid data array";
      return 0;
    }

    if (!isCopy) {
        jsize len = env->GetArrayLength(array);
        jbyte* new_data = new jbyte[len];
        if (new_data == NULL) {
            LOG(ERROR) << "fail to jbyte[] for new data";
            return 0;
        }

        memcpy(new_data, data, len * sizeof(jbyte));
        //set new data
        data = new_data;
    }

    /* create a native Mat object */
    cv::Mat* native_ptr = new cv::Mat(height, width, CV_8UC(channels), data);

    /* associate native object with JVM object */
    SetNativeAddress(env, object, native_ptr);

    return (long) data;
}

/*
 * Class:     com_yahoo_ml_jcaffe_Mat
 * Method:    deallocate
 * Signature: (JZ)V
 */
JNIEXPORT void JNICALL Java_com_yahoo_ml_jcaffe_Mat_deallocate
   (JNIEnv *env, jobject object, jlong native_ptr, jlong dataaddress) {

    //Mat object is only one responsible for cleaning itself and it's data 
    if(dataaddress){
        delete[] (jbyte*)dataaddress;
    }
    delete (cv::Mat*) native_ptr;
}

/*
 * Class:     com_yahoo_ml_jcaffe_Mat
 * Method:    decode
 * Signature: (IJ)Lcom/yahoo/ml/jcaffe/Mat;
 */
JNIEXPORT void JNICALL Java_com_yahoo_ml_jcaffe_Mat_decode
  (JNIEnv *env, jobject object, jint flags, jlong dataaddress) {

      cv::Mat* native_ptr = (cv::Mat*) GetNativeAddress(env, object);

      cv::imdecode(cv::_InputArray(*native_ptr), flags, native_ptr);

      jclass claz = env->GetObjectClass(object);
      if (claz == NULL) {
	LOG(ERROR) << "unable to get object's class";
	return;
      }
      
      if (dataaddress){
	delete (jbyte*)dataaddress;
      }
}

/* * Class:     com_yahoo_ml_jcaffe_Mat
 * Method:    resize
 * Signature: (II)Lcom/yahoo/ml/jcaffe/Mat;
 */
JNIEXPORT void JNICALL Java_com_yahoo_ml_jcaffe_Mat_resize
   (JNIEnv *env, jobject object, jint height, jint width, jlong dataaddress) {
      cv::Mat* native_ptr = (cv::Mat*) GetNativeAddress(env, object);
      
      cv::Size size(width, height);
      cv::resize(cv::_InputArray(*native_ptr), cv::_OutputArray(*native_ptr), size, 0, 0, cv::INTER_LINEAR);

      jclass claz = env->GetObjectClass(object);
      if (claz == NULL) {
        LOG(ERROR) << "unable to get object's class";
        return;
      }
      
     if (dataaddress){
         delete (jbyte*)dataaddress;
     }
}

/*
 * Class:     com_yahoo_ml_jcaffe_Mat
 * Method:    height
 * Signature: ()I
 */

JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_Mat_height
  (JNIEnv *env, jobject object) {
    cv::Mat* native_ptr = (cv::Mat*) GetNativeAddress(env, object);
    
    return native_ptr->rows;
}

/*
 * Class:     com_yahoo_ml_jcaffe_Mat
 * Method:    width
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_Mat_width
  (JNIEnv *env, jobject object) {
    cv::Mat* native_ptr = (cv::Mat*) GetNativeAddress(env, object);
    
    return native_ptr->cols;
}

/*
 * Class:     com_yahoo_ml_jcaffe_Mat
 * Method:    data
 * Signature: ()[B
 */

JNIEXPORT jbyteArray JNICALL Java_com_yahoo_ml_jcaffe_Mat_data
   (JNIEnv *env, jobject object) {
   cv::Mat* native_ptr = (cv::Mat*) GetNativeAddress(env, object);
   int size = native_ptr->total() * native_ptr->elemSize();

   jbyteArray dataarray = env->NewByteArray(size);
   if(dataarray == NULL){
     LOG(ERROR) << "Out of memory while allocating array for Mat data" ;
     return NULL;
   }
   env->SetByteArrayRegion(dataarray,0, size, (jbyte*)native_ptr->data);
   return dataarray;
}

/*
 * Class:     com_yahoo_ml_jcaffe_Mat
 * Method:    channels
 * Signature: ()I
 */

JNIEXPORT jint JNICALL Java_com_yahoo_ml_jcaffe_Mat_channels
(JNIEnv *env, jobject object) {
  cv::Mat* native_ptr = (cv::Mat*) GetNativeAddress(env, object);

  return native_ptr->channels();
}
