// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import caffe.Caffe.*;
import java.io.IOException;

public class FloatDataTransformer extends BaseObject {
    /**
     * Constructor of a Float data transforsmer
     * @param parameter
     * @param isTrain
     */
    public FloatDataTransformer(TransformationParameter parameter, boolean isTrain) throws IOException {
        String parameter_str = parameter.toString();
        if (!allocate(parameter_str, isTrain))
            throw new RuntimeException("Failed to create FloatDataTransformer object");
    }

    private native boolean allocate(String parameter_str, boolean isTrain);

    @Override
    protected native void deallocate(long address);

    /**
     * @brief Applies the transformation defined in the data layer's
     * transform_param block to a vector of Mat.
     *
     * @param mat_vector
     *    A vector of Mat containing the data to be transformed.
     * @param transformed_blob
     *    This is destination blob. It can be part of top blob's data if
     *    set_cpu_data() is used. See memory_layer.cpp for an example.
     */
    public native void transform(MatVector mat_vector, FloatBlob transformed_blob);

    public void transform(FloatBlob input_blob, FloatBlob transformed_blob) throws Exception {
        throw new Exception("transform from floatblob to floatblob is not implemented");
    }
}