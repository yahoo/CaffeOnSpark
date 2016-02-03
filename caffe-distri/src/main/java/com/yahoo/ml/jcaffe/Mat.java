// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

public class Mat extends BaseObject {
    /* decode flags as defined in opencv2/imgcodecs/imgcodecs_c.h */
    /* 8bit, color or not */
    static final public int CV_LOAD_IMAGE_UNCHANGED  =-1;
    /* 8bit, gray */
    static final public int CV_LOAD_IMAGE_GRAYSCALE  =0;
    /* ?, color */
    static final public int CV_LOAD_IMAGE_COLOR      =1;
    /* any depth, ? */
    static final public int CV_LOAD_IMAGE_ANYDEPTH   =2;
    /* ?, any color */
    static final public int CV_LOAD_IMAGE_ANYCOLOR   =4;

    /**
     * Constructor of a Mat without a specific dimensions
     * @param data
     */
    public Mat(byte[] data) {
        this(data, false);
    }

    /**
     * Constructor of a Mat without a specific dimensions
     * @param data
     * @param signed
     */
    public Mat(byte[] data,  boolean signed) {
        this(data.length, 1, data, signed);
    }

    /**
     * Constructor of a Mat with a specific dimensions
     * @param height
     * @param width
     * @param data
     * @param signed
     */
    public Mat(int height, int width, byte[] data, boolean signed) {
        dataaddress = allocate(height, width, data, signed);
        if (dataaddress == 0)
            throw new RuntimeException("Failed to create Mat object");
    }

    private native long allocate(int height, int width, byte[] data, boolean signed);

    private long dataaddress = 0;
    
    @Override
    protected void deallocate(long address){
        deallocate(address, dataaddress);
    }

    private native void deallocate(long address, long dataaddress);

    /**
     * decode this Mat in place
     *
     * decoded = opencv_imgcodecs.imdecode(encoded, CV_LOAD_IMAGE_COLOR)
     * @param flags flags to be used by imdecode
     */
    public synchronized void decode(int flags){
	decode(flags, dataaddress);
	dataaddress = 0;
    }

    private native void decode(int flags, long dataaddress);

    /**
     * resize this Mat via opencv
     *
     * opencv_imgproc.resize(decoded, decoded, new opencv_core.Size(width, height))
     * @param height
     * @param width
     */
    public synchronized void resize(int height, int width){
        resize(height, width, dataaddress);
	dataaddress = 0;
    }
    private native void resize(int height, int width, long dataaddress);

    /* dimension of this Mat */
    public native int height();
    public native int width();
    /* get Mat data */ 
    public native byte[] data();

    public native int channels();
}
