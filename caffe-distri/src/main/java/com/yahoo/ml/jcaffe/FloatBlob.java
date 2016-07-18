// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import java.lang.RuntimeException;

public class FloatBlob extends BaseObject {
    private boolean shouldDeallocate;

    /**
     * constructor of FloatBlob
     */
    public FloatBlob() {
        this(0, true);
    }

    public FloatBlob(long native_addr, boolean shouldDeallocate) {
        super(native_addr);
        this.shouldDeallocate = shouldDeallocate;
        if (native_addr == 0) {
            if (!allocate()) throw new RuntimeException("Failed to create FloatBlob object");
        }
    }
    private native boolean allocate();

    private long dataaddress = 0;

    @Override
    protected void deallocate(long address) {
        if (shouldDeallocate)
            deallocate1(address, dataaddress);
    }

    private native void deallocate1(long address, long dataaddress);

    /**
     * Reshape a floatblob
     * @param num
     * @param channels
     * @param height
     * @param width
     * @return true iff successful
     */
    public boolean reshape(int num, int channels, int height, int width) {
        int[] shape = {  num, channels, height, width};
        return reshape(shape);
    }

    /**
     * copy content from another floatblob
     * @param source
     * @return true iff successful
     */
    public native boolean copyFrom(FloatBlob source);

    public native FloatArray cpu_data();

    public boolean set_cpu_data(float[] data){
	dataaddress = set_cpu_data(data, dataaddress);
	if(dataaddress != 0)
	    return true;
	return false;
    }
	
    protected native long set_cpu_data(float[] data, long dataaddress);

    public native FloatArray gpu_data();

    /* the # of floats in this FloatBlob */
    public native int count();

    /**
     * @brief Change the dimensions of the blob, allocating new memory if
     *        necessary.
     *
     * This function can be called both to create an initial allocation
     * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
     * or Layer::Forward. When changing the size of blob, memory will only be
     * reallocated if sufficient memory does not already exist, and excess memory
     * will never be freed.
     *
     * Note that reshaping an input blob and immediately calling Net::Backward is
     * an error; either Net::Forward or Net::Reshape need to be called to
     * propagate the new input shape to higher layers.
     */
    public native boolean reshape(int[] shape);
}

