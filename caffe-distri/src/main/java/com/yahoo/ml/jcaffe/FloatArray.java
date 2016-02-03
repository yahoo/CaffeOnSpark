// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import java.lang.RuntimeException;

public class FloatArray extends BaseObject {
    
    protected long arrayAddress = 0;
    /**
     * constructor of FloatArray
     */
    
    public FloatArray(long arrayAddress) {
	super();
	this.arrayAddress = arrayAddress;
    }

    public native float get(int index);

    public native void set(int index, float data);

    @Override
    protected void deallocate(long address) { }
}

