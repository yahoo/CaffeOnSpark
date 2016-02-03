// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import java.lang.Override;

public abstract class BaseObject {
    static {
        System.loadLibrary("caffedistri");
    }

    /* native address */
    protected long address = 0;

    public BaseObject() {
        this(0);
    }
    public BaseObject(long native_addr) {
        address = native_addr;
    }

    /**
     * Called by native libraries to initialize the object fields.
     *
     * @param allocatedAddress the new address value of allocated native memory
     */
    void init(long allocatedAddress) {
        address = allocatedAddress;
    }

    /**
     * Invoked by GC
     */
    @Override
    protected void 	finalize() {
        deallocate();
    }

    /**
     * Application could explicitly release native object
     */
    public synchronized void deallocate() {
        if (address != 0) {
            deallocate(address);
            address = 0;
        }
    }

    /**
     * Each subclass is required to implement deallocate() method
     * @param address a pointer of native object
     * @return true iff sucessful
     */
    protected abstract void deallocate(long address);
}