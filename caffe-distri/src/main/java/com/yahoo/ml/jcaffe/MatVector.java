// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;
import java.util.List;
import java.util.ArrayList;

public class MatVector extends BaseObject {

    //Internal array for reference counting    
    private Mat [] matList = null;

    /**
     * Constructor of a Mat vector
     * @param size of vector
     */
    public MatVector(int size) {
        if (!allocate(size))
            throw new RuntimeException("Failed to create MatVector object");
	matList = new Mat[size];
    }

    private native boolean allocate(int size);

    @Override
    protected void deallocate(long address){
	//Free up the references to all Mat objects in the matList
	for(int i=0; i<matList.length;i++){
            matList[i] = null;
	}
	deallocateVec(address);
    }

    protected native void deallocateVec(long address);

    /**
     * put a Mat into a specific position
     * @param pos position in the vector
     * @param mat a Mat object
     */
    public Mat put(int pos, Mat mat){
	this.putnative(pos,mat);

	Mat oldmat = null;
	if (matList[pos] != null)
	    oldmat = matList[pos];

	matList[pos]=mat;
	return oldmat;
    }

    private native void putnative(int pos, Mat mat);

    //Get dimensions of Mat at pos
    public native int width(int pos);
	
    public native int height(int pos);

    //Get data of Mat at pos
    public native byte[] data(int pos);

    public native int channels(int pos);
}
