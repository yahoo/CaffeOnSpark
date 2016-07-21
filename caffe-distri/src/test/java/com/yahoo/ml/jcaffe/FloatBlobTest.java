// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import org.testng.annotations.Test;
import static org.testng.Assert.*;
import java.util.Random;

public class FloatBlobTest {
    @Test
    public void testAllocate() {
        FloatBlob blob = new FloatBlob();
        blob.deallocate();
    }

    @Test
    public void setcpudatanull() {
	FloatBlob blob = new FloatBlob();
	boolean res = blob.set_cpu_data(null);
	assertFalse(res);
    }
    
    @Test
    public void copyfromnull() {
	FloatBlob blob2 = new FloatBlob();
	blob2.reshape(1, 1, 2, 2);
	assertFalse(blob2.copyFrom(null));
    }
    
    @Test
    public void floatblobnull(){
	boolean fail = false;
	try {
	    FloatBlob blob1 = new FloatBlob(-1,false);
	} catch(Exception e) {
	    fail = true;
	}
	assertFalse(fail);
    }
    
    @Test
    public void reshapeinvalid(){
	boolean fail = false;
	FloatBlob blob = new FloatBlob();
	try {
	    fail = blob.reshape(0,0,0,0);
	} catch(Exception e){
	    fail = true;
	}
	assertFalse(fail);
    }
    
    
    @Test
    public void testBasic() {
	FloatBlob blob = new FloatBlob();
	boolean res = blob.reshape(1, 1, 2, 2);
	assertTrue(res);
	
	float[] input = { 1.0f, 2.0f, 3.0f, 4.0f };
	res = blob.set_cpu_data(input);
	assertTrue(res);
	
	FloatArray output = blob.cpu_data();
	for (int i=0; i<input.length; i++)
	    assertEquals(output.get(i), input[i], 0.1);
	
    }
    
    @Test
    public void testCopy() {
        float[] input = { 1.0f, 2.0f, 3.0f, 4.0f };
        FloatBlob blob1 = new FloatBlob();
        blob1.reshape(1, 1, 2, 2);
        blob1.set_cpu_data(input);
	
        FloatBlob blob2 = new FloatBlob();
        blob2.reshape(1, 1, 2, 2);
        assertTrue(blob2.copyFrom(blob1));
	
        FloatArray output = blob2.cpu_data();
        for (int i=0; i<input.length; i++)
            assertEquals(output.get(i), input[i], 0.1);
    }
    
}
