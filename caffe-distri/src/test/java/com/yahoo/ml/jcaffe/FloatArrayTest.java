// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import org.testng.annotations.Test;
import static org.testng.Assert.*;
import java.util.Random;

public class FloatArrayTest {
  @Test
  public void floatarraygetnegative(){
    FloatBlob data_blob = new FloatBlob();
    data_blob.reshape(5, 1, 1, 1);
    FloatArray fa = null;
    fa = data_blob.cpu_data();
    boolean fail = false;
    if (fa.get(-1) == 0)
      fail = true;

    assertTrue(fail);
  }

  @Test
  public void floatarraysetinvalid(){
    FloatBlob data_blob = new FloatBlob();
    data_blob.reshape(5, 1, 1, 1);
    FloatArray fa = null;
    boolean fail = false;
    fa = data_blob.cpu_data();
    if (fa == null) {
      System.out.println("cpu_data returned null");
      return;
    }
    try {
      fa.set(-1,-1);
    } catch(Exception e) {
      fail = true;
    }
    assertTrue(fail);
    data_blob.deallocate();
  }
}
