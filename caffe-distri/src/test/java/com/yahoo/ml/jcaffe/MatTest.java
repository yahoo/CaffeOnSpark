// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import caffe.Caffe.*;
import com.google.protobuf.TextFormat;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Arrays;
import static org.testng.Assert.*;

public class MatTest {
  String rootPath, imagePath;
  final int batchs = 1;
  final int batch_size = 2;
  final int channels = 3;
  final int height = 227;
  final int width = 227;

  int index = 0;
  List<String> file_list;

  @BeforeMethod
  public void setUp() throws Exception {
    String fullPath = getClass().getClassLoader().getResource("log4j.properties").getPath();
    rootPath = fullPath.substring(0, fullPath.indexOf("caffe-distri/"));
    imagePath = rootPath + "data/images";

    file_list = Files.readAllLines(Paths.get(imagePath + "/labels.txt"), StandardCharsets.UTF_8);
  }

  @Test 
  private void matNull() throws Exception {
    boolean fail = false;
    try {
      Mat mat = new Mat(null);
    } catch(Exception e) {
      fail = true;
    }
    assertTrue(fail);
  }

  @Test
  private void basicTest() throws Exception {
    MatVector matVec = new MatVector(1);
    byte[] buf = new byte[1024 * 1024];
    int width = 227;
    int height = 227;
    String line = file_list.get(index++);
    if (index >= file_list.size()) index = 0;
	
    String[] line_splits = line.split(" ");
    String filename = line_splits[0];
    int label = Integer.parseInt(line_splits[1]);

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    DataInputStream in = new DataInputStream(new FileInputStream(imagePath + "/" + filename));
    int len = in.read(buf, 0, buf.length);
    while (len > 0) {
      out.write(buf, 0, len);
      len = in.read(buf, 0, buf.length);
    }
    in.close();

    byte[] data = out.toByteArray();
	
    Mat mat = new Mat(data);
    //	mat.decode(Mat.CV_LOAD_IMAGE_COLOR);
    //mat.resize(width, height);
    Mat oldmat = matVec.put(0, mat);
    if (oldmat != null)
      oldmat.deallocate();
    assertEquals(matVec.width(0), mat.width());	    
    assertEquals(matVec.height(0), mat.height());	    
    //GC doesn't have any affect on mat with value 227
    width++;
    height++;

    //reuse matVec for new mat with value 228 and clean old mat with 227 properly
    mat = new Mat(data);
    mat.decode(Mat.CV_LOAD_IMAGE_COLOR);
    mat.resize(width, height);
    oldmat = matVec.put(0, mat);
    oldmat.deallocate();
    assertEquals(matVec.width(0), 228);	    
    assertEquals(matVec.height(0), 228);
    //GC to deallocate mat with value 227. Currently matVec has mat with 228
    //Irrespective of GC,  mat with 228 shouldn't get deallocated before matVec deallocate
    mat = null;
    assertEquals(matVec.width(0), 228);	    
    assertEquals(matVec.height(0), 228);	    	
    matVec.deallocate();
    out.close();
  }

  private byte[] getDataFromFile(int pos) throws Exception{
    byte[] buf = new byte[1024 * 1024];
    String line = file_list.get(pos);
    String[] line_splits = line.split(" ");
    String filename = line_splits[0];
    int label = Integer.parseInt(line_splits[1]);
	
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    DataInputStream in = new DataInputStream(new FileInputStream(imagePath + "/" + filename));
    int len = in.read(buf, 0, buf.length);
    while (len > 0) {
      out.write(buf, 0, len);
      len = in.read(buf, 0, buf.length);
    }
    in.close();
    byte[] b = out.toByteArray();
    out.close();
    return b;
  }


  @Test
  private void matResizeDecodeTest() throws Exception {
    byte[] data0 = getDataFromFile(0);
    Mat m = new Mat(data0);
    m.resize(227,227);
    m.decode(Mat.CV_LOAD_IMAGE_COLOR);
    m.deallocate();
  }

  @Test
  private void matChannelsTest() throws Exception {
    byte[] data0 = getDataFromFile(0);
    Mat m = new Mat(3, 9, 9, data0);
    assertEquals(m.channels(), 3);
    m.deallocate();
  }

  @Test
  private void basicDataTest() throws Exception {
    MatVector matVec = new MatVector(1);
    byte[] data0 = getDataFromFile(0);
    Mat mat = new Mat(data0);
    matVec.put(0, mat);
    byte[] resultData0 = matVec.data(0);
    //What we wrote is what we get
    assertTrue(Arrays.equals(data0, resultData0));

    //Now replace matVec 0th mat object with a new one and make sure it is the new one
    byte[] data1 = getDataFromFile(1);
    mat = new Mat(data1);
    resultData0 = matVec.data(0);
    assertTrue(Arrays.equals(data0, resultData0));
    Mat oldmat = matVec.put(0, mat);
    byte[] resultData1 = matVec.data(0);
    assertTrue(Arrays.equals(data1, resultData1));
    matVec.deallocate();
    assertTrue(Arrays.equals(data0, oldmat.data()));
    oldmat.deallocate();
  }

  @Test
  private void getMatDecodeWithInvalidFlag() throws Exception {
    byte[] data0 = getDataFromFile(0);
    Mat mat = new Mat(data0);
    boolean fail = false;
    try {
      mat.decode(-1);
    } catch(Exception e) {
      fail = true;
    }
    assertFalse(fail);
  }
    
  @Test
  private void matResizeInvalid() throws Exception {
    byte[] data0 = getDataFromFile(0);
    Mat m = new Mat(data0);
    boolean fail = false;
    try {
      m.resize(-1,-1);
    } catch(Exception e) {
      fail = true;
    }
    assertTrue(fail);
  }

}
