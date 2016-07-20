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

public class MatVectorTest {
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
  private void matVecNegativeIndex() throws Exception {
    boolean fail = false;
    try {
      MatVector matVec = new MatVector(-1);
    } catch(Exception e) {
      fail = true;
    }
    assertTrue(fail);
  }

  @Test 
  private void matNullInMatVec() throws Exception {
    MatVector matVector = new MatVector(1);
    boolean fail = false;
    try {
      matVector.put(0,null);
    } catch(Exception e) {
      fail = true;
    }
    assertTrue(fail);
  }

  @Test
  private void matOnWrongMatVecIndex() throws Exception {
    MatVector matVector = new MatVector(1);
    byte[] data = getDataFromFile(0);
    Mat mat = new Mat(data);
    boolean fail = false;
    try {
      matVector.put(1,mat);
    } catch(Exception e) {
      fail = true;
    }
    assertTrue(fail);
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
  private void getMatVecDatafromInvalidIndex() throws Exception {
    MatVector matVec = new MatVector(1);
    byte[] data0 = getDataFromFile(0);
    Mat mat = new Mat(data0);
    matVec.put(0, mat);
    byte[] resultData0 = matVec.data(-1);
    assertEquals(resultData0, null);
  }
  
  @Test
  private void invalidheight() throws Exception {
    MatVector matVec = new MatVector(1);
    byte[] data0 = getDataFromFile(0);
    Mat mat = new Mat(data0);
    matVec.put(0, mat);
    assertEquals(matVec.height(-1),-1);
  }

  @Test
  private void invalidwidth() throws Exception {
    MatVector matVec = new MatVector(1);
    byte[] data0 = getDataFromFile(0);
    Mat mat = new Mat(data0);
    matVec.put(0, mat);
    assertEquals(matVec.width(-1),-1);
  }

  @Test
  private void invalidchannel() throws Exception {
    MatVector matVec = new MatVector(1);
    byte[] data0 = getDataFromFile(0);
    Mat mat = new Mat(data0);
    matVec.put(0, mat);
    assertEquals(matVec.channels(-1),-1);
  }
  
}
