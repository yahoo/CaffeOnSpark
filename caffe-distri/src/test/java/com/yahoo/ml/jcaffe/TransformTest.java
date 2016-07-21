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

import static org.testng.Assert.*;

public class TransformTest {
  String rootPath, imagePath;
  MatVector matVec;
  final int batchs = 1000;
  final int batch_size = 4;
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
    matVec = new MatVector(batch_size);
  }

  @AfterMethod
  public void tearDown() throws Exception {
    matVec.deallocate();
  }

  private void nextBatch() throws Exception {
    byte[] buf = new byte[1024 * 1024];

    for (int idx=0; idx<batch_size; idx ++) {
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
      mat.decode(Mat.CV_LOAD_IMAGE_COLOR);
      mat.resize(227, 227);

      Mat oldmat = matVec.put(idx, mat);
      if(oldmat != null)
	oldmat.deallocate();

      out.close();
    }
  }

  @Test
  public void floatdatatransformernull(){
    boolean fail = false;
    try {
      FloatDataTransformer test_xform = new FloatDataTransformer(null, false);
    } catch(Exception e) {
      fail = true;
    }
    assertTrue(fail);
  }

  @Test
  public void transformnull() throws Exception {
    SolverParameter.Builder solver_builder = SolverParameter.newBuilder();
    FileReader reader = new FileReader(rootPath + "caffe-distri/src/test/resources/caffenet_solver.prototxt");
    TextFormat.merge(reader, solver_builder);
    reader.close();

    SolverParameter solver_param = solver_builder.build();
    String net_proto_file = solver_param.getNet();

    NetParameter.Builder net_builder = NetParameter.newBuilder();
    reader = new FileReader(rootPath + "caffe-distri/" + net_proto_file);
    TextFormat.merge(reader, net_builder);
    reader.close();

    NetParameter net_param = net_builder.build();

    //blob
    FloatBlob blob = new FloatBlob();
    blob.reshape(batch_size, channels, height, width);

    //train
    LayerParameter train_layer_param = net_param.getLayer(0);
    TransformationParameter param = train_layer_param.getTransformParam();
    FloatDataTransformer trans_xform = new FloatDataTransformer(param, true);
    nextBatch();
    boolean fail=false;
    try {
	trans_xform.transform((MatVector) null, null);
    } catch(Exception e) {
      fail = true;
    }
    assertTrue(fail);
  }

  @Test
  public void testTransform() throws Exception {
    SolverParameter.Builder solver_builder = SolverParameter.newBuilder();
    FileReader reader = new FileReader(rootPath + "caffe-distri/src/test/resources/caffenet_solver.prototxt");
    TextFormat.merge(reader, solver_builder);
    reader.close();

    SolverParameter solver_param = solver_builder.build();
    String net_proto_file = solver_param.getNet();

    NetParameter.Builder net_builder = NetParameter.newBuilder();
    reader = new FileReader(rootPath + "caffe-distri/" + net_proto_file);
    TextFormat.merge(reader, net_builder);
    reader.close();

    NetParameter net_param = net_builder.build();

    //blob
    FloatBlob blob = new FloatBlob();
    blob.reshape(batch_size, channels, height, width);

    //train
    LayerParameter train_layer_param = net_param.getLayer(0);
    TransformationParameter param = train_layer_param.getTransformParam();
    FloatDataTransformer trans_xform = new FloatDataTransformer(param, true);
    System.out.print("TRAINING:");
    for (int i=0; i<batchs; i++) {
      System.out.print(".");
      nextBatch();
      trans_xform.transform(matVec, blob);
      assertTrue(blob.count() > 1000);
    }
    System.out.println();

    //test
    FloatDataTransformer test_xform = new FloatDataTransformer(param, false);
    System.out.print("TEST:");
    for (int i =0; i<batchs; i++) {
      System.out.print(".");
      test_xform.transform(matVec, blob);
      assertTrue(blob.count() > 1000);
    }
    System.out.println();

    //release C++ resource
    trans_xform.deallocate();
    test_xform.deallocate();
    blob.deallocate();
  }
}
