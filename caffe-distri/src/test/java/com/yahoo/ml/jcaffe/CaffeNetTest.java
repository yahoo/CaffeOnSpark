// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import caffe.Caffe.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import com.google.protobuf.TextFormat;

import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;
import static org.testng.Assert.*;

public class CaffeNetTest {
    String rootPath, solver_config_path, imagePath;
    CaffeNet net, test_net, socket_net;
    SolverParameter solver_param;
    int index = 0;
    List<String> file_list;

    final int batchs = 20;
    final int batch_size = 5;
    final int channels = 3;
    final int height = 227;
    final int width = 227;

    @BeforeMethod
    public void setUp() throws Exception {
        String fullPath = getClass().getClassLoader().getResource("log4j.properties").getPath();
        rootPath = fullPath.substring(0, fullPath.indexOf("caffe-distri/"));
        solver_config_path = rootPath + "caffe-distri/src/test/resources/caffenet_solver.prototxt";
        net = new CaffeNet(solver_config_path,
                "",
                "",
                1, //num_local_devices,
                1, //cluster_size,
                0, //node_rank,
                true, //isTraining,
                0, //NONE
		-1, 0);
        assertTrue(net != null);

        test_net = new CaffeNet(solver_config_path,
                "",
                "",
                1, //num_local_devices,
                1, //cluster_size,
                0, //node_rank,
                false, //isTraining,
                0, //NONE
		-1, 0);
        assertTrue(test_net != null);

	socket_net = new CaffeNet(solver_config_path,
				  "",
				  "",
				  1, //num_local_devices,
				  2, //cluster_size,
				  0, //node_rank,
				  false, //isTraining,
				  CaffeNet.SOCKET, //NONE
				  -1, 0);
	assertTrue(socket_net != null);
        solver_param = Utils.GetSolverParam(solver_config_path);
        assertEquals(solver_param.getSolverMode(), SolverParameter.SolverMode.CPU);

        imagePath = rootPath + "data/images";
        file_list = Files.readAllLines(Paths.get(imagePath + "/labels.txt"), StandardCharsets.UTF_8);
    }

    @AfterMethod
    public void tearDown() throws Exception {
        net.deallocate();
	test_net.deallocate();
	socket_net.deallocate();
    }

    @Test
    public void initinvalid() {
	assertFalse(net.init(-1));
    }

    @Test
    public void deviceIDinvalid() {
      assertEquals(net.deviceID(-1), -1);
    }

    @Test 
    public void inititerinvalid() {
      assertEquals(net.getInitIter(-1), -1);
    }
  
    @Test
    public void maxiterinvalid() {
      assertEquals(net.getMaxIter(-1), -1);
    }

    @Test
    public void snapshotfilenameinvalid() {
      assertNull(net.snapshotFilename(-1,false));
    }

    @Test
    public void connectnull(){
      String[] addrs = null;
      assertTrue(net.connect(addrs));
    }

    @Test
    public void connectbogus(){
      String[] addrs = {"0x222", "0x333"};
      boolean pass = true;
      try {
	  pass = socket_net.connect(addrs);
      } catch(Exception e) {
	  pass = false;
      }
      assertFalse(pass);
    }
  
    @Test
    public void testBasic() {
        String[] addrs = net.localAddresses();
        assertEquals(addrs.length, 0);

        assertTrue(net.connect(addrs));

        assertTrue(net.sync());

        assertEquals(net.deviceID(0), 0);

        assertTrue(net.init(0, true));

        int from_iter = net.getInitIter(0);
        assertEquals(from_iter, 0);

        int max_iter = net.getMaxIter(0);
        assertEquals(max_iter, solver_param.getMaxIter());

        int iterId = net.snapshot();
        assertTrue(iterId >= 0);

        String state_snapshot_fn = net.snapshotFilename(0, true);
        assertTrue(state_snapshot_fn.startsWith(solver_param.getSnapshotPrefix() + "_iter_0.solverstate"));

        String model_snapshot_fn = net.snapshotFilename(0, false);
        assertTrue(model_snapshot_fn.startsWith(solver_param.getSnapshotPrefix() + "_iter_0.caffemodel"));

        String[] testOutputBlobNames = test_net.getValidationOutputBlobNames();
        assertTrue(testOutputBlobNames[0].contains("accuracy"));
        assertTrue(testOutputBlobNames[1].contains("loss"));
    }

    private void nextBatch(MatVector matVec, FloatBlob labels) throws Exception {
        FloatArray labelCPU = labels.cpu_data();
        byte[] buf = new byte[1024 * 1024];

        for (int idx=0; idx<batch_size; idx ++) {
            String line = file_list.get(index++);
            if (index >= file_list.size()) index = 0;

            String[] line_splits = line.split(" ");
            String filename = line_splits[0];
            int label = Integer.parseInt(line_splits[1]);
            labelCPU.set(idx, label);

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
            mat.resize(height, width);

            Mat oldmat=matVec.put(idx, mat);
	    if(oldmat != null)
		oldmat.deallocate();

            out.close();
        }
    }

    @Test
    public void trainnull() throws Exception {
      SolverParameter solver_param = Utils.GetSolverParam(rootPath + "caffe-distri/src/test/resources/caffenet_solver.prototxt");
    
      String net_proto_file = solver_param.getNet();
      NetParameter net_param = Utils.GetNetParam(rootPath + "caffe-distri/" + net_proto_file);
    
      //blob
      MatVector matVec = new MatVector(batch_size);
      FloatBlob[] dataBlobs = new FloatBlob[1];
      FloatBlob data_blob = new FloatBlob();
      data_blob.reshape(batch_size, channels, height, width);
      dataBlobs[0] = data_blob;
    
      FloatBlob labelblob = new FloatBlob();
      labelblob.reshape(batch_size, 1, 1, 1);
    
      //transformer
      LayerParameter train_layer_param = net_param.getLayer(0);
      TransformationParameter param = train_layer_param.getTransformParam();
      FloatDataTransformer xform = new FloatDataTransformer(param, true);
    
      nextBatch(matVec, labelblob);
      xform.transform(matVec, data_blob);
      boolean fail = false;
      try {
	  net.train(0, null);
      } catch(Exception e) {
	  fail = true;
      }
      assertTrue(fail);
      xform.deallocate();
      data_blob.deallocate();
      matVec.deallocate();
  }
  
    @Test
    public void predictnull() throws Exception {
      SolverParameter solver_param = Utils.GetSolverParam(rootPath + "caffe-distri/src/test/resources/caffenet_solver.prototxt");
    
      String net_proto_file = solver_param.getNet();
      NetParameter net_param = Utils.GetNetParam(rootPath + "caffe-distri/" + net_proto_file);
    
      //blob
      MatVector matVec = new MatVector(batch_size);
      FloatBlob[] dataBlobs = new FloatBlob[1];
      FloatBlob data_blob = new FloatBlob();
      data_blob.reshape(batch_size, channels, height, width);
      dataBlobs[0] = data_blob;
    
      FloatBlob labelblob = new FloatBlob();
      labelblob.reshape(batch_size, 1, 1, 1);
    
      //transformer
      LayerParameter train_layer_param = net_param.getLayer(0);
      TransformationParameter param = train_layer_param.getTransformParam();
      FloatDataTransformer xform = new FloatDataTransformer(param, true);
    
      nextBatch(matVec, labelblob);
      xform.transform(matVec, data_blob);
      boolean fail = false;
      String[] test_features = {"loss"};
      try {
	  FloatBlob[] top_blobs_vec = net.predict(0, null, test_features);
      } catch(Exception e) {
	  fail = true;
      }
      assertTrue(fail);
      xform.deallocate();
      data_blob.deallocate();
      matVec.deallocate();
    }

    @Test
    public void testTrain() throws Exception {
        SolverParameter solver_param = Utils.GetSolverParam(rootPath + "caffe-distri/src/test/resources/caffenet_solver.prototxt");

        String net_proto_file = solver_param.getNet();
        NetParameter net_param = Utils.GetNetParam(rootPath + "caffe-distri/" + net_proto_file);

        //blob
        MatVector matVec = new MatVector(batch_size);
        FloatBlob[] dataBlobs = new FloatBlob[2];
        FloatBlob data_blob = new FloatBlob();
        data_blob.reshape(batch_size, channels, height, width);
        dataBlobs[0] = data_blob;

        FloatBlob labelblob = new FloatBlob();
        labelblob.reshape(batch_size, 1, 1, 1);
	dataBlobs[1] = labelblob;

        //transformer
        LayerParameter train_layer_param = net_param.getLayer(0);
        TransformationParameter param = train_layer_param.getTransformParam();
        FloatDataTransformer xform = new FloatDataTransformer(param, true);

        //simplified training
        System.out.print("CaffeNetTest training:");
        for (int i=0; i<batchs; i++) {
            System.out.print(".");
            nextBatch(matVec, labelblob);
	    xform.transform(matVec, dataBlobs[0]);
	    assertTrue(net.train(0, dataBlobs));
        }

        //simplified test
        String[] test_features = {"loss"};
        System.out.print("CaffeNetTest test:");
        for (int i=0; i<batchs; i++) {
            System.out.print(".");
	    nextBatch(matVec, labelblob);
            xform.transform(matVec, dataBlobs[0]);
            FloatBlob[] top_blobs_vec = net.predict(0, dataBlobs, test_features);
            //validate test results
            for (int j = 0; j< top_blobs_vec.length; j++) {
	      FloatArray result_vec = top_blobs_vec[j].cpu_data();
	      assertTrue(result_vec.get(0) < 50.0);
            }
        }

        //release C++ resource
        xform.deallocate();
        data_blob.deallocate();
	matVec.deallocate();
    }
}
