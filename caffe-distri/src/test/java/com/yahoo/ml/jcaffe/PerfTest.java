// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import org.testng.annotations.Test;
import static org.testng.Assert.*;
import java.util.Random;
import java.util.Arrays;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class PerfTest {
    String rootPath, imagePath;

    List<String> file_list;

    @BeforeMethod
    public void setUp() throws Exception {
        String fullPath = getClass().getClassLoader().getResource("log4j.properties").getPath();
        rootPath = fullPath.substring(0, fullPath.indexOf("caffe-distri/"));
        imagePath = rootPath + "data/images";

        file_list = Files.readAllLines(Paths.get(imagePath + "/labels.txt"), StandardCharsets.UTF_8);
    }

    //LEAKTEST: uncomment to test leak for set_cpu_data()
    //    @Test
    public void testLeak() {
	FloatBlob blob = new FloatBlob();
        boolean res = blob.reshape(1, 1, 2, 2);
	float[] input = { 1.0f, 2.0f, 3.0f, 4.0f };
        blob.set_cpu_data(input);    
        for(int j=0; j < 1000000; j++){
            System.out.print("+");
	    blob.cpu_data();
	}
	blob.deallocate();
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
    //Perf test for Mat and MatVector
    //@Test
    private void perfMatVecTest() throws Exception {
        int iter = 100000;
        int arraysize = 1000;
        MatVector matVec = new MatVector(arraysize);
        for (int i=0; i < iter; i++) {
            System.out.print("+");
	    byte[] data0 = getDataFromFile(0);
            for (int j=0; j < arraysize; j++){
		Mat mat = new Mat(data0);
                Mat oldmat = matVec.put(j, mat);
                if(oldmat != null)
                    oldmat.deallocate();
                
                if ( j == 50)
                    data0 = getDataFromFile(1);
            }

	    //What we wrote is what we get
            data0=getDataFromFile(0);
            for(int j=0; j < arraysize; j++){
                assertTrue(Arrays.equals(data0,matVec.data(j)));
                if(j == 50)
                    data0 = getDataFromFile(1);
            }
	}
	matVec.deallocate();
        System.out.println();
        System.out.println("DONE");
    }

    //@Test
    private void perfMatTest() throws Exception {
	int iter = 100000;
	int arraysize = 1000;
	MatVector matVec = new MatVector(arraysize);
	byte[] data0 = getDataFromFile(0);
	for (int i=0; i < iter; i++) {
	    System.out.print("+");
	    for (int j=0; j < arraysize; j++){
		Mat mat = new Mat(data0);
		mat.decode(Mat.CV_LOAD_IMAGE_COLOR);
		mat.resize(227,227);
		Mat oldmat = matVec.put(j, mat);
		if(oldmat != null)
		  oldmat.deallocate();
	    }
	}
    }
}
