// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;
import static org.testng.Assert.*;

import java.io.FileReader;
import java.io.StringReader;

import caffe.Caffe.*;
import com.google.protobuf.TextFormat;

public class ProtoTest {

    String rootPath, solver_config_path;

    @BeforeMethod
    public void setUp() throws Exception {
        String fullPath = getClass().getClassLoader().getResource("log4j.properties").getPath();
        rootPath = fullPath.substring(0, fullPath.indexOf("caffe-distri/"))+"caffe-distri/";
        solver_config_path = rootPath + "src/test/resources/caffenet_solver.prototxt";
    }

    @Test
    public void testParse() throws Exception {
        SolverParameter solver_param = Utils.GetSolverParam(solver_config_path);
        String net_proto_file = solver_param.getNet();
        assertEquals(net_proto_file, "src/test/resources/caffenet_train_net.prototxt");

        NetParameter net_param = Utils.GetNetParam(rootPath + net_proto_file);
        LayerParameter layer_param = net_param.getLayer(0);
        String layer_name = layer_param.getName();
        assertEquals(layer_name, "data");
        assertTrue(layer_param.hasTransformParam());

        TransformationParameter param1 = layer_param.getTransformParam();
        assertTrue(param1.getMirror());
        assertEquals(param1.getCropSize(), 227);
        assertEquals(param1.getMeanValueCount(), 3);
        assertEquals(param1.getMeanValue(0), 104, 0.1);
        assertEquals(param1.getMeanValue(1), 117, 0.1);
        assertEquals(param1.getMeanValue(2), 123, 0.1);
        String param_str = param1.toString();

        TransformationParameter.Builder param2_builder = TransformationParameter.newBuilder();
        StringReader reader = new StringReader(param_str);
        TextFormat.merge(reader, param2_builder);
        reader.close();
        TransformationParameter  param2 =  param2_builder.build();
        assertEquals(param1.getCropSize(), param2.getCropSize());
        assertEquals(param1.getMeanValueCount(), param2.getMeanValueCount());
        assertEquals(param1.getMeanValue(0), param2.getMeanValue(0));
        assertEquals(param1.getMeanValue(1), param2.getMeanValue(1));
        assertEquals(param1.getMeanValue(2), param2.getMeanValue(2));
    }
}
