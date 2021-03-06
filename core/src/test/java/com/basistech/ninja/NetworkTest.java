/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.basistech.ninja;

import com.basistech.ninja.ejml.ColVector;
import com.basistech.ninja.ejml.NinjaMatrix;
import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class NetworkTest {
    private boolean isZero(ColVector v) {
        if (v.numRows() != 1) {
            throw new RuntimeException(String.format(
                    "only 1x1 matrix supported; got %dx%d", v.numRows(), v.numCols()));
        }
        double value = v.get(0);
        return value > 0.0 && value < 0.1;
    }

    private boolean isOne(ColVector v) {
        if (v.numRows() != 1) {
            throw new RuntimeException(String.format(
                    "only 1x1 matrix supported; got %dx%d", v.numRows(), v.numCols()));
        }
        double value = v.get(0);
        return value > 0.9 && value < 1.0;
    }

    @Test
    public void testAnd() {
        NinjaMatrix w1 = new NinjaMatrix(1, 3, true, -15, 10, 10);
        Network net = new Network(w1);
        assertTrue(isZero(net.apply(0, 0)));
        assertTrue(isZero(net.apply(0, 1)));
        assertTrue(isZero(net.apply(1, 0)));
        assertTrue(isOne(net.apply(1, 1)));
    }

    @Test
    public void testOr() {
        NinjaMatrix w1 = new NinjaMatrix(1, 3, true, -15, 20, 20);
        Network net = new Network(w1);
        assertTrue(isZero(net.apply(0, 0)));
        assertTrue(isOne(net.apply(0, 1)));
        assertTrue(isOne(net.apply(1, 0)));
        assertTrue(isOne(net.apply(1, 1)));
    }

    @Test
    public void testNot() {
        NinjaMatrix w1 = new NinjaMatrix(1, 2, true, 5, -10);
        Network net = new Network(w1);
        assertTrue(isOne(net.apply(0)));
        assertTrue(isZero(net.apply(1)));
    }

    @Test
    public void testNand() {
        NinjaMatrix w1 = new NinjaMatrix(1, 3, true, 15, -10, -10);
        Network net = new Network(w1);
        assertTrue(isOne(net.apply(0, 0)));
        assertTrue(isOne(net.apply(0, 1)));
        assertTrue(isOne(net.apply(1, 0)));
        assertTrue(isZero(net.apply(1, 1)));
    }

    @Test
    public void testThreeLayerNand() {
        NinjaMatrix w1 = new NinjaMatrix(1, 3, true, -15, 10, 10);
        NinjaMatrix w2 = new NinjaMatrix(1, 2, true, 5, -10);
        Network net = new Network(w1, w2);
        assertTrue(isOne(net.apply(0, 0)));
        assertTrue(isOne(net.apply(0, 1)));
        assertTrue(isOne(net.apply(1, 0)));
        assertTrue(isZero(net.apply(1, 1)));
    }

    @Test
    public void testThreeLayerNandDeltas() {
        NinjaMatrix w1 = new NinjaMatrix(1, 3, true, -15, 10, 10);
        NinjaMatrix w2 = new NinjaMatrix(1, 2, true, 5, -10);

        ColVector x = new ColVector(0.0, 0.0);
        ColVector y = new ColVector(1.0);

        Network net = new Network(w1, w2);
        ColVector[] d = net.backprop(net.feedForward(x), y);

        ColVector d3 = d[2];
        assertEquals(1, d3.numRows());
        assertEquals(1, d3.numCols());
        assertEquals(-0.007, d3.get(0) , 0.001);

        ColVector d2 = d[1];
        assertEquals(1, d2.numRows());
        assertEquals(1, d2.numCols());
        assertEquals(0.0, d2.get(0), 0.001);
    }

    @Test
    public void testFullthreeLayerDeltas() {
        NinjaMatrix w1 = new NinjaMatrix(4, 4, true,
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        );
        NinjaMatrix w2 = new NinjaMatrix(2, 5, true,
                1, 2, 3, 4, 5,
                -6, -7, -8, -9, -10
        );

        ColVector x = new ColVector(0.0, 0.0, 0.0);
        ColVector y = new ColVector(0.0, 0.0);

        Network net = new Network(w1, w2);
        ColVector[] d = net.backprop(net.feedForward(x), y);

        ColVector d3 = d[2];
        assertEquals(2, d3.numRows());
        assertEquals(1, d3.numCols());
        assertEquals(1.0, d3.get(0), 0.001);
        assertEquals(0.0, d3.get(1), 0.001);

        ColVector d2 = d[1];
        assertEquals(4, d2.numRows());
        assertEquals(1, d2.numCols());
        assertEquals(0.393, d2.get(0), 0.001);
        assertEquals(0.02, d2.get(1), 0.001);
        assertEquals(0.0, d2.get(2), 0.001);
        assertEquals(0.0, d2.get(3), 0.001);
    }

    @Test
    public void testFullThreeLayer() {
        // >>> w2 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        // >>> w2
        // array([[ 1,  2,  3,  4],
        //        [ 5,  6,  7,  8],
        //        [ 9, 10, 11, 12],
        //        [13, 14, 15, 16]])
        //
        // >>> w3 = np.array([[1,2,3,4,5],[-6,-7,-8,-9,-10]])
        // >>> w3
        // array([[  1,   2,   3,   4,   5],
        //        [ -6,  -7,  -8,  -9, -10]])
        //
        // >>> x = np.array([[1],[1],[1],[1]])
        // >>> x
        // array([[1],
        //        [1],
        //        [1],
        //        [1]])
        //
        // >>> a2 = np.vstack([1, np.dot(w2, x)])
        // >>> a2
        // array([[ 1],
        //        [10],
        //        [26],
        //        [42],
        //        [58]])
        //
        // >>> a3 = np.dot(w3, a2)
        // >>> a3
        // array([[  557],
        //        [-1242]])

        NinjaMatrix w2 = new NinjaMatrix(4, 4, true,
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        );
        NinjaMatrix w3 = new NinjaMatrix(2, 5, true,
                1, 2, 3, 4, 5,
                -6, -7, -8, -9, -10
        );

        Network net = new Network(w2, w3);
        ColVector output = net.apply(1, 1, 1);
        assertEquals(2, output.numRows());
        assertEquals(1, output.numCols());
        assertEquals(1.0, output.get(0), 0.00001);
        assertEquals(0.0, output.get(1), 0.00001);
    }

    @Test
    public void testFromText() throws IOException {
        List<String> lines = Lists.newArrayList(
                "num_layers=3",
                "layer_sizes=3 4 2",
                "w",
                "",
                "1 2 3 4",
                "5 6 7 8",
                "9 10 11 12",
                "13 14 15 16",
                "",
                "1 2 3 4 5",
                "-6 -7 -8 -9 -10");
        Reader reader = new StringReader(Joiner.on('\n').join(lines));
        Network net = Network.loadModel(reader);

        assertEquals(3, net.getNumLayers());
        assertEquals(3, net.getNumUnits(0));
        assertEquals(4, net.getNumUnits(1));
        assertEquals(2, net.getNumUnits(2));

        NinjaMatrix w1 = new NinjaMatrix(4, 4, true,
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        );
        NinjaMatrix w2 = new NinjaMatrix(2, 5, true,
                1, 2, 3, 4, 5,
                -6, -7, -8, -9, -10
        );

        assertTrue(w1.isIdentical(net.getWeightMatrix(0), 0.00001));
        assertTrue(w2.isIdentical(net.getWeightMatrix(1), 0.00001));
    }

    @Test
    public void testWriteModel() throws Exception {
        NinjaMatrix w1 = new NinjaMatrix(4, 4, true,
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16
        );
        NinjaMatrix w2 = new NinjaMatrix(2, 5, true,
            1, 2, 3, 4, 5,
            -6, -7, -8, -9, -10
        );
        Network net = new Network(w1, w2);

        StringWriter sw = new StringWriter();
        BufferedWriter bw = new BufferedWriter(sw);
        net.writeModel(bw);
        bw.close();

        Reader reader = new StringReader(sw.toString());
        Network net2 = Network.loadModel(reader);

        assertTrue(w1.isIdentical(net2.getWeightMatrix(0), 0.00001));
        assertTrue(w2.isIdentical(net2.getWeightMatrix(1), 0.00001));
    }

    @Test
    public void testRandomInitialize() {
        NinjaMatrix w1 = new NinjaMatrix(25, 401);
        NinjaMatrix w2 = new NinjaMatrix(10, 26);
        Network net = new Network(w1, w2);
        net.randomInitialize();

        double epsilon = Math.sqrt(6) / Math.sqrt(400 + 25);
        NinjaMatrix m = net.getWeightMatrix(0);
        for (int i = 0; i < m.numRows(); i++) {
            for (int j = 0; j < m.numCols(); j++) {
                double weight = m.get(i, j);
                assertTrue(-epsilon <= weight && weight <= epsilon);
            }
        }

        epsilon = Math.sqrt(6) / Math.sqrt(25 + 10);
        m = net.getWeightMatrix(1);
        for (int i = 0; i < m.numRows(); i++) {
            for (int j = 0; j < m.numCols(); j++) {
                double weight = m.get(i, j);
                assertTrue(-epsilon <= weight && weight <= epsilon);
            }
        }
    }

    @Test
    public void testStochasticGD() {
        NinjaMatrix w1 = new NinjaMatrix(1, 3);
        NinjaMatrix w2 = new NinjaMatrix(1, 2);
        Network net = new Network(w1, w2);
        net.randomInitialize();

        // NAND:
        // 0 0 --> 1
        // 0 1 --> 1
        // 1 0 --> 1
        // 1 1 --> 0
        ColVector[] x = new ColVector[] {
            new ColVector(0.0, 0.0),
            new ColVector(0.0, 1.0),
            new ColVector(1.0, 0.0),
            new ColVector(1.0, 1.0)
        };
        ColVector[] y = new ColVector[] {
            new ColVector(1.0),
            new ColVector(1.0),
            new ColVector(1.0),
            new ColVector(0.0),
        };

        int maxEpochs = 100000;
        double learningRate = 0.01;
        for (int i = 0; i < maxEpochs; i++) {
            net.trainBatch(x, y, learningRate);
        }

        System.out.println(net.apply(0, 0));
        System.out.println(net.apply(0, 1));
        System.out.println(net.apply(1, 0));
        System.out.println(net.apply(1, 1));

        assertTrue(isOne(net.apply(0, 0)));
        assertTrue(isOne(net.apply(0, 1)));
        assertTrue(isOne(net.apply(1, 0)));
        assertTrue(isZero(net.apply(1, 1)));
    }
}
