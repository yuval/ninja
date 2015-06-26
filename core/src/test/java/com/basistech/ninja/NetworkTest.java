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

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import org.ejml.simple.SimpleMatrix;
import org.junit.Test;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class NetworkTest {
    private boolean isZero(SimpleMatrix m) {
        if (m.numRows() != 1 && m.numCols() != 1) {
            throw new RuntimeException(String.format(
                    "only 1x1 matrix supported; got %dx%d", m.numRows(), m.numCols()));
        }
        double value = m.get(0, 0);
        return value > 0.0 && value < 0.1;
    }

    private boolean isOne(SimpleMatrix m) {
        if (m.numRows() != 1 && m.numCols() != 1) {
            throw new RuntimeException(String.format(
                    "only 1x1 matrix supported; got %dx%d", m.numRows(), m.numCols()));
        }
        double value = m.get(0, 0);
        return value > 0.9 && value < 1.0;
    }

    @Test
    public void testAnd() {
        SimpleMatrix w1 = new SimpleMatrix(1, 3, true, -15, 10, 10);
        Network net = new Network(w1);
        assertTrue(isZero(net.apply(0, 0)));
        assertTrue(isZero(net.apply(0, 1)));
        assertTrue(isZero(net.apply(1, 0)));
        assertTrue(isOne(net.apply(1, 1)));
    }

    @Test
    public void testOr() {
        SimpleMatrix w1 = new SimpleMatrix(1, 3, true, -15, 20, 20);
        Network net = new Network(w1);
        assertTrue(isZero(net.apply(0, 0)));
        assertTrue(isOne(net.apply(0, 1)));
        assertTrue(isOne(net.apply(1, 0)));
        assertTrue(isOne(net.apply(1, 1)));
    }

    @Test
    public void testNot() {
        SimpleMatrix w1 = new SimpleMatrix(1, 2, true, 5, -10);
        Network net = new Network(w1);
        assertTrue(isOne(net.apply(0)));
        assertTrue(isZero(net.apply(1)));
    }

    @Test
    public void testNand() {
        SimpleMatrix w1 = new SimpleMatrix(1, 3, true, 15, -10, -10);
        Network net = new Network(w1);
        assertTrue(isOne(net.apply(0, 0)));
        assertTrue(isOne(net.apply(0, 1)));
        assertTrue(isOne(net.apply(1, 0)));
        assertTrue(isZero(net.apply(1, 1)));
    }

    @Test
    public void testThreeLayerNand() {
        SimpleMatrix w1 = new SimpleMatrix(1, 3, true, -15, 10, 10);
        SimpleMatrix w2 = new SimpleMatrix(1, 2, true, 5, -10);
        Network net = new Network(w1, w2);
        assertTrue(isOne(net.apply(0, 0)));
        assertTrue(isOne(net.apply(0, 1)));
        assertTrue(isOne(net.apply(1, 0)));
        assertTrue(isZero(net.apply(1, 1)));

        System.out.println(net.apply(0, 0).get(0, 0));
        System.out.println(net.apply(0, 1).get(0, 0));
        System.out.println(net.apply(1, 0).get(0, 0));
        System.out.println(net.apply(1, 1).get(0, 0));
    }

    @Test
    public void testThreeLayerNandDeltas() {
        SimpleMatrix w1 = new SimpleMatrix(1, 3, true, -15, 10, 10);
        SimpleMatrix w2 = new SimpleMatrix(1, 2, true, 5, -10);
        #Network net = new Network(w1, w2);

        SimpleMatrix x = new SimpleMatrix(2, 1, true, 0, 0);
        SimpleMatrix y = new SimpleMatrix(1, 1, true, 1);

        // feedforward to compute z and a for each layer
        SimpleMatrix a1 = Network.addBiasUnit(x);
        SimpleMatrix z2 = w1.mult(a1);
        SimpleMatrix a2 = Network.addBiasUnit(Functions.apply(Functions.SIGMOID, z2));
        SimpleMatrix z3 = w2.mult(a2);
        SimpleMatrix a3 = Functions.apply(Functions.SIGMOID, z3);

        // backprop to compute d for each layer (one d for every a)
        SimpleMatrix d3 = a3.minus(y);
        SimpleMatrix d2 = w2.transpose().mult(d3).elementMult(Functions.apply(
                Functions.SIGMOID_PRIME, Network.addBiasUnit(z2)));
        d2 = Network.stripBiasUnit(d2);

        assertEquals(-0.007, d3.get(0, 0), 0.001);
        assertEquals(0.0, d2.get(0, 0), 0.001);

        assertEquals(1, d3.numRows());
        assertEquals(1, d3.numCols());
        assertEquals(1, d2.numRows());
        assertEquals(1, d2.numCols());


        // TODO: do for more examples?
    }

    // TODO: start with w2 (like Neilsen) or w1 like (Ng)?
    @Test
    public void testFullthreeLayerDeltas() {
        SimpleMatrix w1 = new SimpleMatrix(4, 4, true,
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        );
        SimpleMatrix w2 = new SimpleMatrix(2, 5, true,
                1, 2, 3, 4, 5,
                -6, -7, -8, -9, -10
        );
        //Network net = new Network(w1, w2);

        SimpleMatrix x = new SimpleMatrix(3, 1, true, 0, 0, 0);
        SimpleMatrix y = new SimpleMatrix(2, 1, true, 0, 0);

        // feedforward to compute z and a for each layer
        SimpleMatrix a1 = Network.addBiasUnit(x);
        SimpleMatrix z2 = w1.mult(a1);
        SimpleMatrix a2 = Network.addBiasUnit(Functions.apply(Functions.SIGMOID, z2));
        SimpleMatrix z3 = w2.mult(a2);
        SimpleMatrix a3 = Functions.apply(Functions.SIGMOID, z3);

        // backprop to compute d for each layer (one d for every a)
        SimpleMatrix d3 = a3.minus(y);
        SimpleMatrix d2 = w2.transpose().mult(d3).elementMult(Functions.apply(
                Functions.SIGMOID_PRIME, Network.addBiasUnit(z2)));
        d2 = Network.stripBiasUnit(d2);

        assertEquals(1.0, d3.get(0, 0), 0.001);
        assertEquals(0.0, d3.get(1, 0), 0.001);
        assertEquals(0.393, d2.get(0, 0), 0.001);
        assertEquals(0.02, d2.get(1, 0), 0.001);
        assertEquals(0.0, d2.get(2, 0), 0.001);
        assertEquals(0.0, d2.get(3, 0), 0.001);

        assertEquals(2, d3.numRows());
        assertEquals(1, d3.numCols());
        assertEquals(4, d2.numRows());
        assertEquals(1, d2.numCols());
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

        SimpleMatrix w2 = new SimpleMatrix(4, 4, true,
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        );
        SimpleMatrix w3 = new SimpleMatrix(2, 5, true,
                1, 2, 3, 4, 5,
                -6, -7, -8, -9, -10
        );

        Network net;
        SimpleMatrix output;
        net = new Network(Functions.IDENTITY, w2, w3);
        output = net.apply(1, 1, 1);
        assertEquals(2, output.numRows());
        assertEquals(1, output.numCols());
        assertEquals(557.0, output.get(0, 0), 0.00001);
        assertEquals(-1242.0, output.get(1, 0), 0.00001);

        net = new Network(Functions.SIGMOID, w2, w3);
        output = net.apply(1, 1, 1);
        assertEquals(2, output.numRows());
        assertEquals(1, output.numCols());
        assertEquals(1.0, output.get(0, 0), 0.00001);
        assertEquals(0.0, output.get(1, 0), 0.00001);
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
        Network net = Network.fromText(reader);

        assertEquals(3, net.getNumLayers());
        assertEquals(3, net.getNumNeurons(0));
        assertEquals(4, net.getNumNeurons(1));
        assertEquals(2, net.getNumNeurons(2));

        SimpleMatrix w1 = new SimpleMatrix(4, 4, true,
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        );
        SimpleMatrix w2 = new SimpleMatrix(2, 5, true,
                1, 2, 3, 4, 5,
                -6, -7, -8, -9, -10
        );

        assertTrue(w1.isIdentical(net.getWeightMatrix(0), 0.00001));
        assertTrue(w2.isIdentical(net.getWeightMatrix(1), 0.00001));
    }
}
