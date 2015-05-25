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

import org.ejml.simple.SimpleMatrix;
import org.junit.Test;

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
        assertTrue(isZero(net.apply(1, 0, 0)));
        assertTrue(isZero(net.apply(1, 0, 1)));
        assertTrue(isZero(net.apply(1, 1, 0)));
        assertTrue(isOne(net.apply(1, 1, 1)));
    }

    @Test
    public void testOr() {
        SimpleMatrix w1 = new SimpleMatrix(1, 3, true, -15, 20, 20);
        Network net = new Network(w1);
        assertTrue(isZero(net.apply(1, 0, 0)));
        assertTrue(isOne(net.apply(1, 0, 1)));
        assertTrue(isOne(net.apply(1, 1, 0)));
        assertTrue(isOne(net.apply(1, 1, 1)));
    }

    @Test
    public void testNot() {
        SimpleMatrix w1 = new SimpleMatrix(1, 2, true, 5, -10);
        Network net = new Network(w1);
        assertTrue(isOne(net.apply(1, 0)));
        assertTrue(isZero(net.apply(1, 1)));
    }

    @Test
    public void testNand() {
        SimpleMatrix w1 = new SimpleMatrix(1, 3, true, 15, -10, -10);
        Network net = new Network(w1);
        assertTrue(isOne(net.apply(1, 0, 0)));
        assertTrue(isOne(net.apply(1, 0, 1)));
        assertTrue(isOne(net.apply(1, 1, 0)));
        assertTrue(isZero(net.apply(1, 1, 1)));
    }

    @Test
    public void testThreeLayerNand() {
        SimpleMatrix w1 = new SimpleMatrix(1, 3, true, -15, 10, 10);
        SimpleMatrix w2 = new SimpleMatrix(1, 2, true, 5, -10);
        Network net = new Network(w1, w2);
        assertTrue(isOne(net.apply(1, 0, 0)));
        assertTrue(isOne(net.apply(1, 0, 1)));
        assertTrue(isOne(net.apply(1, 1, 0)));
        assertTrue(isZero(net.apply(1, 1, 1)));
    }

    @Test
    public void testFullThreeLayer() {
        // >>> z2 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        // >>> z2
        // array([[ 1,  2,  3,  4],
        //        [ 5,  6,  7,  8],
        //        [ 9, 10, 11, 12],
        //        [13, 14, 15, 16]])
        //
        // >>> z3 = np.array([[1,2,3,4,5],[-6,-7,-8,-9,-10]])
        // >>> z3
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
        // >>> a2 = np.vstack([1, np.dot(z2, x)])
        // >>> a2
        // array([[ 1],
        //        [10],
        //        [26],
        //        [42],
        //        [58]])
        //
        // >>> a3 = np.dot(z3, a2)
        // >>> a3
        // array([[  557],
        //        [-1242]])

        SimpleMatrix z2 = new SimpleMatrix(4, 4, true,
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        );
        SimpleMatrix z3 = new SimpleMatrix(2, 5, true,
                1, 2, 3, 4, 5,
                -6, -7, -8, -9, -10
        );

        Network net;
        SimpleMatrix output;
        net = new Network(Functions.IDENTITY, z2, z3);
        output = net.apply(1, 1, 1, 1);
        assertEquals(2, output.numRows());
        assertEquals(1, output.numCols());
        assertEquals(557.0, output.get(0, 0), 0.00001);
        assertEquals(-1242.0, output.get(1, 0), 0.00001);

        net = new Network(Functions.SIGMOID, z2, z3);
        output = net.apply(1, 1, 1, 1);
        assertEquals(2, output.numRows());
        assertEquals(1, output.numCols());
        assertEquals(1.0, output.get(0, 0), 0.00001);
        assertEquals(0.0, output.get(1, 0), 0.00001);
    }
}
