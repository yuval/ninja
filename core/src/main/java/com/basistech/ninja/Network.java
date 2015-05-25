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

public class Network {
    private final SimpleMatrix[] w;
    private final Function activationFunction;

    public Network(SimpleMatrix ... w) {
        this(Functions.SIGMOID, w);
    }

    public Network(Function activationFunction, SimpleMatrix ... w) {
        this.activationFunction = activationFunction;
        this.w = w;
    }

    private SimpleMatrix addBiasUnit(SimpleMatrix m) {
        SimpleMatrix result = new SimpleMatrix(m.numRows() + 1, 1);
        result.set(0, 0, 1.0);
        for (int i = 0; i < m.numRows(); i++) {
            result.set(i + 1, 0, m.get(i, 0));
        }
        return result;
    }

    SimpleMatrix apply(double ... values) {
        int layers = w.length + 1;
        SimpleMatrix[] a = new SimpleMatrix[layers];
        a[0] = addBiasUnit(new SimpleMatrix(values.length, 1, true, values));
        for (int l = 1; l < layers; l++) {
            a[l] = Functions.apply(activationFunction, w[l - 1].mult(a[l - 1]));
            if (l != layers - 1) {
                a[l] = addBiasUnit(a[l]);
            }
        }
        return a[layers - 1];
    }
}
