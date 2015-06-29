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

public final class Functions {
    static final Function IDENTITY = new Identity();
    static final Function SIGMOID = new Sigmoid();
    static final Function SIGMOID_PRIME = new SigmoidPrime();

    private Functions() {
        // empty
    }

    private static class Identity implements Function {
        public double apply(double x) {
            return x;
        }
    }

    private static class Sigmoid implements Function {
        public double apply(double x) {
            return 1.0 / (1 + Math.exp(-x));
        }
    }

    private static class SigmoidPrime implements Function {
        public double apply(double x) {
            double sigmoid = SIGMOID.apply(x);
            return sigmoid * (1.0 - sigmoid);
        }
    }

    public static SimpleMatrix apply(Function f, SimpleMatrix m) {
        SimpleMatrix result = new SimpleMatrix(m.numRows(), m.numCols());
        for (int i = 0; i < m.numRows(); i++) {
            for (int j = 0; j < m.numCols(); j++) {
                result.set(i, j, f.apply(m.get(i, j)));
            }
        }
        return result;
    }
}
