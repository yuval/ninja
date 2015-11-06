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

/**
 * {@code Functions} is a collection of useful functions for neural networks.
 */
public final class Functions {
    /**
     * sigmoid(x) = 1.0 / (1 + Math.exp(-x))
     */
    public static final Function SIGMOID = new Sigmoid();
    /**
     * The derivative of the sigmoid function
     */
    public static final Function SIGMOID_PRIME = new SigmoidPrime();

    private Functions() {
        // empty
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

    /**
     * Applies the function on every element of the column vector.
     *
     * @param f the function to apply
     * @param vec the input vector to apply the function on
     * @return the result of applying the function on the input vector
     */
    public static ColVector apply(Function f, ColVector vec) {
        ColVector result = new ColVector(vec.numRows());
        for (int i = 0; i < vec.numRows(); i++) {
            result.set(i, f.apply(vec.get(i)));
        }
        return result;
    }
}
