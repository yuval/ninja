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

import java.util.List;

public class Network {

    private final SimpleMatrix inputWeights;
    private final List<SimpleMatrix> hiddenWeights;

    public Network(SimpleMatrix inputWeights, List<SimpleMatrix> hiddenWeights) {
        this.inputWeights = inputWeights;
        this.hiddenWeights = hiddenWeights;
    }

    public SimpleMatrix feedForward(SimpleMatrix input) {
        SimpleMatrix result = sigmoid(inputWeights.mult(input));
        for (SimpleMatrix hidden : hiddenWeights) {
            result = sigmoid(hidden.mult(result));
        }
        return result;
    }

    private SimpleMatrix sigmoid(SimpleMatrix matrix) {
        SimpleMatrix result = new SimpleMatrix(matrix.numRows(), matrix.numCols());
        for(int i = 0; i < matrix.numRows(); i++){
            for (int j = 0; j < matrix.numCols(); j++) {
                result.set(i, j, sigmoid(matrix.get(i, j)));
            }
        }
        return result;
    }

    private double sigmoid(double val) {
        return 1.0 / (1.0 + Math.exp(-1.0 * val));
    }


}
