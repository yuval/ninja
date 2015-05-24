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

public class FeedforwardNet {
    private final Function f;
    private final int inputCount;
    private final int outputCount;
    private final Matrix[] w;

    FeedforwardNet(int inputCount, int outputCount, Matrix ... w) {
        this.f = Functions.SIGMOID;
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.w = w;
    }

    FeedforwardNet(Function f, int inputCount, int outputCount, Matrix ... w) {
        this.f = f;
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.w = w;
    }

    Matrix apply(double ... values) {
        int layers = w.length + 1;
        Matrix[] a = new Matrix[layers];
        a[0] = new Matrix(values.length, 1, values);
        //return Matrix.multiply(w[0], a[0]).apply(f);
        for (int l = 1; l < layers; l++) {
            a[l] = Matrix.multiply(w[l - 1], a[l - 1]).apply(f);
            if (l != layers - 1) {
                a[l] = a[l].insertBias();
            }
        }
        return a[layers - 1];
    }
}
