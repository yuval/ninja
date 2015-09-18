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

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

public class NinjaMatrix {
    private DenseMatrix64F data;
    
    public NinjaMatrix(int rows, int cols) {
        data = new DenseMatrix64F(rows, cols);
    }

    public NinjaMatrix(DenseMatrix64F matrix) {
        this.data = matrix;
    }

    public DenseMatrix64F getMatrix() {
        return data;
    }

    public int numRows() {
        return data.numRows;
    }

    public int numCols() {
        return data.numCols;
    }

    public double get(int row) {
        return data.get(row);
    }

    public void set(int row, double value) {
        data.set(row, value);
    }

    public void transpose() {
        CommonOps.transpose(this.data);
    }

    public double[] getData() {
        return data.getData();
    }

    public void plus(NinjaMatrix other) {
        CommonOps.addEquals(this.data, other.data);
    }

    public void minus(NinjaMatrix other) {
        CommonOps.subtractEquals(this.data, other.data);
    }

    public void elementMult(NinjaMatrix other) {
        CommonOps.elementMult(this.data, other.data);
    }

    public NinjaMatrix mult(NinjaMatrix other) {
        DenseMatrix64F result = new DenseMatrix64F(data.numRows, other.numCols());
        CommonOps.mult(this.data, other.data, result);
        return new NinjaMatrix(result);
    }

    @Override
    public String toString() {
        return data.toString();
    }
}
