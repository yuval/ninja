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
import org.ejml.ops.MatrixFeatures;
import org.ejml.ops.SpecializedOps;

public class NinjaMatrix {
    private DenseMatrix64F data;
    
    public NinjaMatrix(int rows, int cols) {
        data = new DenseMatrix64F(rows, cols);
    }

    public NinjaMatrix(DenseMatrix64F matrix) {
        this.data = matrix;
    }

    public NinjaMatrix(int numRows, int numCols, boolean rowMajor, double... data) {
        this.data = new DenseMatrix64F(numRows, numCols, rowMajor, data);
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

    public double get(int row, int col) {
        return data.get(row, col);
    }

    public void set(int row, int col, double value) {
        data.set(row, col, value);
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

    public void divide(double val) {
        CommonOps.divide(this.data, val);
    }

    public void scale(double alpha) {
        CommonOps.scale(alpha, this.data);
    }

    /**
     * See {@link org.ejml.simple.SimpleBase#extractVector(boolean extractRow, int element)}
     */
    public NinjaMatrix extractVector(boolean extractRow, int element) {
        int length = extractRow ? numCols() : numRows();
        NinjaMatrix result = extractRow ? new NinjaMatrix(1, length) : new NinjaMatrix(length, 1);
        if (extractRow) {
            SpecializedOps.subvector(this.data, element, 0, length, true, 0, result.data);
        } else {
            SpecializedOps.subvector(this.data, 0, element, length, false, 0, result.data);
        }
        return result;
    }

    public NinjaMatrix copy() {
        NinjaMatrix result = new NinjaMatrix(numRows(), numCols());
        result.data.set(this.data);
        return result;
    }

    /**
     * See {@link org.ejml.simple.SimpleBase#isIdentical(org.ejml.simple.SimpleBase, double)}
     */
    public boolean isIdentical(NinjaMatrix other, double tol) {
        return MatrixFeatures.isIdentical(this.data, other.data, tol);
    }

    @Override
    public String toString() {
        return data.toString();
    }
}
