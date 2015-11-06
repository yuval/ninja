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

package com.basistech.ninja.ejml;

/**
 * {@code ColVector} is a vector with a single column. It's implemented as a
 * {@code NinjaMatrix} with a single column.
 */
public class ColVector {
    private final NinjaMatrix data;

    public ColVector(int rows) {
        this(new double[rows]);
    }

    public ColVector(double... values) {
        data = new NinjaMatrix(values.length, 1, false, values);
    }

    public ColVector(NinjaMatrix matrix) {
        if (matrix.numCols() > 1) {
            throw new IllegalArgumentException("Matrix must be a column vector! (i.e., single column)");
        }
        this.data = matrix;
    }

    public int numRows() {
        return data.numRows();
    }

    public int numCols() {
        return data.numCols();
    }

    public double get(int row) {
        return data.get(row, 0);
    }

    public void set(int row, double value) {
        data.set(row, 0, value);
    }

    public NinjaMatrix transpose() {
        // TODO: introduce RowVector
        NinjaMatrix result = new NinjaMatrix(data.getMatrix());  //TODO: name?  getMatrix?
        result.transpose();
        return result;
    }

    public double[] getData() {
        return data.getData();
    }

    public void minus(ColVector other) {
        data.minus(other.data);
    }

    public void elementMult(ColVector other) {
        data.elementMult(other.data);
    }

    public NinjaMatrix mult(NinjaMatrix matrix) {
        return this.data.mult(matrix);
    }

    public static ColVector mult(NinjaMatrix matrix, ColVector vec) {
        return new ColVector(matrix.mult(vec.data));
    }

    public ColVector copy() {
        return new ColVector(this.data.copy());
    }

    @Override
    public String toString() {
        return data.toString();
    }
}
