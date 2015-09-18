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

import com.basistech.ninja.NinjaMatrix;
import org.ejml.simple.SimpleMatrix;

public class ColVector {
    private final NinjaMatrix data;

    public ColVector(int rows) {
        data = new NinjaMatrix(rows, 1);
    }

    public ColVector(double... values) {
        data = new NinjaMatrix(values.length, 1, true, values);
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
        return data.get(row);
    }

    public void set(int row, double value) {
        data.set(row, value);
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

    public ColVector minus(ColVector other) {
        return new ColVector(data.minus(other.data));
    }

    public ColVector elementMult(ColVector other) {
        return new ColVector(data.elementMult(other.data));
    }

    public SimpleMatrix mult(SimpleMatrix matrix) {
        return this.data.mult(matrix);
    }

    public static ColVector mult(SimpleMatrix matrix, ColVector vec) {
        return new ColVector(matrix.mult(vec.data));
    }

    @Override
    public String toString() {
        return data.toString();
    }
}
