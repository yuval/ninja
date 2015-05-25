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

import java.util.Arrays;

public class Matrix {
    private final int rows;
    private final int columns;
    private final double[] data;

    Matrix(int rows, int columns, double ... values) {
        if (rows * columns != values.length) {
            throw new IllegalArgumentException(String.format(
                    "invalid matrix dimensions: %d rows, %d columns, %d values",
                    rows, columns, values.length));
        }
        this.rows = rows;
        this.columns = columns;
        data = Arrays.copyOf(values, values.length);
    }

    double get(int i, int j) {
        if (i < 0 || i >= rows || j < 0 || j >= columns) {
            throw new IllegalArgumentException(String.format(
                    "Cannot access %d, %d in %s", i, j, getDimensions()));
        }
        return data[i * columns + j];
    }

    int getRows() {
        return rows;
    }

    int getColumns() {
        return columns;
    }

    String getDimensions() {
        return rows + "x" + columns;
    }

    Matrix insertBias() {
        if (columns != 1) {
            throw new RuntimeException("only column vectors supported, got: " + getDimensions());
        }
        double[] d = new double[data.length + 1];
        System.arraycopy(data, 0, d, 1, data.length);
        d[0] = 1.0;
        return new Matrix(rows + 1, columns, d);
    }


    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                sb.append(get(i, j));
                sb.append("  ");
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    Matrix transpose() {
        double[] d = new double[rows * columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                d[j * rows + i] = get(i, j);
            }
        }
        return new Matrix(columns, rows, d);
    }

    static Matrix multiply(Matrix a, Matrix b) {
        if (a.getColumns() != b.getRows()) {
            throw new RuntimeException(String.format(
                    "cannot multiply %s by %s",
                    a.getDimensions(), b.getDimensions()));
        }
        double[] d = new double[a.rows * b.columns];
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < b.columns; j++) {
                double sum = 0.0;
                for (int k = 0; k < a.columns; k++) {
                    sum += a.get(i, k) * b.get(k, j);
                }
                d[i * b.columns + j] = sum;
            }
        }
        return new Matrix(a.rows, b.columns, d);
    }

    static Matrix add(Matrix a, Matrix b) {
        if (a.rows != b.rows || a.columns != b.columns) {
            throw new RuntimeException(String.format(
                    "cannot add %s and %s",
                    a.getDimensions(), b.getDimensions()));
        }
        double[] d = new double[a.rows * a.columns];
        for (int i = 0; i < d.length; i++) {
            d[i] = a.data[i] + b.data[i];
        }
        return new Matrix(a.rows, a.columns, d);
    }

    Matrix apply(Function f) {
        double[] d = new double[rows * columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                d[i * columns + j] = f.apply(get(i, j));
            }
        }
        return new Matrix(rows, columns, d);
    }
}
