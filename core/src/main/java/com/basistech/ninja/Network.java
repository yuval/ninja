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

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import org.ejml.simple.SimpleMatrix;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.List;
import java.util.Random;

public class Network {
    private static final Random RANDOM = new Random(8723643324L);
    private final SimpleMatrix[] w;
    private final Function activationFunction = Functions.SIGMOID;

    public Network(SimpleMatrix ... w) {
        this.w = w;
    }

    public Network(List<SimpleMatrix> w) {
        this.w = new SimpleMatrix[w.size()];
        w.toArray(this.w);
    }

    static SimpleMatrix stripBiasUnit(SimpleMatrix m) {
        SimpleMatrix result = new SimpleMatrix(m.numRows() - 1, 1);
        for (int i = 0; i < result.numRows(); i++) {
            result.set(i, 0, m.get(i + 1, 0));
        }
        return result;
    }

    static SimpleMatrix addBiasUnit(SimpleMatrix m) {
        SimpleMatrix result = new SimpleMatrix(m.numRows() + 1, 1);
        result.set(0, 0, 1.0);
        for (int i = 0; i < m.numRows(); i++) {
            result.set(i + 1, 0, m.get(i, 0));
        }
        return result;
    }

    private void randomInitializeLayer(int layer) {
        int inputSize = getNumNeurons(layer);
        int outputSize = getNumNeurons(layer + 1);
        double epsilon = Math.sqrt(6) / Math.sqrt(inputSize + outputSize);
        // TODO: why does Ng keep epsilon fixed for different layers?
        // TODO: Nielsen uses something different, and parameterizes the init function
        for (int i = 0; i < w[layer].numRows(); i++) {
            for (int j = 0; j < w[layer].numCols(); j++) {
                double value = RANDOM.nextDouble() * 2 * epsilon - epsilon;
                w[layer].set(i, j, value);
            }
        }
    }

    // TODO: think of a way to initialize during construction
    public void randomInitialize() {
        for (int l = 0; l < getNumLayers() - 1; l++) {
            randomInitializeLayer(l);
        }
    }

    public int getNumLayers() {
        return w.length + 1;
    }

    // 0-based; doesn't count bias unit
    public int getNumNeurons(int layer) {
        if (layer == 0) {
            // subtract the bias unit
            return w[layer].numCols() - 1;
        } else {
            return w[layer - 1].numRows();
        }
    }

    // 0-based
    public SimpleMatrix getWeightMatrix(int layer) {
        return w[layer].copy();
    }

    public SimpleMatrix[] feedForward(SimpleMatrix m) {
        return feedForward(m.getMatrix().getData());
    }

    public SimpleMatrix[] feedForward(double ... values) {
        int layers = w.length + 1;
        SimpleMatrix[] z = new SimpleMatrix[layers];
        SimpleMatrix[] a = new SimpleMatrix[layers];
        z[0] = Network.addBiasUnit(new SimpleMatrix(values.length, 1, true, values));
        a[0] = z[0];
        for (int l = 1; l < layers; l++) {
            z[l] = w[l - 1].mult(a[l - 1]);
            a[l] = Functions.apply(activationFunction, z[l]);
            if (l != layers - 1) {
                z[l] = Network.addBiasUnit(z[l]);
                a[l] = Network.addBiasUnit(a[l]);
            }
        }
        return z;
    }

    public SimpleMatrix apply(double ... values) {
        SimpleMatrix[] z =  feedForward(values);
        return Functions.apply(activationFunction, z[z.length - 1]);
    }

    // TODO: parameterizable cost function
    public SimpleMatrix[] backprop(SimpleMatrix x, SimpleMatrix y) {
        SimpleMatrix[] zVectors  = feedForward(x);
        SimpleMatrix[] deltas = new SimpleMatrix[getNumLayers() - 1];
        for (int l = deltas.length - 1; l >= 0; l--) {
            if (l == deltas.length - 1) {
                // delta "L"
                SimpleMatrix a = Functions.apply(activationFunction, zVectors[l + 1]);
                deltas[l] = a.minus(y);
            } else {
                deltas[l] = w[l + 1].transpose().mult(deltas[l + 1]).elementMult(
                        Functions.apply(Functions.SIGMOID_PRIME, zVectors[l + 1]));
                deltas[l] = Network.stripBiasUnit(deltas[l]);
            }
        }
        return deltas;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (SimpleMatrix m : w) {
            sb.append(m.toString());
        }
        return sb.toString();
    }

    public static Network fromText(File file) throws IOException {
        try (Reader reader = new InputStreamReader(new FileInputStream(file), Charsets.US_ASCII)) {
            return Network.fromText(reader);
        }
    }

    // TODO: better error messages, e.g. line number, invalid header fields, too many lines, etc.
    public static Network fromText(Reader reader) throws IOException {
        int numLayers = 0;
        List<Integer> layerSizes = Lists.newArrayList();
        BufferedReader br = new BufferedReader(reader);
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) {
                continue;
            }
            if ("w".equals(line)) {
                break;
            }
            int pos = line.indexOf('=');
            String key = line.substring(0, pos).trim();
            String value = line.substring(pos + 1).trim();
            if ("num_layers".equals(key)) {
                numLayers = Integer.parseInt(value);
            } else if ("layer_sizes".equals(key)) {
                for (String field : value.split(" ")) {
                    layerSizes.add(Integer.parseInt(field));
                }
            }
        }

        List<SimpleMatrix> w = Lists.newArrayList();
        for (int l = 1; l < numLayers; l++) {
            int rows = layerSizes.get(l);
            int cols = layerSizes.get(l - 1) + 1;
            SimpleMatrix m = new SimpleMatrix(rows, cols);
            int row = 0;
            while (row < rows) {
                line = br.readLine();
                if (line == null) {
                    break;
                }
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                String[] fields = line.split(" ");
                if (fields.length != cols) {
                    throw new RuntimeException("wrong number of columns");
                }
                for (int j = 0; j < cols; j++) {
                    m.set(row, j, Double.parseDouble(fields[j]));
                }
                row++;
            }
            w.add(m);
        }

        return new Network(w);
    }
}
