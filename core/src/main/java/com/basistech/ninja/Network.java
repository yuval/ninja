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
import com.google.common.base.Charsets;
import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Network {
    private static final Random RANDOM = new Random(8723643324L);
    private final List<Integer> layerSizes;
    private final NinjaMatrix[] w;
    private final Function activationFunction = Functions.SIGMOID;

    private NinjaMatrix[] grad;
    private NinjaMatrix[] deltaGrad;

    public Network(NinjaMatrix ... w) {
        this.w = w;
        layerSizes = updateLayerSizes();
        initWorkspace();
    }

    /**
     *
     * @param layerSizes number of units in each layer, not including
     *                   bias unit
     */
    public Network(List<Integer> layerSizes) {
        this.layerSizes = layerSizes;
        w = new NinjaMatrix[layerSizes.size() - 1];
        for (int i = 0; i < w.length; i++) {
            w[i] = new NinjaMatrix(layerSizes.get(i + 1), layerSizes.get(i) + 1);
        }
        randomInitialize();
        initWorkspace();
    }

    private void initWorkspace() {
        grad = new NinjaMatrix[getNumLayers() - 1];
        deltaGrad = new NinjaMatrix[grad.length];
        for (int l = 0; l < getNumLayers() - 1; l++) {
            grad[l] = new NinjaMatrix(w[l].numRows(), w[l].numCols());
            deltaGrad[l] = new NinjaMatrix(w[l].numRows(), w[l].numCols());
        }
    }

    private List<Integer> updateLayerSizes() {
        List<Integer> result = Lists.newArrayList();
        for (int i = 0; i < getNumLayers(); i++) {
            int layerSize = i == 0 ? w[i].numCols() - 1 : w[i - 1].numRows();
            result.add(layerSize);
        }
        return result;
    }

    static ColVector stripBiasUnit(ColVector vec) {
        ColVector result = new ColVector(vec.numRows() - 1);
        for (int i = 0; i < result.numRows(); i++) {
            result.set(i, vec.get(i + 1));
        }
        return result;
    }

    static ColVector addBiasUnit(ColVector vec) {
        ColVector result = new ColVector(vec.numRows() + 1);
        result.set(0, 1.0);
        for (int i = 0; i < vec.numRows(); i++) {
            result.set(i + 1, vec.get(i));
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
        return layerSizes.get(layer);
    }

    // 0-based
    public NinjaMatrix getWeightMatrix(int layer) {
        return w[layer].copy();
    }

    public static class ForwardVectors {
        ColVector[] z;
        ColVector[] a;
        ForwardVectors(ColVector[] z, ColVector[] a) {
            this.z = z;
            this.a = a;
        }
    }

    public ForwardVectors feedForward(ColVector vec) {
        return feedForward(vec.getData());
    }

    // z[0] is always null
    public ForwardVectors feedForward(double ... values) {
        int layers = w.length + 1;
        ColVector[] z = new ColVector[layers];
        ColVector[] a = new ColVector[layers];
        a[0] = Network.addBiasUnit(new ColVector(values));
        for (int l = 1; l < layers; l++) {
            z[l] = ColVector.mult(w[l - 1], a[l - 1]);
            a[l] = Functions.apply(activationFunction, z[l]);
            if (l != layers - 1) {
                a[l] = Network.addBiasUnit(a[l]);
            }
        }
        return new ForwardVectors(z, a);
    }

    public ColVector apply(double ... values) {
        return feedForward(values).a[getNumLayers() - 1];
    }

    public ColVector apply(ColVector x) {
        return apply(x.getData());
    }

    // TODO: The API would be simpler if instead we had 'List<Result> apply(ColVector vec)'
    public static List<Result> sort(ColVector vec) {
        List<Result> results = Lists.newArrayList();
        double[] values = vec.getData();
        for (int i = 0; i < values.length; i++) {
            results.add(new Result(i, values[i]));
        }
        // sorting by score
        Collections.sort(results, Collections.reverseOrder());
        return results;
    }

    // deltas[0] is always null
    public ColVector[] backprop(ForwardVectors fv, ColVector y) {
        int layers = getNumLayers();
        ColVector[] deltas = new ColVector[layers];  // just don't use slot 0
        // TODO: How to prevent copying without making it very hard to follow?
        deltas[layers - 1] = fv.a[layers - 1].copy();
        deltas[layers - 1].minus(y);
        for (int l = deltas.length - 2; l >= 1; l--) {
            // TODO: How to prevent copying? I don't think we can change w
            NinjaMatrix t = w[l].copy();
            t.transpose();
            ColVector v = ColVector.mult(t, deltas[l + 1]);
            v.elementMult(Functions.apply(Functions.SIGMOID_PRIME, Network.addBiasUnit(fv.z[l])));
            deltas[l] = Network.stripBiasUnit(v);
        }
        return deltas;
    }

    public void stochasticGD(ColVector[] x, ColVector[] y, double learningRate) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("x and y must be the same length!");
        }
        computeGradient(x, y);
        for (int i = 0; i < w.length; i++) {
            grad[i].scale(learningRate);
            w[i].minus(grad[i]);
        }
    }

    void computeGradient(ColVector[] x, ColVector[] y) {
        int numExamples = x.length;
        for (int i = 0; i < numExamples; i++) {
            ForwardVectors fv = feedForward(x[i]);
            ColVector[] deltas = backprop(fv, y[i]);
            for (int l = 0; l < grad.length; l++) {
                //grad[l].plus(deltas[l + 1].mult(fv.a[l].transpose()));
                NinjaMatrix transposed = fv.a[l].transpose();
                deltas[l + 1].mult(transposed, deltaGrad[l]);
                grad[l].plus(deltaGrad[l]);
            }
        }
        for (int i = 0; i < getNumLayers() - 1; i++) {
            grad[i].divide(numExamples);
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (NinjaMatrix m : w) {
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

        NinjaMatrix[] w = new NinjaMatrix[numLayers - 1];
        for (int l = 1; l < numLayers; l++) {
            int rows = layerSizes.get(l);
            int cols = layerSizes.get(l - 1) + 1;
            NinjaMatrix m = new NinjaMatrix(rows, cols);
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
            w[l - 1] = m;
        }

        return new Network(w);
    }

    void writeModel(BufferedWriter writer) throws IOException {
        // num_layers=3
        // layer_sizes=784 30 10
        // w
        // ...

        writer.write("num_layers=" + getNumLayers());
        writer.newLine();
        writer.write("layer_sizes=" + Joiner.on(' ').join(layerSizes));
        writer.newLine();
        writer.write("w");
        writer.newLine();
        writer.newLine();
        for (NinjaMatrix m : w) {
            for (int i = 0; i < m.numRows(); i++) {
                NinjaMatrix row = m.extractVector(true, i);
                writer.write(Joiner.on(' ').join(Doubles.asList(row.getData())));
                writer.newLine();
            }
            writer.newLine();
        }
    }

    void writeModel(File file) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
            new FileOutputStream(file), Charsets.UTF_8))) {
            writeModel(writer);
        }
    }
}
