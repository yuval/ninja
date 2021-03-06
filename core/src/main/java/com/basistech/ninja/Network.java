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
import com.basistech.ninja.ejml.NinjaMatrix;
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

/**
 * {@code Network} represents a neural network. It provides methods to train a
 * model and make predictions from a model. Currently, it implements a fully connected
 * feed forward network.
 */
public class Network {
    private static final Random RANDOM = new Random(8723643324L);
    private final List<Integer> layerSizes;
    private final NinjaMatrix[] w;
    private final Function activationFunction = Functions.SIGMOID;

    /**
     * Constructs a network from weight matrices. The number of
     * weight matrices defines the number of layers in the network.
     * A network with three layers (input, hidden, and output) has two
     * weight matrices.
     *
     * @param w the weight matrices
     */
    public Network(NinjaMatrix ... w) {
        this.w = w;
        layerSizes = computeLayerSizes();
    }

    /**
     * Constructs a randomly initialized network from a given architecture.
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
    }

    private List<Integer> computeLayerSizes() {
        // computes the architecture using the weight matrices
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
        int inputSize = getNumUnits(layer);
        int outputSize = getNumUnits(layer + 1);
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

    /**
     * Randomly initializes weight matrices. The initialization scheme is
     * similar to the one used in Andrew Ng's machine learning class.
     */
    // TODO: think of a way to initialize during construction
    public void randomInitialize() {
        for (int l = 0; l < getNumLayers() - 1; l++) {
            randomInitializeLayer(l);
        }
    }

    /**
     * Returns the number of layers in the network including input and output layers.
     *
     * @return the number of layers
     */
    public int getNumLayers() {
        return w.length + 1;
    }

    /**
     * Returns the number of units for the given layer, including the bias unit.
     *
     * @param layer the layer number (input layer is layer zero)
     * @return the number of units
     */
    public int getNumUnits(int layer) {
        return layerSizes.get(layer);
    }

    /**
     * Returns the weight matrix for the given layer.
     *
     * @param layer the layer number (input layer is layer zero)
     * @return the weight matrix
     */
    public NinjaMatrix getWeightMatrix(int layer) {
        return w[layer].copy();
    }

    ForwardVectors feedForward(ColVector vec) {
        return feedForward(vec.getData());
    }

    static class ForwardVectors {
        ColVector[] z;
        ColVector[] a;
        ForwardVectors(ColVector[] z, ColVector[] a) {
            this.z = z;
            this.a = a;
        }
    }

    // z[0] is always null
    ForwardVectors feedForward(double ... values) {
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

    /**
     * Applies the input to the network. The current implementation runs
     * feed forward.
     *
     * @param values the input values
     * @return the network output for the given input
     */
    public ColVector apply(double... values) {
        return feedForward(values).a[getNumLayers() - 1];
    }

    /**
     * Applies the input to the network. The current implementation runs
     * feed forward.
     *
     * @param input the input vector
     * @return the network output for the given input
     */
    public ColVector apply(ColVector input) {
        return apply(input.getData());
    }

    /**
     * Sorts the vector by value, retaining original index. Typically used
     * after calling 'apply' to get ordered predictions.
     *
     * @param vec the vector to sort
     * @return the list of ordered results
     */
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
    ColVector[] backprop(ForwardVectors fv, ColVector y) {
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

    /**
     * Updates the weight matrices given a batch of training examples.
     * Throws IllegalArgumentException if 'x' and 'y' have different lengths.
     *
     * @param x a batch of inputs
     * @param y a batch of outputs
     * @param learningRate the learning rate to use during training
     */
    public void trainBatch(ColVector[] x, ColVector[] y, double learningRate) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("x and y must be the same length!");
        }
        // TODO: gradient checking
        NinjaMatrix[] grad = computeGradient(x, y);
        for (int i = 0; i < w.length; i++) {
            grad[i].scale(learningRate);
            w[i].minus(grad[i]);
        }
    }

    NinjaMatrix[] computeGradient(ColVector[] x, ColVector[] y) {
        int numExamples = x.length;

        NinjaMatrix[] bigDelta = new NinjaMatrix[getNumLayers() - 1];
        for (int l = 0; l < getNumLayers() - 1; l++) {
            bigDelta[l] = new NinjaMatrix(w[l].numRows(), w[l].numCols());
        }

        for (int i = 0; i < numExamples; i++) {
            ForwardVectors fv = feedForward(x[i]);
            ColVector[] deltas = backprop(fv, y[i]);
            for (int l = 0; l < bigDelta.length; l++) {
                bigDelta[l].plus(deltas[l + 1].mult(fv.a[l].transpose()));
            }
        }

        for (int i = 0; i < getNumLayers() - 1; i++) {
            // this is the gradient
            bigDelta[i].divide(numExamples);
        }

        return bigDelta;
    }

    /**
     * Loads a model from a text file.
     * See README file for model format.
     *
     * @param file the model file
     * @return a new network instance
     * @throws IOException
     */
    public static Network loadModel(File file) throws IOException {
        try (Reader reader = new InputStreamReader(new FileInputStream(file), Charsets.US_ASCII)) {
            return Network.loadModel(reader);
        }
    }

    /**
     * Loads a model from a reader.
     * See README file for model format.
     *
     * @param reader the model reader
     * @return a new network instance
     * @throws IOException
     */
    // TODO: better error messages, e.g. line number, invalid header fields, too many lines, etc.
    public static Network loadModel(Reader reader) throws IOException {
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

    /**
     * Writes a model to a writer.
     * See README file for model format.
     *
     * @param writer the writer
     * @throws IOException
     */
    public void writeModel(BufferedWriter writer) throws IOException {
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

    /**
     * Writes a model to a file.
     * See README file for model format.
     *
     * @param file the output file
     * @throws IOException
     */
    public void writeModel(File file) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
            new FileOutputStream(file), Charsets.UTF_8))) {
            writeModel(writer);
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
}
