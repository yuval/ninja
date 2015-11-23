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

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.List;

/**
 * Command line driver for generating predictions given a trained neural network model
 * and an examples file in the following format:
 *
 * <ul>
 *  <li> one example per line
 *  <li> fields are separated by a space
 *  <li> first field is an integer label (required but currently ignored for prediction)
 *  <li> remaining fields are of the form feature:value, where feature is an integer and
 *       value is a floating point number feature indexes start at 0
 * </ul>
 *
 * For example:
 *
 * <pre>
 *  6 0:0.0 1:0.0 ...  99:0.09375000 ... 783:0.0
 *  2 0:0.0 1:0.0 ... 151:0.26171875 ... 783:0.0
 *  3 0:0.0 1:0.0 ... 152:0.99609375 ... 783:0.0
 * </pre>
 */
public class Predict {
    Network net;

    Predict(Network net) {
        this.net = net;
    }

    List<Result> predict(ColVector x) {
        ColVector outVector = net.apply(x);
        return Network.sort(outVector);
    }

    /**
     * Command line interface to make predictions.
     *
     * <pre>
     *  Usage: Predict model examples response [--verbose]
     * </pre>
     *
     * @param args command line arguments
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        if (args.length != 3 && args.length != 4) {
            System.err.println("Usage: Predict model examples response [--verbose]");
            System.exit(1);
        }

        Network net = Network.loadModel(new File(args[0]));
        File examplesFile = new File(args[1]);
        File responseFile = new File(args[2]);
        boolean verbose = args.length == 4 && args[3].equalsIgnoreCase("--verbose");
        Predict that = new Predict(net);

        int inputNeurons = net.getNumUnits(0);

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
            new FileInputStream(examplesFile), Charsets.UTF_8));
             BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
                 new FileOutputStream(responseFile), Charsets.UTF_8))) {

            int lineno = 0;
            String line;
            while ((line = reader.readLine()) != null) {
                // 1 1:1 2:1 5:1
                String[] fields = line.split("\\s+");
                ColVector x = new ColVector(inputNeurons);
                for (int i = 1; i < fields.length; i++) {
                    String[] feature = fields[i].split(":");
                    int index = Integer.parseInt(feature[0]);
                    double value = Double.parseDouble(feature[1]);
                    if (index < 0 || index >= inputNeurons) {
                        throw new RuntimeException(
                            String.format(
                                "line %d: index (%d) out of range [0, %d); wrong network architecture?",
                                lineno + 1,
                                index,
                                inputNeurons));
                    }
                    x.set(index, value);
                }

                List<Result> results = that.predict(x);
                String prediction = String.valueOf(results.get(0).getIndex());
                writer.append(prediction);
                writer.append('\t');
                writer.append(String.format("%f", results.get(0).getScore()));
                writer.newLine();
                if (verbose) {
                    for (Result result : results) {
                        writer.append(String.format("\t%f\t%s", result.getScore(), result.getIndex()));
                        writer.newLine();
                    }
                }
                lineno++;
            }
        }
    }
}
