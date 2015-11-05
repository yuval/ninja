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

public class Predict {
    Network net;

    Predict(Network net) {
        this.net = net;
    }

    List<Result> predict(ColVector x) {
        ColVector outVector = net.apply(x);
        return Network.sort(outVector);
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 3 && args.length != 4) {
            System.err.println("Usage: Predict model examples response [--verbose]");
            System.exit(1);
        }

        Network net = Network.fromText(new File(args[0]));
        File examplesFile = new File(args[1]);
        File responseFile = new File(args[2]);
        boolean verbose = args.length == 4 && args[3].equalsIgnoreCase("--verbose");
        Predict that = new Predict(net);

        int inputNeurons = net.getNumNeurons(0);

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
                    int index = Integer.valueOf(feature[0]);
                    double value = Double.valueOf(feature[1]);
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
                String prediction = String.valueOf(results.get(0).getId());
                writer.append(prediction);
                writer.append('\t');
                writer.append(String.format("%f", results.get(0).getScore()));
                writer.newLine();
                if (verbose) {
                    for (Result result : results) {
                        writer.append(String.format("\t%f\t%s", result.getScore(), result.getId()));
                        writer.newLine();
                    }
                }
                lineno++;
            }
        }
    }
}
