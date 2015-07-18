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
import java.util.List;

public class Train {
    private final Network net;
    private final File examplesFile;
    private SimpleMatrix x;
    private SimpleMatrix y;

    Train(List<Integer> layerSizes, File examplesFile) {
        net = new Network(layerSizes);
        this.examplesFile = examplesFile;
    }

    void train(int batchSize, int epochs, double learningRate, File modelFile) throws IOException {
        readExmaples(examplesFile);
        net.batchGD(x, y, epochs, learningRate);
        net.writeModel(modelFile);
    }

    void readExmaples(File file) throws IOException {
        int lines = 0;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
            new FileInputStream(file), Charsets.UTF_8))) {

            while (reader.readLine() != null) {
                lines++;
            }
        }
        x = new SimpleMatrix(lines, net.getNumNeurons(0));
        y = new SimpleMatrix(lines, net.getNumNeurons(net.getNumLayers() - 1));

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
            new FileInputStream(file), Charsets.UTF_8))) {

            int lineno = 0;
            String line;
            while ((line = reader.readLine()) != null) {
                // 1 1:1 2:1 5:1
                String[] fields = line.split("\\s+");
                int yval = Integer.valueOf(fields[0]);
                y.set(lineno, yval, 1.0);
                for (int i = 1; i < fields.length; i++) {
                    String[] feature = fields[i].split(":");
                    int index = Integer.valueOf(feature[0]);
                    double value = Double.valueOf(feature[1]);
                    x.set(lineno, index, value);
                }
                lineno++;
            }
        }
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 3) {
            System.err.println("Usage: Train config examples model");
            System.exit(1);
        }

        //void train(int batchSize, int epochs, double learningRate, File modelFile) throws IOException {
        List<Integer> layerSizes = Lists.newArrayList(784, 30, 10);
        Train that = new Train(layerSizes, new File(args[1]));
        that.train(123, 20, 0.8, new File(args[2]));
    }
}
