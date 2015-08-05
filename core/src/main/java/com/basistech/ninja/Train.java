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
        int numBatches = (int) Math.ceil((double) x.numRows() / batchSize);
        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch: " + i);
            for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
                int startRow = batchIndex * batchSize;
                int endRow = (batchIndex + 1) * batchSize;
                if (endRow > x.numRows()) {
                    endRow = x.numRows();
                }
                SimpleMatrix batchX = x.extractMatrix(startRow, endRow, 0, x.numCols());
                SimpleMatrix batchY = y.extractMatrix(startRow, endRow, 0, y.numCols());
                net.stochasticGD(batchX, batchY, learningRate);
            }
        }
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
//                                             size      tp        fn        fp         P         R        F1
//        0                                    980       963        17        38     0.962     0.983     0.972
//        1                                   1135      1118        17        27     0.976     0.985     0.981
//        2                                   1032       973        59        36     0.964     0.943     0.953
//        3                                   1010       972        38        66     0.936     0.962     0.949
//        4                                    982       940        42        28     0.971     0.957     0.964
//        5                                    892       845        47        26     0.970     0.947     0.959
//        6                                    958       933        25        44     0.955     0.974     0.964
//        7                                   1028       979        49        36     0.965     0.952     0.958
//        8                                    974       924        50        54     0.945     0.949     0.947
//        9                                   1009       949        60        49     0.951     0.941     0.946
//
//        Total counts:                      10000      9596       404       404         -         -         -
//        Macro Average:                         -         -         -         -     0.960     0.959     0.959
//        Micro Average:                         -         -         -         -     0.960     0.960     0.960
        that.train(100, 15, 0.7, new File(args[2]));
    }
}
