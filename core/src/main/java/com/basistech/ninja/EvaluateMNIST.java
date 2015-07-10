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
import org.ejml.simple.SimpleMatrix;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

public final class EvaluateMNIST {
    private EvaluateMNIST() {
    }

    static SimpleMatrix[] readExamples(File f, int numExamples) throws IOException {
        SimpleMatrix x = new SimpleMatrix(numExamples, 784);
        SimpleMatrix y = new SimpleMatrix(numExamples, 10);

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(f), Charsets.UTF_8))) {
            int raw = 0;
            String line;
            while ((line = reader.readLine()) != null) {
                // 1 qid:1 1:1 2:1 5:1
                String[] fields = line.split("\\s+");
                int actualDigit = Integer.valueOf(fields[0]);
                y.set(raw, actualDigit, 1.0);
                for (int i = 1; i < fields.length; i++) {
                    int col = Integer.valueOf(fields[i].split(":")[0]);
                    double value = Double.valueOf(fields[i].split(":")[1]);
                    x.set(raw, col, value);
                }
                raw++;
            }
        }

        return new SimpleMatrix[]{x, y};
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.err.println("Too few/many arguments: " + args.length);
            System.err.println("Usage: Train examples.train examples.test");
            System.exit(1);
        }

        File train = new File(args[0]);
        File test = new File(args[1]);

        SimpleMatrix[] examples = readExamples(train, 50000);
        System.out.println("Finished reading examples.");
        SimpleMatrix x = examples[0];
        SimpleMatrix y = examples[1];

        // training
        SimpleMatrix w1 = new SimpleMatrix(30, 785);
        SimpleMatrix w2 = new SimpleMatrix(10, 31);
        Network net = new Network(w1, w2);
        net.randomInitialize();

        int epochs = 150;
        double epsilon = 0.8; //learning rate
        net.batchGD(x, y, epochs, epsilon);

        // testing
        examples = readExamples(test, 10000);
        x = examples[0];
        y = examples[1];

        int correct = 0;
        for (int r = 0; r < x.numRows(); r++) {
            SimpleMatrix output = net.apply(x.extractVector(true, r));
            List<Result> predicted  = Network.sort(output);
            List<Result> actual = Network.sort(y.extractVector(true, r));
            if (predicted.get(0).getId() == actual.get(0).getId()) {
                correct++;
            }
        }

        System.out.println(correct);
    }

}
