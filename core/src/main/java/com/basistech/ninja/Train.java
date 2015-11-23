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
import com.google.common.collect.Lists;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Command line driver for training a neural network given an examples file in
 * the following format:
 *
 * <ul>
 *  <li> one example per line
 *  <li> fields are separated by a space
 *  <li> first field is an integer label
 *  <li> remaining fields are of the form feature:value, where feature is an integer and
 *       value is a floating point number feature indexes start at 0
 * </ul>
 * For example, here are the first few lines of the sample training data:
 *
 * <pre>
 *  $ head -n3 samples/data/mnist/examples.train
 *  6 0:0.0 1:0.0 ...  99:0.09375000 ... 783:0.0
 *  2 0:0.0 1:0.0 ... 151:0.26171875 ... 783:0.0
 *  3 0:0.0 1:0.0 ... 152:0.99609375 ... 783:0.0
 * </pre>
 */
public class Train {
    private final Network net;
    private final File examplesFile;
    private ColVector[] x;
    private ColVector[] y;

    Train(List<Integer> layerSizes, File examplesFile) {
        net = new Network(layerSizes);
        this.examplesFile = examplesFile;
    }

    void train(int batchSize, int epochs, double learningRate, File modelFile) throws IOException {
        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch: " + (i + 1));
            for (List<String> batch : new ExamplesIterator(examplesFile, batchSize)) {
                parseExamples(batch);
                net.trainBatch(x, y, learningRate);
            }
        }
        net.writeModel(modelFile);
    }

    void parseExamples(List<String> lines) {
        int inputNeurons = net.getNumUnits(0);
        int outputNeurons = net.getNumUnits(net.getNumLayers() - 1);

        x = new ColVector[lines.size()];
        y = new ColVector[lines.size()];

        int lineno = 0;
        for (String line : lines) {
            x[lineno] = new ColVector(inputNeurons);
            y[lineno] = new ColVector(outputNeurons);

            // 1 1:1 2:1 5:1
            String[] fields = line.split("\\s+");
            int yval = Integer.parseInt(fields[0]);
            if (yval < 0 || yval >= outputNeurons) {
                // TODO: Fix bogus lineno
                throw new RuntimeException(
                        String.format(
                                "line %d: yval (%d) out of range [0, %d); wrong network architecture?",
                                lineno + 1,
                                yval,
                                outputNeurons));
            }
            y[lineno].set(yval, 1.0);
            for (int i = 1; i < fields.length; i++) {
                String[] feature = fields[i].split(":");
                int index = Integer.parseInt(feature[0]);
                double value = Double.parseDouble(feature[1]);
                if (index < 0 || index >= inputNeurons) {
                    // TODO: Fix bogus lineno
                    throw new RuntimeException(
                            String.format(
                                    "line %d: index (%d) out of range [0, %d); wrong network architecture?",
                                    lineno + 1,
                                    index,
                                    inputNeurons));
                }
                x[lineno].set(index, value);
            }
            lineno++;
        }
    }

    private static void usage(Options options) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.setWidth(80);
        formatter.printHelp("Train [options]", options);
        System.out.println();
    }

    /**
     * Command line interface to train a model.
     *
     * <pre>
     *  usage: Train [options]
     *  --batch-size <arg>      batch size (default = 10)
     *  --epochs <arg>          epochs (default = 5)
     *  --examples <arg>        input examples file (required)
     *  --layer-sizes <arg>     layer sizes, including input/output, e.g. 3 4 2 (required)
     *  --learning-rate <arg>   learning-rate (default = 0.7)
     *  --model <arg>           output model file (required)
     * </pre>
     *
     * @param args command line arguments
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        String defaultBatchSize = "10";
        String deafaultEpochs = "5";
        String defaultLearningRate = "0.7";

        Options options = new Options();
        Option option;
        option = new Option(null, "examples", true, "input examples file (required)");
        option.setRequired(true);
        options.addOption(option);
        option = new Option(null, "model", true, "output model file (required)");
        option.setRequired(true);
        options.addOption(option);
        option = new Option(null, "layer-sizes", true,
            "layer sizes, including input/output, e.g. 3 4 2 (required)");
        option.setRequired(true);
        option.setArgs(Option.UNLIMITED_VALUES);
        options.addOption(option);
        option = new Option(null, "batch-size", true,
            String.format("batch size (default = %s)", defaultBatchSize));
        options.addOption(option);
        option = new Option(null, "epochs", true,
            String.format("epochs (default = %s)", deafaultEpochs));
        options.addOption(option);
        option = new Option(null, "learning-rate", true,
            String.format("learning-rate (default = %s)", defaultLearningRate));
        options.addOption(option);

        CommandLineParser parser = new GnuParser();
        CommandLine cmdline = null;
        try {
            cmdline = parser.parse(options, args);
        } catch (org.apache.commons.cli.ParseException e) {
            System.err.println(e.getMessage());
            usage(options);
            System.exit(1);
        }
        String[] remaining = cmdline.getArgs();
        if (remaining == null) {
            usage(options);
            System.exit(1);
        }

        List<Integer> layerSizes = Lists.newArrayList();
        for (String s : cmdline.getOptionValues("layer-sizes")) {
            layerSizes.add(Integer.parseInt(s));
        }

        File examplesFile = new File(cmdline.getOptionValue("examples"));
        Train that = new Train(layerSizes, examplesFile);
        int batchSize = Integer.parseInt(cmdline.getOptionValue("batch-size", defaultBatchSize));
        int epochs = Integer.parseInt(cmdline.getOptionValue("epochs", deafaultEpochs));
        double learningRate = Double.parseDouble(cmdline.getOptionValue("learning-rate", defaultLearningRate));
        File modelFile = new File(cmdline.getOptionValue("model"));

        that.train(batchSize, epochs, learningRate, modelFile);
    }
}
