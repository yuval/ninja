Neural Networks in Java (ninja)
===============================

Introduction
------------

Ninja is 100% Java neural net library that supports multi-class
classification.  The current target is small- or medium-sized
datasets.  The work is based on [Neural Networks and Deep
Learning](http://neuralnetworksanddeeplearning.com/) and Andrew Ng's
[machine learning course](https://www.coursera.org/learn/machine-learning).

Building
--------

```
$ git clone git@github.com:yuval/ninja.git
$ cd ninja
$ mvn clean install
```

Sample Data
-----------

We provide a small portion of the [MNIST data
set](http://www.deeplearning.net/tutorial/gettingstarted.html) for
handwritten digit recognition.  There are 500 examples in the training
set and 100 examples in the test set.

```
$ wc -l samples/data/mnist/*
    100 samples/data/mnist/examples.test
    500 samples/data/mnist/examples.train
```

Each example has an integer label from 0 through 9 which represents
the corresponding digit.  The features, numbered 0 through 783, are a
flattened representation of the gray-scaled pixel values from the
28x28 image.

Examples Format
---------------

The command line utilities `Train` and `Predict` require an examples
file with the following format:

* one example per line
* fields are separated by a space
* first field is an integer label
* remaining field are of the form `feature:value`, where `feature` is
  an integer and value is a floating point number
* feature indexes start at 0

For example, here are the first few lines on the sample training data:

```
$ head -n3 samples/data/mnist/examples.train
6 0:0.0 1:0.0 ...  99:0.09375000 ... 783:0.0
2 0:0.0 1:0.0 ... 151:0.26171875 ... 783:0.0
3 0:0.0 1:0.0 ... 152:0.99609375 ... 783:0.0
```

In this example, all feature values are shown, including zero-valued
features.  However, ninja supports sparse examples where only
non-zero-valued features are included.

Training
--------

The `Train` command line program trains a model and saves it to disk.

```
$ script/run-java.sh com.basistech.ninja.Train
Missing required options: examples, model, layer-sizes
usage: Train [options]
    --batch-size <arg>      batch size (default = 10)
    --epochs <arg>          epochs (default = 5)
    --examples <arg>        input examples file (required)
    --layer-sizes <arg>     layer sizes, including input/output, e.g. 3 4 2
                            (required)
    --learning-rate <arg>   learning-rate (default = 0.7)
    --model <arg>           output model file (required)
```

`run-java.sh` is just a helper script to launch java with the maven
classpath.  Windows users can run:

```
$ cd core
$ mvn exec:java -Dexec.mainClass=com.basistech.ninja.Train
```

For our sample data, there are 784 input features (representing pixels
from a 28x28 image) and 10 outputs, representing each of the digits, 0
through 9.

This fixes the first and last values in the `--layer-sizes` parameter.
You may have one or more hidden layers, each with as many units as
desired.  In the example below, we choose a single hidden layer with
30 units.  So we pass ``--layer-sizes 784 30 10``.

The learning algorithm is Stochastic Gradient Descent with a
configurable batch size.  An `epoch` is one iteration over all the
training examples.  The learning will stop after the final epoch.
Both `epoch` and `batch-size` affect training speed and model
accuracy.

```
$ time script/run-java.sh com.basistech.ninja.Train \
--examples samples/data/mnist/examples.train \
--model model --layer-sizes 784 30 10
Epoch: 1
Epoch: 2
Epoch: 3
Epoch: 4
Epoch: 5

real	0m1.651s
user	0m2.140s
sys	0m0.080s
```

This write the `model` file to disk.  It looks like this:

```
$ cat model
num_layers=3
layer_sizes=784 30 10
w

0.08056926059161915 0.040969447167538975 0.04270309844302295 ...
...

-0.2537711745652441 -0.8768034539787096 -1.6626568463542548
...
```

The floating point numbers represent the learned weight matrices.  The
weight matrix for each layer is separated by an empty line.

Prediction
----------

The `Predict` command line takes a model and an examples file, and
produces a file of predictions.  Currently, the examples file must
include labels, even though they are ignored.

```
$ script/run-java.sh com.basistech.ninja.Predict
Usage: Predict model examples response [--verbose]
```

```
$ script/run-java.sh com.basistech.ninja.Predict \
model samples/data/mnist/examples.test response.txt
```

Below, we show the gold labels on the left and the predicted labels on
the right, for the first 10 test examples.  The model predicts 8 of
the 10 examples correctly.

```
$ paste <(head samples/data/mnist/examples.test | cut -f1 -d ' ') \
<(head response.txt)
2	2	0.988703
3	3	0.833391
1	1	0.980135
5	0	0.304052
1	1	0.954124
4	6	0.432976
5	5	0.501261
1	1	0.818590
0	0	0.925333
7	7	0.984165
```

The accuracy on the full test set is also 80%. The third column is the score of the predicted output.
You can run Predict in verbose mode to see the score of every output node. 

You may get better results by tuning the learning parameters,
e.g. number epochs, learning-rate, etc.

Authors
-------

[Yuval Merhav](https://github.com/yuval/)

[Joel Barry](https://github.com/joelb-git/)

License
-------

Ninja is released under the Apache License.
