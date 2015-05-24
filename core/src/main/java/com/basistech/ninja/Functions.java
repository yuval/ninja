package com.basistech.ninja;

public class Functions {
    static final Function IDENTITY = new Identity();
    static final Function SIGMOID = new Sigmoid();

    private static class Identity implements Function {
        public double apply(double x) {
            return x;
        }
    }

    private static class Sigmoid implements Function {
        public double apply(double x) {
            return 1.0 / (1 + Math.exp(-x));
        }
    }
}
