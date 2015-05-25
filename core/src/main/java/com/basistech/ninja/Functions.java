package com.basistech.ninja;

import org.ejml.simple.SimpleMatrix;

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

    public static SimpleMatrix apply(Function f, SimpleMatrix m) {
        SimpleMatrix result = new SimpleMatrix(m.numRows(), m.numCols());
        for (int i = 0; i < m.numRows(); i++) {
            for (int j = 0; j < m.numCols(); j++) {
                result.set(i, j, f.apply(m.get(i, j)));
            }
        }
        return result;
    }
}
