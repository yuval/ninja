package com.basistech.ninja;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class MatrixTest {
    @Test
    public void testBasic() {
        Matrix m = new Matrix(3, 2, 1, 2, 3, 4, 5, 6);
        assertEquals(3, m.getRows());
        assertEquals(2, m.getColumns());
        assertEquals("3x2", m.getDimensions());
        assertEquals(m.get(0, 0), 1, 0.00001);
        assertEquals(m.get(0, 1), 2, 0.00001);
        assertEquals(m.get(1, 0), 3, 0.00001);
        assertEquals(m.get(1, 1), 4, 0.00001);
        assertEquals(m.get(2, 0), 5, 0.00001);
        assertEquals(m.get(2, 1), 6, 0.00001);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testIllegalAccess() {
        Matrix m = new Matrix(3, 2, 1, 2, 3, 4, 5, 6);
        m.get(0, 3);
    }

    @Test
    public void testTranspose() {
        Matrix m = new Matrix(3, 2, 1, 2, 3, 4, 5, 6).transpose();
        assertEquals(2, m.getRows());
        assertEquals(3, m.getColumns());
        assertEquals("2x3", m.getDimensions());
        assertEquals(m.get(0, 0), 1, 0.00001);
        assertEquals(m.get(0, 1), 3, 0.00001);
        assertEquals(m.get(0, 2), 5, 0.00001);
        assertEquals(m.get(1, 0), 2, 0.00001);
        assertEquals(m.get(1, 1), 4, 0.00001);
        assertEquals(m.get(1, 2), 6, 0.00001);
    }

    @Test
    public void testAdd() {
        Matrix a = new Matrix(3, 2, 1, 2, 3, 4, 5, 6);
        Matrix m = Matrix.add(a, a);
        assertEquals(m.get(0, 0), 2, 0.00001);
        assertEquals(m.get(0, 1), 4, 0.00001);
        assertEquals(m.get(1, 0), 6, 0.00001);
        assertEquals(m.get(1, 1), 8, 0.00001);
        assertEquals(m.get(2, 0), 10, 0.00001);
        assertEquals(m.get(2, 1), 12, 0.00001);
    }

    @Test
    public void testMultiply() {
        Matrix a = new Matrix(3, 2, 1, 2, 3, 4, 5, 6);
        Matrix b = a.transpose();
        Matrix m = Matrix.multiply(a, b);
        assertEquals(m.get(0, 0), 5, 0.00001);
        assertEquals(m.get(0, 1), 11, 0.00001);
        assertEquals(m.get(0, 2), 17, 0.00001);
        assertEquals(m.get(1, 0), 11, 0.00001);
        assertEquals(m.get(1, 1), 25, 0.00001);
        assertEquals(m.get(1, 2), 39, 0.00001);
        assertEquals(m.get(2, 0), 17, 0.00001);
        assertEquals(m.get(2, 1), 39, 0.00001);
        assertEquals(m.get(2, 2), 61, 0.00001);
    }

    @Test
    public void testApplyIdentity() {
        Matrix a = new Matrix(3, 2, 1, 2, 3, 4, 5, 6);
        Matrix m = a.apply(Functions.IDENTITY);
        assertEquals(m.get(0, 0), 1, 0.00001);
        assertEquals(m.get(0, 1), 2, 0.00001);
        assertEquals(m.get(1, 0), 3, 0.00001);
        assertEquals(m.get(1, 1), 4, 0.00001);
        assertEquals(m.get(2, 0), 5, 0.00001);
        assertEquals(m.get(2, 1), 6, 0.00001);
    }

    @Test
    public void testApplySigmoid() {
        Matrix a = new Matrix(3, 2, 1, 2, 3, 4, 5, 6);
        Matrix m = a.apply(Functions.SIGMOID);
        assertEquals(m.get(0, 0), 0.7310585786300049, 0.00001);
        assertEquals(m.get(0, 1), 0.8807970779778823, 0.00001);
        assertEquals(m.get(1, 0), 0.9525741268224334, 0.00001);
        assertEquals(m.get(1, 1), 0.9820137900379085, 0.00001);
        assertEquals(m.get(2, 0), 0.9933071490757153, 0.00001);
        assertEquals(m.get(2, 1), 0.9975273768433653, 0.00001);
    }

    @Test
    public void testInsertBias() {
        Matrix m = new Matrix(3, 1, 1, 2, 3).insertBias();
        assertEquals(1, m.get(0, 0), 0.00001);
        assertEquals(1, m.get(1, 0), 0.00001);
        assertEquals(2, m.get(2, 0), 0.00001);
        assertEquals(3, m.get(3, 0), 0.00001);
    }
}
