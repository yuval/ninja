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

package com.basistech.ninja.ejml;

import org.ejml.EjmlParameters;
import org.ejml.data.DenseMatrix32F;
import org.ejml.ops.MatrixDimensionException;

import java.util.Arrays;

public class CommonOpsMatrix32F {

    public static void transpose(DenseMatrix32F mat) {
        if( mat.numCols == mat.numRows ){
            square(mat);
        } else {
            DenseMatrix32F b = new DenseMatrix32F(mat.numCols,mat.numRows);
            transpose(mat, b);
            mat.set(b);
        }
    }

    public static DenseMatrix32F transpose(DenseMatrix32F A, DenseMatrix32F A_tran)
    {
        if( A_tran == null ) {
            A_tran = new DenseMatrix32F(A.numCols,A.numRows);
        } else {
            if( A.numRows != A_tran.numCols || A.numCols != A_tran.numRows ) {
                throw new IllegalArgumentException("Incompatible matrix dimensions");
            }
        }

        if( A.numRows > EjmlParameters.TRANSPOSE_SWITCH &&
                A.numCols > EjmlParameters.TRANSPOSE_SWITCH )
            block(A, A_tran, EjmlParameters.BLOCK_WIDTH);
        else
            standard(A, A_tran);

        return A_tran;
    }

    // copied from org.ejml.alg.dense.misc.TransposeAlgs
    public static void block(DenseMatrix32F A, DenseMatrix32F A_tran,
                              final int blockLength)
    {
        for( int i = 0; i < A.numRows; i += blockLength ) {
            int blockHeight = Math.min( blockLength , A.numRows - i);

            int indexSrc = i*A.numCols;
            int indexDst = i;

            for( int j = 0; j < A.numCols; j += blockLength ) {
                int blockWidth = Math.min( blockLength , A.numCols - j);

//                int indexSrc = i*A.numCols + j;
//                int indexDst = j*A_tran.numCols + i;

                int indexSrcEnd = indexSrc + blockWidth;
//                for( int l = 0; l < blockWidth; l++ , indexSrc++ ) {
                for( ; indexSrc < indexSrcEnd;  indexSrc++ ) {
                    int rowSrc = indexSrc;
                    int rowDst = indexDst;
                    int end = rowDst + blockHeight;
//                    for( int k = 0; k < blockHeight; k++ , rowSrc += A.numCols ) {
                    for( ; rowDst < end; rowSrc += A.numCols ) {
                        // faster to write in sequence than to read in sequence
                        A_tran.data[ rowDst++ ] = A.data[ rowSrc ];
                    }
                    indexDst += A_tran.numCols;
                }
            }
        }
    }

    // copied from org.ejml.alg.dense.misc.TransposeAlgs
    public static void standard(DenseMatrix32F A, DenseMatrix32F A_tran) {
        int index = 0;
        for( int i = 0; i < A_tran.numRows; i++ ) {
            int index2 = i;

            int end = index + A_tran.numCols;
            while( index < end ) {
                A_tran.data[index++ ] = A.data[ index2 ];
                index2 += A.numCols;
            }
        }
    }

    // copied from org.ejml.alg.dense.misc.TransposeAlgs
    public static void square(DenseMatrix32F mat)
    {
        int index = 1;
        int indexEnd = mat.numCols;
        for( int i = 0; i < mat.numRows;
             i++ , index += i+1 , indexEnd += mat.numCols ) {
            int indexOther = (i+1)*mat.numCols + i;
            for( ; index < indexEnd; index++, indexOther += mat.numCols) {
                float val = mat.data[ index ];
                mat.data[ index ] = mat.data[ indexOther ];
                mat.data[indexOther] = val;
            }
        }
    }

    public static void addEquals(DenseMatrix32F a, DenseMatrix32F b ) {
        if( a.numCols != b.numCols || a.numRows != b.numRows ) {
            throw new IllegalArgumentException("The 'a' and 'b' matrices do not have compatible dimensions");
        }

        final int length = a.getNumElements();

        for( int i = 0; i < length; i++ ) {
            a.plus(i, b.get(i));
        }
    }

    public static void subtractEquals(DenseMatrix32F a, DenseMatrix32F b) {
        if( a.numCols != b.numCols || a.numRows != b.numRows ) {
            throw new IllegalArgumentException("The 'a' and 'b' matrices do not have compatible dimensions");
        }

        final int length = a.getNumElements();

        for( int i = 0; i < length; i++ ) {
            a.data[i] -= b.data[i];
        }
    }

    public static void elementMult(DenseMatrix32F a, DenseMatrix32F b ) {
        if( a.numCols != b.numCols || a.numRows != b.numRows ) {
            throw new IllegalArgumentException("The 'a' and 'b' matrices do not have compatible dimensions");
        }

        int length = a.getNumElements();

        for( int i = 0; i < length; i++ ) {
            a.times(i , b.get(i));
        }
    }

    public static void mult(DenseMatrix32F a, DenseMatrix32F b, DenseMatrix32F c ) {
        if( b.numCols == 1 ) {
            multColVec(a, b, c);
        } else if( b.numCols >= EjmlParameters.MULT_COLUMN_SWITCH ) {
            mult_reorder(a, b, c);
        } else {
            mult_small(a, b, c);
        }
    }

    // copied from MatrixVectorMult#mult
    public static void multColVec(DenseMatrix32F A, DenseMatrix32F B, DenseMatrix32F C) {
        if( C.numCols != 1 ) {
            throw new MatrixDimensionException("C is not a column vector");
        } else if( C.numRows != A.numRows ) {
            throw new MatrixDimensionException("C is not the expected length");
        }

        if( B.numRows == 1 ) {
            if( A.numCols != B.numCols ) {
                throw new MatrixDimensionException("A and B are not compatible");
            }
        } else if( B.numCols == 1 ) {
            if( A.numCols != B.numRows ) {
                throw new MatrixDimensionException("A and B are not compatible");
            }
        } else {
            throw new MatrixDimensionException("B is not a vector");
        }

        if( A.numCols == 0 ) {
            fill(C, 0);
            return;
        }

        int indexA = 0;
        int cIndex = 0;
        float b0 = B.get(0);
        for( int i = 0; i < A.numRows; i++ ) {
            float total = A.get(indexA++) * b0;

            for( int j = 1; j < A.numCols; j++ ) {
                total += A.get(indexA++) * B.get(j);
            }

            C.set(cIndex++, total);
        }
    }

    public static void mult_reorder(DenseMatrix32F a, DenseMatrix32F b ,DenseMatrix32F c) {
        if( a == c || b == c )
            throw new IllegalArgumentException("Neither 'a' or 'b' can be the same matrix as 'c'");
        else if( a.numCols != b.numRows ) {
            throw new MatrixDimensionException("The 'a' and 'b' matrices do not have compatible dimensions");
        } else if( a.numRows != c.numRows || b.numCols != c.numCols ) {
            throw new MatrixDimensionException("The results matrix does not have the desired dimensions");
        }

        if( a.numCols == 0 || a.numRows == 0 ) {
            fill(c, 0);
            return;
        }
        float valA;
        int indexCbase= 0;
        int endOfKLoop = b.numRows*b.numCols;

        for( int i = 0; i < a.numRows; i++ ) {
            int indexA = i*a.numCols;

            // need to assign c.data to a value initially
            int indexB = 0;
            int indexC = indexCbase;
            int end = indexB + b.numCols;

            valA = a.get(indexA++);

            while( indexB < end ) {
                c.set(indexC++ , valA*b.get(indexB++));
            }

            // now add to it
            while( indexB != endOfKLoop ) { // k loop
                indexC = indexCbase;
                end = indexB + b.numCols;

                valA = a.get(indexA++);

                while( indexB < end ) { // j loop
                    c.plus(indexC++ , valA*b.get(indexB++));
                }
            }
            indexCbase += c.numCols;
        }
    }

    public static void fill(DenseMatrix32F a, float value) {
        Arrays.fill(a.data, 0, a.getNumElements(), value);
    }

    public static void mult_small(DenseMatrix32F a , DenseMatrix32F b , DenseMatrix32F c ) {
        if( a == c || b == c )
            throw new IllegalArgumentException("Neither 'a' or 'b' can be the same matrix as 'c'");
        else if( a.numCols != b.numRows ) {
            throw new MatrixDimensionException("The 'a' and 'b' matrices do not have compatible dimensions");
        } else if( a.numRows != c.numRows || b.numCols != c.numCols ) {
            throw new MatrixDimensionException("The results matrix does not have the desired dimensions");
        }

        int aIndexStart = 0;
        int cIndex = 0;

        for( int i = 0; i < a.numRows; i++ ) {
            for( int j = 0; j < b.numCols; j++ ) {
                float total = 0;

                int indexA = aIndexStart;
                int indexB = j;
                int end = indexA + b.numRows;
                while( indexA < end ) {
                    total += a.get(indexA++) * b.get(indexB);
                    indexB += b.numCols;
                }

                c.set( cIndex++ , total );
            }
            aIndexStart += a.numCols;
        }
    }

    public static void scale(float alpha, DenseMatrix32F a) {
        // on very small matrices (2 by 2) the call to getNumElements() can slow it down
        // slightly compared to other libraries since it involves an extra multiplication.
        final int size = a.getNumElements();

        for( int i = 0; i < size; i++ ) {
            a.data[i] *= alpha;
        }
    }

    public static void subvector(DenseMatrix32F A, int rowA, int colA, int length , boolean row, int offsetV, DenseMatrix32F v) {
        if( row ) {
            for( int i = 0; i < length; i++ ) {
                v.set( offsetV +i , A.get(rowA,colA+i) );
            }
        } else {
            for( int i = 0; i < length; i++ ) {
                v.set( offsetV +i , A.get(rowA+i,colA));
            }
        }
    }

    public static void divide(DenseMatrix32F a , float alpha) {
        final int size = a.getNumElements();

        for( int i = 0; i < size; i++ ) {
            a.data[i] /= alpha;
        }
    }

    // copied from MatrixFeatures.isIdentical
    public static boolean isIdentical(DenseMatrix32F a, DenseMatrix32F b , float tol ) {
        if( a.numRows != b.numRows || a.numCols != b.numCols ) {
            return false;
        }
        if( tol < 0 )
            throw new IllegalArgumentException("Tolerance must be greater than or equal to zero.");

        final int length = a.getNumElements();
        for( int i = 0; i < length; i++ ) {
            float valA = a.get(i);
            float valB = b.get(i);

            // if either is negative or positive infinity the result will be positive infinity
            // if either is NaN the result will be NaN
            float diff = Math.abs(valA-valB);

            // diff = NaN == false
            // diff = infinity == false
            if( tol >= diff )
                continue;

            if( Double.isNaN(valA) ) {
                return Double.isNaN(valB);
            } else if( Double.isInfinite(valA) ) {
                return valA == valB;
            } else {
                return false;
            }
        }

        return true;
    }

}
