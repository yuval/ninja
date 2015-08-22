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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

public class ExamplesIterator implements Iterator<List<String>>, Iterable<List<String>> {
    private final BufferedReader reader;
    private final int batchSize;
    private int linesSeen;
    private int numLines;

    // taking a file and not a reader so that we can
    // iterate over it twice. TODO: fix this
    public ExamplesIterator(File f, int batchSize) throws IOException {
        // TODO: Avoid reading the file twice!
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(f), Charsets.UTF_8))) {
            while (reader.readLine() != null) {
                numLines++;
            }
        }
        this.reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(f), Charsets.UTF_8));
        this.batchSize = batchSize;
    }

    /**
     * Returns an iterator over a set of elements of type List<String>.
     *
     * @return an Iterator.
     */
    @Override
    public Iterator<List<String>> iterator() {
        return this;
    }

    /**
     * Returns {@code true} if the iteration has more elements.
     * (In other words, returns {@code true} if {@link #next} would
     * return an element rather than throwing an exception.)
     *
     * @return {@code true} if the iteration has more elements
     */
    @Override
    public boolean hasNext() {
        boolean hasNext = linesSeen < numLines;
        if (!hasNext) {
            try {
                reader.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        return hasNext;
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     * @throws NoSuchElementException if the iteration has no more elements
     */
    @Override
    public List<String> next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        List<String> batch = Lists.newArrayList();
        String line;
        try {
            while ((line = reader.readLine()) != null) {
                batch.add(line);
                linesSeen++;
                if (batch.size() == batchSize) {
                    return batch;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return batch;
    }

    /**
     * @throws UnsupportedOperationException the {@code remove}
     * operation is not supported by this iterator
     */
    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
