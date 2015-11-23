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

import com.google.common.base.Objects;

/**
 * {@code Result} represents the score of an output node.
 */
public class Result implements Comparable<Result> {
    private final int index;
    private final double score;

    Result(int index, double score) {
        this.index = index;
        this.score = score;
    }

    /**
     * Returns the index of this output node.
     *
     * @return the index of this output node
     */
    public int getIndex() {
        return index;
    }

    /**
     * Returns the score of this output node.
     *
     * @return the score of this output node
     */
    public double getScore() {
        return score;
    }

    @Override
    public int compareTo(Result o) {
        return Double.compare(score, o.score);
    }

    @Override
    public String toString() {
        return Objects.toStringHelper(this)
                .add("index", getIndex())
                .add("score", getScore()).toString();
    }

}
