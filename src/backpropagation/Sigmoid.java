
/**
 * Copyright 2023 Heinz Silberbauer
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     https://www.apache.org/licenses/LICENSE-2.0
 *     
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package backpropagation;

/**
 * Fast table lookup implementation for the sigmoid function, speeding up computation time.
 */
public class Sigmoid {

    private static final int TABLE_SIZE = 10000;
    private static final double TABLE_MIN = -6.0;
    private static final double TABLE_MAX = 6.0;
    private static final double[] SIGMOID_TABLE = new double[TABLE_SIZE];
    private static final double TABLE_STEP = (TABLE_MAX - TABLE_MIN) / (TABLE_SIZE - 1);

    static {
        for (int i = 0; i < TABLE_SIZE; i++) {
            double x = TABLE_MIN + i * TABLE_STEP;
            SIGMOID_TABLE[i] = 1.0 / (1.0 + Math.exp(-x));
        }
    }

    /**
     * Returns the sigmoid function result using a table lookup.
     * 
     * @param x		the <code>x</code> of <code>sigmoid(x)</code>sigmoid(x)
     * @return the sigmoid function result
     */
    public static double sigmoid(double x) {
    	
        if (x <= TABLE_MIN) {
            return SIGMOID_TABLE[0];
        } else if (x >= TABLE_MAX) {
            return SIGMOID_TABLE[TABLE_SIZE - 1];
        } else {
            return SIGMOID_TABLE[(int) ((x - TABLE_MIN) / TABLE_STEP)];
        }
    }
}
