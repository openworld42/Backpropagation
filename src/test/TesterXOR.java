
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
package test;

import backpropagation.*;

/**
 * Neural network: Backpropagation tester using a simple XOR gate with 
 * one hidden layer containing two neurons.
 * 
 * Note: a training data set is defined at the end of this file.
 */
public class TesterXOR {
	
	/**
	 * Construct the tester, model, test data, run the training and 
	 * display some output.
	 * 
	 * @param args
	 */
	public TesterXOR(String[] args) {
		
		// create the network
		BackpropNeuralNetwork neuralNetwork = new BackpropNeuralNetwork(2, 1, 1);
		neuralNetwork.setLearningRate(BackpropNeuralNetwork.DEFAULT_LEARNING_RATE);		// note: DEFAULT_LEARNING_RATE is already set
		neuralNetwork.createInOutVectors(trainingData);
		// train the network using the truth table of a XOR gate
		neuralNetwork.createInOutVectors(trainingData);
		neuralNetwork.trainRandom(40, 1);						// four data sets, so each data set is trained about 10 times

	
		// TODO output 
	
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		new TesterXOR(args);
	}

	// data set for training (truth table): {input vector: two inputs}, {output vector: one output}
	public static final int[][] trainingData = {
			{0, 0}, {0},
			{1, 0}, {1},
			{0, 1}, {1}, 
			{1, 1}, {0},
	};
}
