
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

import java.util.*;

import backpropagation.*;

/**
 * Neural network: Backpropagation tester using a simple XOR gate with 
 * one hidden layer containing two neurons. A XOR is used in many simple
 * backpropagation tests, due to a small number of neurons and a
 * somewhat "complicated" behavior.
 * 
 * Note: a training data set is defined at the end of this file.
 */
public class TesterXOR {
	
	/**
	 * Construct the tester, model, test data, run some training and 
	 * display the output of each training.
	 */
	public TesterXOR() {
		
		// create the network
		BackpropNeuralNetwork neuralNetwork = new BackpropNeuralNetwork(2, 2, 1);
		neuralNetwork.setLearningRate(BackpropNeuralNetwork.DEFAULT_LEARNING_RATE);		// note: DEFAULT_LEARNING_RATE is already set
		neuralNetwork.createInOutVectors(trainingData);
		// train the network using the truth table of a XOR gate
		neuralNetwork.createInOutVectors(trainingData);
		// four data sets (input vectors, so each data set will be trained 10 times)
		int trainigStepsPerSet = 10;
		int trainings = 8000;
		// display the result of the training
		neuralNetwork.trainRandom(trainings, trainigStepsPerSet);			
		System.out.println("\n***** Training (default learning rate): " + trainings + " trainings (each " 
				+ trainigStepsPerSet + " steps) *****");
		displayAfterTraining(neuralNetwork);
		// do it again using more training
		trainings = 20000;
		neuralNetwork = new BackpropNeuralNetwork(2, 2, 1);
		neuralNetwork.createInOutVectors(trainingData);
		neuralNetwork.trainRandom(trainings, trainigStepsPerSet);			
		System.out.println("\n***** Training (default learning rate): " + trainings + " trainings (each " 
				+ trainigStepsPerSet + " steps) *****");
		displayAfterTraining(neuralNetwork);
		// do it again using a higher learning rate and less training
		trainings = 2000;
		neuralNetwork = new BackpropNeuralNetwork(2, 2, 1);
		double learningRate = 0.3;			// high due to XOR behavior and few inputs
		neuralNetwork.setLearningRate(learningRate);		
		neuralNetwork.createInOutVectors(trainingData);
		neuralNetwork.trainRandom(trainings, trainigStepsPerSet);			
		System.out.println("\n***** Training (learning rate " + learningRate 
				+ "): " + trainings + " trainings (each " 
				+ trainigStepsPerSet + " steps) *****");
		displayAfterTraining(neuralNetwork);
	}

	/**
	 * Displays the results after a training.
	 * 
	 * @param neuralNetwork				the network
	 */
	private void displayAfterTraining(BackpropNeuralNetwork neuralNetwork) {
		
		for (int i = 0; i < trainingData.length; i += 2) {
			double[] inputs = trainingData[i];
			System.out.println("\nInput: " + Arrays.toString(inputs));
			double[] outputs = neuralNetwork.forwardPass(inputs);
			System.out.println("Output: " + Arrays.toString(outputs));
			System.out.println("Desired output: " + Arrays.toString(trainingData[i + 1]));
		}
	}

	/**
	 * Runs the test.
	 * 
	 * @param args		the arguments
	 */
	public static void main(String[] args) {

		new TesterXOR();
	}

	/** data set for training (a truth table): {input vector: two inputs}, {output vector: one output} */
	public static final double[][] trainingData = {
			{0, 0}, {0},
			{1, 0}, {1},
			{0, 1}, {1}, 
			{1, 1}, {0},
	};
}
