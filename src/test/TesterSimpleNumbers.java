
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
 * One of the simplest neural networks: Backpropagation tester using two 
 * input numbers to train the network for a third number as output/result.
 */
public class TesterSimpleNumbers {
	
	/**
	 * Runs <code>TesterSimpleNumbers</code>.
	 * 
	 * @param args		the arguments
	 */
	public static void main(String[] args) {

		int inputNodeCount = 2;
		int hiddenNodeCount = 3;
		int outputNodeCount = 1;
		// create the network
		BackpropNeuralNetwork neuralNetwork = new BackpropNeuralNetwork(inputNodeCount, hiddenNodeCount, outputNodeCount);
		// train the network
		double[] input = {0.1, 0.4};
		double[] desiredOutput = {0.7};
		int trainigSteps = 200;
		for (int trainingStep = 0; trainingStep < trainigSteps; trainingStep++) {
			neuralNetwork.train(input, desiredOutput);
		}
		// display the result of the training
		double[] output = neuralNetwork.forwardPass(input);
		System.out.println("\nTraining with " + trainigSteps + " steps for: " + Arrays.toString(desiredOutput) + " ->\n");
		System.out.println("Input: " + Arrays.toString(input));
		System.out.println("Output: " + Arrays.toString(output));
	}
}
