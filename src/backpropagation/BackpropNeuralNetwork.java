
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

import java.util.*;

/**
 * A generalized model of a backpropagation neural network to be trained and used afterwards.
 * 
 * <pre>
 * Some hints for best practices, using literature:
 * 
 * 		* depending on your problem to be solved, use one or two hidden layers (three are rare, 
 * 			four usually do not have advantages over three)
 * 		* one hidden layer (or the first of the hidden layers) should have some more neurons
 * 			than the input layer (around 10%). With many training data sets the number of neurons 
 * 			may be increased - to store more "information"
 * 		* too many neurons in hidden layers: the model does not "learn", 
 * 			it "stores" just the input as weights and increases
 * 			the amount of computation (time, energy), therefore an
 * 			unknown new data to the model will not be recognized
 * 			(the model does not "realize" the principle)
 * 		* not enough neurons: the model is to small to learn "all data"
 * 		* sometimes it makes sense to start with a higher 
 * 			learning rate and reduce it after a while of training
 * 		* learning rates are usually in the range of 0.01 to 0.9
 * 		* high learning rates may miss an optimum or oscillate over it
 * 		* low learning rates may increase computation time a lot, 
 * 			especially in "flat" areas of optimization and/or get stuck in local 
 * 			minima
 * 		* the weights (and biases) are initialized in a random way within this model,
 * 			to avoid some initialization traps (the model uses "symmetry breaking")
 * 
 * References, algorithms and literature: 
 * 
 * 		R. Rojas, Neural Networks, Springer-Verlag, Berlin
 * 		David Kriesel, 2007, A Brief Introduction to Neural Networks, www.dkriesel.com/en/science/neural_networks 
 * 
 * </pre>
 */
public class BackpropNeuralNetwork {
	
	/** the version of <code>BackpropNeuralNetwork</code> */
	public static final String VERSION = "1.0.1";
	/** the default learning rate */
	public static final float DEFAULT_LEARNING_RATE = 0.05f;

	/** the number of inputs */
    private int inputNodeCount;
	/** the number of hidden nodes */
    private int hiddenNodeCount;
	/** the number of outputs */
    private int outputNodeCount;
	/** the weights between input and hidden nodes */
    private double[][] weightsIH;
	/** the weights hidden input and output nodes */
    private double[][] weightsHO;
	/** the outputs of the hidden nodes */
    private double[] hiddenNodeOutputs;
	/** the output nodes */
    private double[] outputs;
	/** the bias of the hidden nodes */
    private double[] biasH;
	/** the bias of the output nodes */
    private double[] biasO;
	/** the errors of the outputs in a training step, used for backpropagation */
    private double[] outputErrors;
	/** the errors of the hidden nodes in a training step, used for backpropagation */
	private double[] hiddenErrors;
	/** training data inputs: N data sets with M inputs each */
	private double[][] inputTrainVectors;
	/** training data expected outputs: N data sets with M outputs each */
	private double[][] outputTrainVectors;
	/** the learning rate */
    private double learningRate;
	/** a <code>Random</code> for "symmetry breaking" */
    private Random random;
	/** the error of the last training */
    private double error;

	/**
	 * Constructs an empty default model, a setup may be needed afterwards.
	 * This default model has one hidden layer with about 10% neurons than the input layer,
	 * the default learning rate and weights/biases initialized using random numbers<br>
	 * in the ranges [0.1 .. 0.5]. 
	 *  
     * @param inputNodeCount		the number of inputs
     * @param hiddenNodeCount		the number of hidden nodes
     * @param outputNodeCount		the number of outputs
     */
    public BackpropNeuralNetwork(int inputNodeCount, int hiddenNodeCount, int outputNodeCount) {
    	
    	this(inputNodeCount, hiddenNodeCount, outputNodeCount, DEFAULT_LEARNING_RATE, null);
    }

	/**
	 * Constructs an empty default model, a setup may be needed afterwards.
	 * This default model has one hidden layer and weights/biases initialized using random numbers<br>
	 * in the ranges [0.1 .. 0.5].  
	 * 
     * @param inputNodeCount		the number of inputs
     * @param hiddenNodeCount		the number of hidden nodes
     * @param outputNodeCount		the number of outputs
     * @param learningRate			the learning rate, usually in the range of 0.01 to 0.9
     * @param random				a <code>Random</code>, may be <code>null</code> for a default <code>Random</code>
     */
    public BackpropNeuralNetwork(int inputNodeCount, int hiddenNodeCount, int outputNodeCount, 
    		double learningRate, Random random) {
    	
        this.inputNodeCount = inputNodeCount;
        this.hiddenNodeCount = hiddenNodeCount;
        this.outputNodeCount = outputNodeCount;
        this.learningRate = learningRate;
        if (random == null) {
        	random = new Random(42);
		}
        this.random = random;
        // allocate memory for all variables
        weightsIH = new double[inputNodeCount][hiddenNodeCount];
        weightsHO = new double[hiddenNodeCount][outputNodeCount];
        hiddenNodeOutputs = new double[hiddenNodeCount];
        outputs = new double[outputNodeCount];
        biasH = new double[hiddenNodeCount];
        biasO = new double[outputNodeCount];
        outputErrors = new double[outputNodeCount];
        hiddenErrors = new double[hiddenNodeCount];
        // randomly initialize weights and biases: using "symmetry breaking"
        for (int i = 0; i < inputNodeCount; i++) {
            for (int j = 0; j < hiddenNodeCount; j++) {
                weightsIH[i][j] = nextRandom();
            }
        }
        for (int i = 0; i < hiddenNodeCount; i++) {
            for (int j = 0; j < outputNodeCount; j++) {
                weightsHO[i][j] = nextRandom();
            }
            biasH[i] = nextRandom();
        }
        for (int i = 0; i < outputNodeCount; i++) {
            biasO[i] = nextRandom();
        }
    }
	
	/**
	 * Create the input and output vectors for training for a training data set.
	 * 
	 * @param trainingData		the training data: input vector 0, desired output vector 0, 
	 * 							input vector 1, desired output vector 1, and so on
	 */
	public void createInOutVectors(double trainingData[][]) {
		
		int inputLength = trainingData[0].length;
		int outputLength = trainingData[1].length;
		int dataSetCount = trainingData.length / 2;
		inputTrainVectors = new double[dataSetCount][inputLength];
		outputTrainVectors = new double[dataSetCount][outputLength];
		for (int i = 0; i < dataSetCount; i++) {
			double[] inputVec = trainingData[i * 2];
			double[] vector = inputTrainVectors[i];
			for (int j = 0; j < inputVec.length; j++) {
				vector[j] = inputVec[j];
			}
			double[] outputVec = trainingData[i * 2 + 1];
			vector = outputTrainVectors[i];
			for (int j = 0; j < outputVec.length; j++) {
				vector[j] = outputVec[j];
			}
		}
	}

    /**
     * Compute the outputs (output vector) for an input vector.
     * Usually the neural network has been trained before.
     * 
     * @param inputs		the inputs (input vector)
     * @return the ouputs
     */
    public double[] forwardPass(double[] inputs) {
    	
        // calculate the output of the hidden layer
        for (int i = 0; i < hiddenNodeCount; i++) {
            double activation = 0;
            for (int j = 0; j < inputNodeCount; j++) {
                activation += inputs[j] * weightsIH[j][i];
            }
            activation += biasH[i];
            hiddenNodeOutputs[i] = Sigmoid.sigmoid(activation);
        }
        // calculate the output of the output layer
        for (int i = 0; i < outputNodeCount; i++) {
            double activation = 0;
            for (int j = 0; j < hiddenNodeCount; j++) {
                activation += hiddenNodeOutputs[j] * weightsHO[j][i];
            }
            activation += biasO[i];
            outputs[i] = Sigmoid.sigmoid(activation);
        }
        return outputs;
    }

	/**
	 * @return the error of the last training 
	 */
	public double getError() {
		
		return error;
	}
	
	/**
	 * @return the errors of the hidden nodes of the last training 
	 */
	public double[] getHiddenErrors() {
		
		return hiddenErrors;
	}

	/**
	 * @return the errors of the output nodes of the last training 
	 */
    public double[] getOutputErrors() {
    	
		return outputErrors;
	}

	/**
	 * @return the learningRate
	 */
	public double getLearningRate() {
		
		return learningRate;
	}

    /**
     * @return the next random number to initialize weights and biases
     */
    protected double nextRandom() {
    	
		return random.nextDouble(0.1, 0.5);
    }

	/**
	 * Set the learning rate. It may be useful for some problems to
	 * changes the learning rate according to the error of the current training
	 * state.
	 * 
	 * @param learningRate 			the learningRate to set
	 */
	public void setLearningRate(double learningRate) {
		
		this.learningRate = learningRate;
	}

    /**
     * Train the network one step using an input vector and the current learning rate.
     * 
     * @param inputs			the input vector to train
     * @param desiredOutputs	the desired output vector
     */
    public void train(double[] inputs, double[] desiredOutputs) {
    	
        // forward pass goes first to compute the outputs and outputs of the hidden layers
    	forwardPass(inputs);
        // backpropagation, first compute output error(s)
        for (int i = 0; i < outputNodeCount; i++) {
        	double output = outputs[i];
        	// sigmoid derivative: output * (1 - output)
            outputErrors[i] = (desiredOutputs[i] - output) * output * (1 - output);
        }
        // backpropagate the output errors to the hidden nodes
        for (int i = 0; i < hiddenNodeCount; i++) {
            error = 0;
            for (int j = 0; j < outputNodeCount; j++) {
                error += outputErrors[j] * weightsHO[i][j];
            }
        	// sigmoid derivative:  hiddenOutputs[i] * (1 - hiddenOutputs[i])
            hiddenErrors[i] = error * hiddenNodeOutputs[i] * (1 - hiddenNodeOutputs[i]);
        }
        // update weights and biases: input nodes to hidden nodes and hidden to output nodes
        for (int i = 0; i < inputNodeCount; i++) {
            for (int j = 0; j < hiddenNodeCount; j++) {
                weightsIH[i][j] += learningRate * hiddenErrors[j] * inputs[i];
            }
        }
        for (int i = 0; i < hiddenNodeCount; i++) {
            for (int j = 0; j < outputNodeCount; j++) {
                weightsHO[i][j] += learningRate * outputErrors[j] * hiddenNodeOutputs[i];
            }
            biasH[i] += learningRate * hiddenErrors[i];
        }
        for (int i = 0; i < outputNodeCount; i++) {
            biasO[i] += learningRate * outputErrors[i];
        }
    }

	/**
	 * Train the model with a number data set, using one or more steps in the direction of the given data set.
	 * 
     * @param inputVector				the input vector to train
     * @param desiredOutputVector		the desired output vector
	 * @param trainigSteps				the number of steps to train the model with this input/output
	 */
	public void trainDataSet(double[] inputVector, double[] desiredOutputVector, int trainigSteps) {
		
		for (int i = 0; i < trainigSteps; i++) {
			train(inputVector, desiredOutputVector);
		}
	}

	/**
	 * Train the model with a number data sets, choosing randomly different data.
	 * This method call may be repeated. 
	 * Callers can change the learning rate or use other training data sets 
	 * by calling createInOutVectors() before.
	 * 
	 * @param trainings				the number of data sets to be trained, usually much more than the number of 
	 * 								data sets - they will be repeated depending on the probability of Random		
	 * @param trainigStepsPerSet	the number of training steps per set: if greater than one, the model will step
	 * 								trainigStepsPerSet for each data set before taking the next data set
	 */
	public void trainRandom(int trainings, int trainigStepsPerSet) {
		
		trainRandom(random, trainings, trainigStepsPerSet);
	}

	/**
	 * Train the model with a number data sets, choosing randomly different data.
	 * This method call may be repeated. 
	 * Callers can change the learning rate or use other training data sets 
	 * by calling createInOutVectors() before.
	 * 
	 * @param random				a random number generator to change the order of training or null for a default
	 * @param trainings				the number of data sets to be trained, usually much more than the number of 
	 * 								data sets - they will be repeated depending on the probability of Random		
	 * @param trainigStepsPerSet	the number of training steps per set: if greater than one, the model will step
	 * 								trainigStepsPerSet for each data set before taking the next data set
	 */
	public void trainRandom(Random random, int trainings, int trainigStepsPerSet) {
		
		random = random == null ? new Random() : random;
		for (int i = 0; i < trainings; i++) {
			int dataSetIndex = random.nextInt(inputTrainVectors.length);
			double[] inputVector = inputTrainVectors[dataSetIndex];
			double[] outputVector = outputTrainVectors[dataSetIndex];
			trainDataSet(inputVector, outputVector, trainigStepsPerSet);
		}
	}
}



