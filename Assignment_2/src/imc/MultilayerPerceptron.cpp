/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"
#include "util.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>

using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Obtain an integer random number in the range [Low,High]
int randomInt(int Low, int High)
{
	return rand() % High + Low;
}

// ------------------------------
// Obtain a real random number in the range [Low,High]
double randomDouble(double Low, double High)
{
	return ((double)rand() / (double)RAND_MAX) * (High - Low) + Low;
}

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	nOfLayers = 0;
	nOfTrainingPatterns = 0;
	layers = nullptr;
	eta = 0.1;
	mu = 0.9;
	validationRatio = 0.0;
	decrementFactor = 1.0;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[])
{
	// Create the layers.
	nOfLayers = nl;
	nOfTrainingPatterns = 0;
	layers = new Layer[nOfLayers];

	//-------
	for (auto i = 0; i < nOfLayers; i++)
	{
		layers[i].nOfNeurons = npl[i];
		layers[i].neurons = new Neuron[npl[i]];

		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			layers[i].neurons[j].out = 1.0;
			layers[i].neurons[j].delta = 0.0;

			if (i != 0)
			{ // Hidden and last layers.
				int nOfWeight = layers[i - 1].nOfNeurons + 1;
				layers[i].neurons[j].w = new double[nOfWeight];
				layers[i].neurons[j].deltaW = new double[nOfWeight];
				layers[i].neurons[j].lastDeltaW = new double[nOfWeight];
				layers[i].neurons[j].wCopy = new double[nOfWeight];
			}
		}
	}

	//-------
	// Check if layers is NULL.
	for (auto i = 0; i < nOfLayers; i++)
		if (layers == nullptr)
			perror("Value of layers: nullptr.");

	return 1;
}

// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron()
{
	freeMemory();
}

// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory()
{
	for (auto i = nOfLayers - 1; i >= 0; i--)
	{
		if (i != 0) // Just free the elements of the hidden and last layers.
		{
			for (auto j = layers[i].nOfNeurons - 1; j >= 0; j--)
			{
				// Deallocation for each component of each neuron.
				delete[] layers[i].neurons[j].w;
				delete layers[i].neurons[j].deltaW;
				delete layers[i].neurons[j].lastDeltaW;
				delete[] layers[i].neurons[j].wCopy;
			}
		}
		delete[] layers[i].neurons;
	}
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights()
{
	for (auto i = 1; i < nOfLayers; i++)
	{
		for (auto j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (auto k = 0; k < layers[i - 1].nOfNeurons; k++)
			{
				layers[i].neurons[j].w[k] = randomDouble(-1, 1);
				layers[i].neurons[j].deltaW[k] = 0.0;
				layers[i].neurons[j].lastDeltaW[k] = 0.0;
				layers[i].neurons[j].wCopy[k] = 0.0;
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double *input)
{
	for (auto i = 0; i < layers[0].nOfNeurons; i++)
		layers[0].neurons[i].out = input[i];
}

// ------------------------------
// Get the outputs predicted by the network (out vector of the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double *output)
{
	for (auto i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
		output[i] = layers[nOfLayers - 1].neurons[i].out;
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights()
{
	for (auto i = 1; i < nOfLayers; i++)
		for (auto j = 0; j < layers[i].nOfNeurons; j++)
			for (auto k = 0; k < layers[i - 1].nOfNeurons; k++)
				layers[i].neurons[j].wCopy[k] = layers[i].neurons[j].w[k];
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights()
{
	for (auto i = 1; i < nOfLayers; i++)
		for (auto j = 0; j < layers[i].nOfNeurons; j++)
			for (auto k = 0; k < layers[i - 1].nOfNeurons; k++)
				layers[i].neurons[j].w[k] = layers[i].neurons[j].wCopy[k];
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate()
{
	if (!online)
	{
		for (auto i = 1; i < nOfLayers; i++)
		{
			for (auto j = 0; j < layers[i].nOfNeurons; j++)
			{
				double sum = 0.0, net = 0.0;
				for (auto k = 0; k < layers[i - 1].nOfNeurons; k++) // From 0 to nOfNeurons due to the bias.
					sum += (layers[i].neurons[j].w[k] * layers[i - 1].neurons[k].out);

				net = layers[i].neurons[j].w[layers[i - 1].nOfNeurons] + sum;
				layers[i].neurons[j].out = 1.0 / (1.0 + exp(-net)); // Activation function: Sigmoid.
			}
		}
	}

	else if (online)
	{
		for (auto i = 1; i < nOfLayers - 1; i++) // Until penultimate.
		{
			for (auto j = 0; j < layers[i].nOfNeurons; j++)
			{
				double sum = 0.0, net = 0.0;
				for (auto k = 0; k < layers[i - 1].nOfNeurons; k++) // From 0 to nOfNeurons due to the bias.
					sum += (layers[i].neurons[j].w[k] * layers[i - 1].neurons[k].out);

				net = layers[i].neurons[j].w[layers[i - 1].nOfNeurons] + sum;
				layers[i].neurons[j].out = 1.0 / (1.0 + exp(-net)); // Activation function: Sigmoid.
			}
		}

		double *net = new double[layers[nOfLayers - 1].nOfNeurons];
		for (auto i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
			for (auto j = 0; j < layers[nOfLayers - 1].nOfNeurons; j++)
				net[i] += layers[nOfLayers - 1].neurons[i].w[j] * layers[nOfLayers - 2].neurons[i].out;

		double net_sum = 0.0;
		for (auto i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
			net_sum += exp(net[i]);

		for (auto i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
			layers[nOfLayers - 1].neurons[i].out = exp(net[i]) / net_sum; // Softmax
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::obtainError(double *target, int errorFunction)
{
	double sum = 0.0;

	if (errorFunction) // Cross Entropy.
	{
		for (auto i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
			sum += target[i] * log(layers[nOfLayers - 1].neurons[i].out);
	}

	else if (!errorFunction) // MSE
	{
		for (auto i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
			sum += pow(target[i] - layers[nOfLayers - 1].neurons[i].out, 2);
	}

	return (1.0 / (double)layers[nOfLayers - 1].nOfNeurons) * sum;
}

// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::backpropagateError(double *target, int errorFunction)
{
	if (!outputFunction) // Sigmoid.
	{
		if (!errorFunction) // MSE.
		{
			for (auto i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
				layers[nOfLayers - 1].neurons[i].delta = -(target[i] - layers[nOfLayers - 1].neurons[i].out) * layers[nOfLayers - 1].neurons[i].out * (1.0 - layers[nOfLayers - 1].neurons[i].out);
		}

		else if (errorFunction) // Cross Entropy.
		{
			for (auto i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
				layers[nOfLayers - 1].neurons[i].delta = -(target[i] / layers[nOfLayers - 1].neurons[i].out) * layers[nOfLayers - 1].neurons[i].out * (1.0 - layers[nOfLayers - 1].neurons[i].out);
		}
	}

	else if (outputFunction) // Softmax.
	{
		double sum = 0.0;
		int I = 0;
		if (!errorFunction) // MSE.
		{
			for (auto i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
			{
				for (auto j = 0; j < layers[nOfLayers - 1].nOfNeurons; j++)
				{
					I = (i == j) ? 1 : 0;
					sum += (target[j] - layers[nOfLayers - 1].neurons[j].out) * layers[nOfLayers - 1].neurons[i].out * (I - layers[nOfLayers - 1].neurons[j].out);
				}

				layers[nOfLayers - 1].neurons[i].delta = -sum;
			}
		}

		else if (errorFunction) // Cross Entropy.
		{
			for (auto i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
			{
				for (auto j = 0; j < layers[nOfLayers - 1].nOfNeurons; j++)
				{
					I = (i == j) ? 1 : 0;
					sum += (target[j] / layers[nOfLayers - 1].neurons[j].out) * layers[nOfLayers - 1].neurons[i].out * (I - layers[nOfLayers - 1].neurons[j].out);
				}

				layers[nOfLayers - 1].neurons[i].delta = -sum;
			}
		}
	}

	for (auto h = nOfLayers - 2; h >= 0; h--) // Hidden layers.
	{
		for (auto j = 0; j < layers[h].nOfNeurons; j++)
		{
			double sum = 0.0;
			for (auto k = 0; k < layers[h + 1].nOfNeurons; k++)
				sum += layers[h + 1].neurons[k].w[j] * layers[h + 1].neurons[k].delta;

			layers[h].neurons[j].delta = sum * layers[h].neurons[j].out * (1 - layers[h].neurons[j].out);
		}
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange()
{
	for (auto i = 1; i < nOfLayers; i++)
	{
		for (auto j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (auto k = 0; k < layers[i - 1].nOfNeurons; k++)
				layers[i].neurons[j].deltaW[k] = layers[i].neurons[j].deltaW[k] + layers[i].neurons[j].delta * layers[i - 1].neurons[k].out;

			layers[i].neurons[j].deltaW[layers[i - 1].nOfNeurons] += layers[i].neurons[j].delta;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment()
{
	if (online)
	{
		for (auto i = 1; i < nOfLayers; i++)
			for (auto j = 0; j < layers[i].nOfNeurons; j++)
				for (auto k = 0; k < layers[i - 1].nOfNeurons + 1; k++) // +1 for the bias.
					layers[i].neurons[j].w[k] = layers[i].neurons[j].w[k] - eta * layers[i].neurons[j].deltaW[k] - mu * (eta * layers[i].neurons[j].lastDeltaW[k]);
	}

	else if (!online)
	{
		for (auto i = 1; i < nOfLayers; i++)
			for (auto j = 0; j < layers[i].nOfNeurons; j++)
				for (auto k = 0; k < layers[i - 1].nOfNeurons + 1; k++) // +1 for the bias.
					layers[i].neurons[j].w[k] = layers[i].neurons[j].w[k] - (eta * layers[i].neurons[j].deltaW[k] / nOfTrainingPatterns) - (mu * eta * layers[i].neurons[j].lastDeltaW[k] / nOfTrainingPatterns);
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork()
{
	for (int i = 0; i < nOfLayers; i++)
	{
		cout << "\nLayer " << i << endl;
		cout << "----------" << endl;
		for (int j = 0; j < layers[i].nOfNeurons; j++)
		{
			cout << "[out = " << layers[i].neurons[j].out;

			if (i != 0)
				for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
					cout << "\t\t w" << k << " = " << layers[i].neurons[j].w[k];

			printf("]\n");
		}
	}
	printf("\n");
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// The step of adjusting the weights must be performed only in the online case
// If the algorithm is offline, the weightAdjustment must be performed in the "train" function
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::performEpoch(double *input, double *target, int errorFunction)
{
	if (online)
	{
		for (int i = 1; i < nOfLayers; i++)
			for (int j = 0; j < layers[i].nOfNeurons; j++)
				for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
					layers[i].neurons[j].deltaW[k] = 0.0;
	}

	feedInputs(input);
	forwardPropagate();
	backpropagateError(target, errorFunction);
	accumulateChange();

	if (online)
		weightAdjustment();
}

// ------------------------------
// Read a dataset from a file name and return it
Dataset *MultilayerPerceptron::readData(const char *fileName)
{

	ifstream myFile(fileName); // Create an input stream

	if (!myFile.is_open())
	{
		cout << "ERROR: I cannot open the file " << fileName << endl;
		return NULL;
	}

	Dataset *dataset = new Dataset;
	if (dataset == NULL)
		return NULL;

	string line;
	int i, j;

	if (myFile.good())
	{
		getline(myFile, line); // Read a line
		istringstream iss(line);
		iss >> dataset->nOfInputs;
		iss >> dataset->nOfOutputs;
		iss >> dataset->nOfPatterns;
	}
	dataset->inputs = new double *[dataset->nOfPatterns];
	dataset->outputs = new double *[dataset->nOfPatterns];

	for (i = 0; i < dataset->nOfPatterns; i++)
	{
		dataset->inputs[i] = new double[dataset->nOfInputs];
		dataset->outputs[i] = new double[dataset->nOfOutputs];
	}

	i = 0;
	while (myFile.good())
	{
		getline(myFile, line); // Read a line
		if (!line.empty())
		{
			istringstream iss(line);
			for (j = 0; j < dataset->nOfInputs; j++)
			{
				double value;
				iss >> value;
				if (!iss)
					return NULL;
				dataset->inputs[i][j] = value;
			}
			for (j = 0; j < dataset->nOfOutputs; j++)
			{
				double value;
				iss >> value;
				if (!iss)
					return NULL;
				dataset->outputs[i][j] = value;
			}
			i++;
		}
	}

	myFile.close();

	return dataset;
}

// ------------------------------
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset *trainDataset, int errorFunction)
{
	if (!online)
	{
		for (int i = 1; i < nOfLayers; i++)
			for (int j = 0; j < layers[i].nOfNeurons; j++)
				for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
					layers[i].neurons[j].deltaW[k] = 0.0;
	}

	for (auto i = 0; i < trainDataset->nOfPatterns; i++)
		performEpoch(trainDataset->inputs[i], trainDataset->outputs[i], errorFunction);

	if (!online)
		weightAdjustment();
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset *dataset, int errorFunction)
{
	double sum = 0.0;
	for (auto i = 0; i < dataset->nOfPatterns; i++)
	{
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		sum += obtainError(dataset->outputs[i], errorFunction);
	}

	if (errorFunction) // Cross Entropy.
		return -(1.0 / (double)dataset->nOfPatterns) * sum;
	else if (!errorFunction) // MSE.
		return (1.0 / (double)dataset->nOfPatterns) * sum;
}

// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset *dataset)
{
	double *confusionMatrix = new double[dataset->nOfOutputs];
	int tp = 0;

	for (auto i = 0; i < dataset->nOfPatterns; i++)
	{
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(confusionMatrix);

		int classification_index = 0;
		int obtained_index = 0;
		for (auto j = 0; j < dataset->nOfOutputs; j++)
		{
			if (dataset->outputs[i][j] > dataset->outputs[i][classification_index])
				classification_index = j;

			if (confusionMatrix[j] > confusionMatrix[obtained_index])
				obtained_index = j;
		}

		if (classification_index == obtained_index)
			tp += 1;
	}

	return 100.0 * (double)tp / (double)dataset->nOfPatterns;
}

// ------------------------------
// Optional Kaggle: Obtain the predicted outputs for a dataset
void MultilayerPerceptron::predict(Dataset *dataset)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers - 1].nOfNeurons;
	double *salidas = new double[numSalidas];

	cout << "Id,Category" << endl;

	for (i = 0; i < dataset->nOfPatterns; i++)
	{

		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(salidas);

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;

		cout << i << "," << maxIndex << endl;
	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
// Both training and test CCRs should be obtained and stored in ccrTrain and ccrTest
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::runBackPropagation(Dataset *trainDataset, Dataset *testDataset, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int errorFunction)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving = 0;
	nOfTrainingPatterns = trainDataset->nOfPatterns;

	Dataset *validationDataset = new Dataset;
	bool validation = false;
	double validationError = 0, previousValidationError = 0;
	int iterWithoutImprovingValidation = 0;

	// Generate validation data
	if (validationRatio > 0.0 && validationRatio < 1.0)
	{
		validation = true;
		int min = 0, max = trainDataset->nOfPatterns - 1, howMany = validationRatio * trainDataset->nOfPatterns;
		if (howMany == 0)
			howMany = 1; // At least take 1 element in the integerRandomVectorWithoutRepeating()

		int *indexes = integerRandomVectorWithoutRepeating(min, max, howMany);

		if (validationDataset == nullptr)
			perror("Error: Validation Dataset no allocated.");
		validationDataset->nOfInputs = trainDataset->nOfInputs;
		validationDataset->nOfOutputs = trainDataset->nOfOutputs;
		validationDataset->nOfPatterns = howMany;
		validationDataset->inputs = new double *[howMany];
		validationDataset->outputs = new double *[howMany];

		for (auto i = 0; i < validationDataset->nOfPatterns; i++)
		{
			validationDataset->inputs[i] = new double[validationDataset->nOfInputs];
			validationDataset->outputs[i] = new double[validationDataset->nOfOutputs];
		}

		for (auto i = 0; i < validationDataset->nOfPatterns; i++)
		{
			for (auto j = 0; j < validationDataset->nOfInputs; j++)
				validationDataset->inputs[i][j] = trainDataset->inputs[indexes[i]][j];

			for (auto j = 0; j < validationDataset->nOfOutputs; j++)
				validationDataset->outputs[i][j] = trainDataset->outputs[indexes[i]][j];
		}
	}

	// Learning
	do
	{

		train(trainDataset, errorFunction);

		double trainError = test(trainDataset, errorFunction);
		if (countTrain == 0 || trainError < minTrainError)
		{
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if ((trainError - minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if (iterWithoutImproving == 50)
		{
			cout << "We exit because the training is not improving!!" << endl;
			restoreWeights();
			countTrain = maxiter;
		}

		countTrain++;

		if (validation)
		{
			if (previousValidationError == 0)
				previousValidationError = 999999999.9999999999;
			else
				previousValidationError = validationError;
			validationError = test(validationDataset, errorFunction);
			if (validationError < previousValidationError)
				iterWithoutImprovingValidation = 0;
			else if ((validationError - previousValidationError) < 0.00001)
				iterWithoutImprovingValidation = 0;
			else
				iterWithoutImprovingValidation++;
			if (iterWithoutImprovingValidation == 50)
			{
				cout << "We exit because validation is not improving!!" << endl;
				restoreWeights();
				countTrain = maxiter;
			}
		}

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Validation error: " << validationError << endl;

	} while (countTrain < maxiter);

	if ((iterWithoutImprovingValidation != 50) && (iterWithoutImproving != 50))
		restoreWeights();

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for (int i = 0; i < testDataset->nOfPatterns; i++)
	{
		double *prediction = new double[testDataset->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for (int j = 0; j < testDataset->nOfOutputs; j++)
			cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;
	}

	*errorTest = test(testDataset, errorFunction);
	*errorTrain = minTrainError;
	*ccrTest = testClassification(testDataset);
	*ccrTrain = testClassification(trainDataset);
}

// -------------------------
// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char *fileName)
{
	// Object for writing the file
	ofstream f(fileName);

	if (!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for (int i = 0; i < nOfLayers; i++)
	{
		f << " " << layers[i].nOfNeurons;
	}
	f << " " << outputFunction;
	f << endl;

	// Write the weight matrix of every layer
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				if (layers[i].neurons[j].w != NULL)
					f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;
}

// -----------------------
// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char *fileName)
{
	// Object for reading a file
	ifstream f(fileName);

	if (!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for (int i = 0; i < nl; i++)
	{
		f >> npl[i];
	}
	f >> outputFunction;

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				if (!(outputFunction == 1 && (i == (nOfLayers - 1)) && (k == (layers[i].nOfNeurons - 1))))
					f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
