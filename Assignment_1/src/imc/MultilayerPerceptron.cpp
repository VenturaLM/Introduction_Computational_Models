/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2021
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
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	nOfLayers = 0;
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
// Feed all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights()
{
	int a = -1.0, b = 1.0;

	for (auto i = 1; i < nOfLayers; i++)
	{
		for (auto j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (auto k = 0; k < layers[i - 1].nOfNeurons; k++)
			{
				layers[i].neurons[j].w[k] = ((double)rand() / (double)RAND_MAX) * (b - a) + a;
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
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
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
	for (auto i = 1; i < nOfLayers; i++)
	{
		for (auto j = 0; j < layers[i].nOfNeurons; j++)
		{
			double sum = 0.0, net = 0.0;
			for (auto k = 0; k < layers[i - 1].nOfNeurons; k++) // +1 due to the bias.
				sum += (layers[i].neurons[j].w[k] * layers[i - 1].neurons[k].out);
				//sum += (layers[i].neurons[j].w[k] * layers[i - 1].neurons[k - 1].out);

			net = layers[i].neurons[j].w[layers[i - 1].nOfNeurons] + sum;
			layers[i].neurons[j].out = 1.0 / (1.0 + exp(-net)); // Activation function: Sigmoid.
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(double *target)
{
	// MSE = 1 / N * sum( (target[i] - out[i])^2 )
	double sum = 0.0;
	for (auto i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
		sum += pow(target[i] - layers[nOfLayers - 1].neurons[i].out, 2);

	return (1.0 / (double)layers[nOfLayers - 1].nOfNeurons) * sum;
}

// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(double *target)
{
	// delta = -( target[i] - out{i}{j} * g'(sigmoid) )
	for (auto i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++)
		layers[nOfLayers - 1].neurons[i].delta = -(target[i] - layers[nOfLayers - 1].neurons[i].out) * layers[nOfLayers - 1].neurons[i].out * (1.0 - layers[nOfLayers - 1].neurons[i].out);

	for (auto h = nOfLayers - 2; h >= 0; h--)
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

			layers[i].neurons[j].deltaW[layers[i - 1].nOfNeurons] += layers[i].neurons[j].delta; // * 1.0 due to the bias.
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment()
{
	for (auto i = 1; i < nOfLayers; i++)
	{
		for (auto j = 0; j < layers[i].nOfNeurons; j++)
		{
			for (auto k = 0; k < layers[i - 1].nOfNeurons + 1; k++) // The bias is set below.
				layers[i].neurons[j].w[k] = layers[i].neurons[j].w[k] - eta * layers[i].neurons[j].deltaW[k] - mu * (eta * layers[i].neurons[j].lastDeltaW[k]);

			//layers[i].neurons[j].w[0] = layers[i].neurons[j].w[0] - eta * layers[i].neurons[j].deltaW[0] - mu * (eta * layers[i].neurons[j].lastDeltaW[0]);
		}
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
void MultilayerPerceptron::performEpochOnline(double *input, double *target)
{
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				layers[i].neurons[j].deltaW[k] = 0.0;

	feedInputs(input);
	forwardPropagate();
	backpropagateError(target);
	accumulateChange();
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
// Perform an online training for a specific trainDataset
void MultilayerPerceptron::trainOnline(Dataset *trainDataset)
{
	int i;
	for (i = 0; i < trainDataset->nOfPatterns; i++)
	{
		performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i]);
	}
}

// ------------------------------
// Test the network with a dataset and return the MSE
double MultilayerPerceptron::test(Dataset *testDataset)
{
	double sum = 0.0;
	for (auto i = 0; i < testDataset->nOfPatterns; i++)
	{
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		sum += obtainError(testDataset->outputs[i]);
	}

	return (1.0 / (double)testDataset->nOfPatterns) * sum;
}

// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset *testDataset)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers - 1].nOfNeurons;
	double *obtained = new double[numSalidas];

	cout << "Id,Predicted" << endl;

	for (i = 0; i < testDataset->nOfPatterns; i++)
	{

		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(obtained);

		cout << i;

		for (j = 0; j < numSalidas; j++)
			cout << "," << obtained[j];
		cout << endl;
	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOnlineBackPropagation(Dataset *trainDataset, Dataset *testDataset, int epochs, double *errorTrain, double *errorTest)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving;
	double testError = 0;

	int iterWithoutImprovingValidation = 0;
	double validationError = 1;

	Dataset *validationDataset = new Dataset;
	bool validation = false;

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
				validationDataset->outputs[i][j] = trainDataset->inputs[indexes[i]][j];
		}
	}

	// Learning
	do
	{
		// Training:
		trainOnline(trainDataset);
		double trainError = test(trainDataset);

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
			countTrain = epochs;
		}

		countTrain++;

		// Check validation stopping condition and force it
		// BE CAREFUL: in this case, we have to save the last validation error, not the minimum one
		// Apart from this, the way the stopping condition is checked is the same than that
		// applied for the training set

		// Validation:
		if (validation)
		{
			validationError = test(validationDataset);

			if ((validationError - testError) < 0.00001) // AQUI VA TESTERROR O TRAINING ERROR (?)
				iterWithoutImprovingValidation = 0;
			else
				iterWithoutImprovingValidation++;

			if (iterWithoutImprovingValidation == 50)
			{
				cout << "We exit because the validation is not improving!!" << endl;
				restoreWeights();
				countTrain = epochs;
			}
		}

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Validation error: " << validationError << endl;

	} while (countTrain < epochs);

	cout << "\nNETWORK WEIGHTS" << endl;
	cout << "===============";
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

	testError = test(testDataset);
	*errorTest = testError;
	*errorTrain = minTrainError;
}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char *file)
{
	// Object for writing the file
	ofstream f(file);

	if (!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for (int i = 0; i < nOfLayers; i++)
		f << " " << layers[i].nOfNeurons;
	f << endl;

	// Write the weight matrix of every layer
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;
}

// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char *file)
{
	// Object for reading a file
	ifstream f(file);

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
		f >> npl[i];

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for (int i = 1; i < nOfLayers; i++)
		for (int j = 0; j < layers[i].nOfNeurons; j++)
			for (int k = 0; k < layers[i - 1].nOfNeurons + 1; k++)
				f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
