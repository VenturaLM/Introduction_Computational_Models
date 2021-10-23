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
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	this->nOfLayers = 0;
	this->layers = nullptr;
	this->eta = 0.1;
	this->mu = 0.9;
	this->validationRatio = 0.0;
	this->decrementFactor = 1.0;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[])
{
	// Create the layers.
	this->nOfLayers = nl;
	this->layers = new Layer[this->nOfLayers];

	//-------
	for (auto i = 0; i < this->nOfLayers - 1; i++)
	{				//TODO: revisar los indices, porque si tenemos mas de una hidden layer, esto no va a funcionar.
		if (i == 0) // First layer.
		{
			this->layers[i].nOfNeurons = npl[0];
			this->layers[i].neurons = new Neuron[npl[0]];
		}
		else
		{
			if (i == nl - 1) // Last layer.
			{
				this->layers[i].nOfNeurons = npl[2];
				this->layers[i].neurons = new Neuron[npl[2]];
			}
			else // Hidden layers.
			{
				this->layers[i].nOfNeurons = npl[1]; // The reason of npl[1] and not npl[i] is that all hidden layers have the same amount of neurons.
				this->layers[i].neurons = new Neuron[npl[1]];
			}
		}

		for (int j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			// First layer.
			this->layers[i].neurons[j].out = 0.0;
			this->layers[i].neurons[j].delta = 1.0;

			if (i != 0) // Hidden and last layers.
			{
				int nOfWeight = this->layers[i - 1].nOfNeurons + 1;
				this->layers[i].neurons[j].w = new double[nOfWeight];
				this->layers[i].neurons[j].deltaW = new double[nOfWeight];
				this->layers[i].neurons[j].lastDeltaW = new double[nOfWeight];
				this->layers[i].neurons[j].wCopy = new double[nOfWeight];
			}
			else
			{
				this->layers[i].neurons[j].w = nullptr;
				this->layers[i].neurons[j].deltaW = nullptr;
				this->layers[i].neurons[j].lastDeltaW = nullptr;
				this->layers[i].neurons[j].wCopy = nullptr;
			}
		}
	}
	//-------
	// Check if layers is NULL.
	for (auto i = 0; i < this->nOfLayers; i++)
		if (this->layers == nullptr)
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
	for (auto i = this->nOfLayers - 1; i >= 0; i--)
	{
		if (i != 0) // Just free the elements of the hidden and last layers.
		{
			for (auto j = this->layers[i].nOfNeurons - 1; j >= 0; j--)
			{
				// Deallocation for each component of each neuron.
				delete[] this->layers[i].neurons[j].w;
				delete this->layers[i].neurons[j].deltaW;
				delete this->layers[i].neurons[j].lastDeltaW;
				delete[] this->layers[i].neurons[j].wCopy;
			}
		}
		delete[] this->layers[i].neurons;
	}
}

// ------------------------------
// Feed all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights()
{
	int a = -1, b = 1;
	for (auto i = 1; i < this->nOfLayers; i++)
		for (auto j = 0; j < this->layers[i].nOfNeurons; j++)
			for (auto k = 0; k < this->layers[i - 1].nOfNeurons; k++)
				this->layers[i].neurons[j].w[k] = ((double)rand() / RAND_MAX) * (b - a) + a;
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double *input)
{
	for (auto i = 0; i < this->layers[0].nOfNeurons; i++)
		this->layers[0].neurons[i].out = input[i];
}

// ------------------------------
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double *output)
{
	for (auto i = 0; i < this->layers[this->nOfLayers].nOfNeurons; i++)
		output[i] = this->layers[this->nOfLayers].neurons[i].out;
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights()
{
	for (auto i = 1; i < this->nOfLayers; i++)
		for (auto j = 0; j < this->layers[i].nOfNeurons; j++)
			for (auto k = 0; k < this->layers[i - 1].nOfNeurons; k++)
				this->layers[i].neurons[j].wCopy[k] = this->layers[i].neurons[j].w[k];
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights()
{
	for (auto i = 1; i < this->nOfLayers; i++)
		for (auto j = 0; j < this->layers[i].nOfNeurons; j++)
			for (auto k = 0; k < this->layers[i - 1].nOfNeurons; k++)
				this->layers[i].neurons[j].w[k] = this->layers[i].neurons[j].wCopy[k];
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate()
{
	for (auto i = 1; i < this->nOfLayers; i++)
	{
		for (auto j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			double sum = 0.0, net = 0.0;
			for (auto k = 1; k < this->layers[i - 1].nOfNeurons + 1; k++) // +1 due to the bias.
				sum = sum + this->layers[i].neurons[j].w[k] * this->layers[i - 1].neurons[k - 1].out;

			net = this->layers[i].neurons[j].w[0] + sum;
			this->layers[i].neurons[j].out = (double)(1.0 / 1.0 + exp(-net)); // Activation function: Sigmoid.
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(double *target)
{
	// MSE = 1 / N * sum( (target[i] - out[i])^2 )
	double sum = 0.0;
	for (auto i = 0; i < this->layers[this->nOfLayers].nOfNeurons; i++)
		sum = sum + pow(target[i] - this->layers[this->nOfLayers].neurons[i].out, 2);

	return (1 / this->layers[this->nOfLayers].nOfNeurons) * sum;
}

// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(double *target)
{
	// delta = -( target[i] - out{i}{j} * g'(sigmoid) )
	for (auto i = 0; i < this->layers[this->nOfLayers].nOfNeurons; i++)
		this->layers[this->nOfLayers].neurons[i].delta = -(target[i] - this->layers[this->nOfLayers].neurons[i].out) * this->layers[this->nOfLayers].neurons[i].out * (1.0 - this->layers[this->nOfLayers].neurons[i].out);

	for (auto i = this->nOfLayers - 1; i >= 1; i--)
	{
		for (auto j = 0; j < this->layers[i].nOfNeurons; j++)
		{
			double sum = 0.0;
			for (auto k = 0; k < this->layers[i + 1].nOfNeurons; k++) // +1 due to the bias.
				sum = sum + this->layers[i + 1].neurons[k].w[j + 1] * this->layers[i + 1].neurons[k].delta;

			this->layers[i].neurons[j].delta = sum * this->layers[i].neurons[j].out * (1 - this->layers[i].neurons[j].out);
		}
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange()
{
	//TODO
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment()
{
	//TODO
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork()
{
	//TODO
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(double *input, double *target)
{
	//TODO
	// feed
	// TRansparencia 63
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
	//TODO
	return -1.0;
}

// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset *pDatosTest)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers - 1].nOfNeurons;
	double *obtained = new double[numSalidas];

	cout << "Id,Predicted" << endl;

	for (i = 0; i < pDatosTest->nOfPatterns; i++)
	{

		feedInputs(pDatosTest->inputs[i]);
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
void MultilayerPerceptron::runOnlineBackPropagation(Dataset *trainDataset, Dataset *pDatosTest, int maxiter, double *errorTrain, double *errorTest)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving;
	double testError = 0;

	double validationError = 1;

	// Generate validation data
	if (validationRatio > 0 && validationRatio < 1)
	{
		// .......
	}

	// Learning
	do
	{

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
			countTrain = maxiter;
		}

		countTrain++;

		// Check validation stopping condition and force it
		// BE CAREFUL: in this case, we have to save the last validation error, not the minimum one
		// Apart from this, the way the stopping condition is checked is the same than that
		// applied for the training set

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Validation error: " << validationError << endl;

	} while (countTrain < maxiter);

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for (int i = 0; i < pDatosTest->nOfPatterns; i++)
	{
		double *prediction = new double[pDatosTest->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for (int j = 0; j < pDatosTest->nOfOutputs; j++)
			cout << pDatosTest->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;
	}

	testError = test(pDatosTest);
	*errorTest = testError;
	*errorTrain = minTrainError;
}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char *archivo)
{
	// Object for writing the file
	ofstream f(archivo);

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
bool MultilayerPerceptron::readWeights(const char *archivo)
{
	// Object for reading a file
	ifstream f(archivo);

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
