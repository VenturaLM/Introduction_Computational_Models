//============================================================================
// Introducción a los Modelos Computacionales
// Name        : la2.cpp
// Author      : Pedro A. Gutiérrez
// Student     : Ventura Lucena Martínez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>   // To obtain current time time()
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>
#include <float.h> // For DBL_MAX

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"

using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv)
{
	// Process arguments of the command line
	bool wflag = false, pflag = false, iflag = false, lflag = false, hflag = false, eflag = false, mflag = false, vflag = false, dflag = false, oflag = false, fflag = false, sflag = false;
	char *Tvalue = nullptr, *wvalue = nullptr, *tvalue = nullptr, *ivalue = nullptr, *lvalue = nullptr, *hvalue = nullptr, *evalue = nullptr, *mvalue = nullptr, *vvalue = nullptr, *dvalue = nullptr, *fvalue = nullptr;
	int c;

	opterr = 0;

	// a: Option that requires an argument
	// a:: The argument required is optional
	while ((c = getopt(argc, argv, "T:w:p:t:i:l:h:e:m:v:d:o::f:s::")) != -1)
	{
		// The parameters needed for using the optional prediction mode of Kaggle have been included.
		// You should add the rest of parameters needed for the lab assignment.
		switch (c)
		{
		case 'T':
			Tvalue = optarg;
			break;
		case 't':
			tvalue = optarg;
			break;
		case 'i':
			iflag = true;
			ivalue = optarg;
			break;
		case 'l':
			lflag = true;
			lvalue = optarg;
			break;
		case 'h':
			hflag = true;
			hvalue = optarg;
			break;
		case 'e':
			eflag = true;
			evalue = optarg;
			break;
		case 'm':
			mflag = true;
			mvalue = optarg;
			break;
		case 'v':
			vflag = true;
			vvalue = optarg;
			break;
		case 'd':
			dflag = true;
			dvalue = optarg;
			break;
		case 'o':
			oflag = true;
			break;
		case 'f':
			fflag = true;
			fvalue = optarg;
			break;
		case 's':
			sflag = true;
			break;
		case 'w':
			wflag = true;
			wvalue = optarg;
			break;
		case 'p':
			pflag = true;
			break;
		case '?':
			if (optopt == 'T' || optopt == 'w' || optopt == 'p' || optopt == 't' || optopt == 'i' || optopt == 'l' || optopt == 'h' || optopt == 'e' || optopt == 'v' || optopt == 'd' || optopt == 'f')
				fprintf(stderr, "The option -%c requires an argument.\n", optopt);
			else if (isprint(optopt))
				fprintf(stderr, "Unknown option `-%c'.\n", optopt);
			else
				fprintf(stderr,
						"Unknown character `\\x%x'.\n",
						optopt);
			return EXIT_FAILURE;
		default:
			return EXIT_FAILURE;
		}
	}

	if (!pflag)
	{
		//////////////////////////////////
		// TRAINING AND EVALUATION MODE //
		//////////////////////////////////

		// Multilayer perceptron object
		MultilayerPerceptron mlp;
		if (eflag)
			mlp.eta = stod(evalue);
		if (mflag)
			mlp.mu = stod(mvalue);
		if (vflag)
			mlp.validationRatio = stod(vvalue);
		if (dflag)
			mlp.decrementFactor = stod(dvalue);

		// Off-line / Online version.
		mlp.online = false;
		if (oflag)
			mlp.online = true;

		// Output function: 0 = sigmoid / 1 = softmax.
		mlp.outputFunction = 0;
		if (sflag)
			mlp.outputFunction = 1;

		// Type of error considered
		int error = 0;
		if (fflag)
		{
			error = stoi(fvalue);
			if (error != 1)
				error = 1;
		}

		// Maximum number of iterations
		int maxIter = 500; // This should be completed
		if (iflag)
			maxIter = stoi(ivalue);

		// Read training and test data: call to mlp.readData(...)
		Dataset *trainDataset = mlp.readData(tvalue);
		Dataset *testDataset = mlp.readData(Tvalue);

		// Initialize topology vector
		int layers = 1;
		if (lflag)
			layers = stoi(lvalue);

		int *topology = new int[layers + 2];

		if (hflag)
		{
			int h = stoi(hvalue);
			topology[0] = trainDataset->nOfInputs;

			for (int i = 1; i < (layers + 2 - 1); i++)
				topology[i] = h;

			topology[layers + 2 - 1] = trainDataset->nOfOutputs;
		}

		mlp.initialize(layers + 2, topology);

		// Seed for random numbers
		int seeds[] = {1, 2, 3, 4, 5};
		double *trainErrors = new double[5];
		double *testErrors = new double[5];
		double *trainCCRs = new double[5];
		double *testCCRs = new double[5];
		double bestTestError = DBL_MAX;
		for (int i = 0; i < 5; i++)
		{
			//cout << "**********" << endl;
			//cout << "SEED " << seeds[i] << endl;
			//cout << "**********" << endl;
			srand(seeds[i]);

			mlp.runBackPropagation(trainDataset, testDataset, maxIter, &(trainErrors[i]), &(testErrors[i]), &(trainCCRs[i]), &(testCCRs[i]), error);
			cout << "We end!! => Final test CCR: " << testCCRs[i] << endl;

			// We save the weights every time we find a better model
			if (wflag && testErrors[i] <= bestTestError)
			{
				mlp.saveWeights(wvalue);
				bestTestError = testErrors[i];
			}
		}

		double trainAverageError = 0, trainStdError = 0;
		double testAverageError = 0, testStdError = 0;
		double trainAverageCCR = 0, trainStdCCR = 0;
		double testAverageCCR = 0, testStdCCR = 0;

		// Obtain training and test averages and standard deviations.
		computeAverageErrors(testErrors, trainErrors, testAverageError, trainAverageError);
		computeStdErrors(testErrors, trainErrors, testStdError, trainStdError, testAverageError, trainAverageError);

		computeAverageErrors(testCCRs, trainCCRs, testAverageCCR, trainAverageCCR);
		computeStdErrors(testCCRs, trainCCRs, testStdCCR, trainStdCCR, testAverageCCR, trainAverageCCR);

		cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

		cout << "FINAL REPORT" << endl;
		cout << "*************" << endl;
		cout << "Train error (Mean +- SD): " << trainAverageError << " +- " << trainStdError << endl;
		cout << "Test error (Mean +- SD): " << testAverageError << " +- " << testStdError << endl;
		cout << "Train CCR (Mean +- SD): " << trainAverageCCR << " +- " << trainStdCCR << endl;
		cout << "Test CCR (Mean +- SD): " << testAverageCCR << " +- " << testStdCCR << endl;
		return EXIT_SUCCESS;
	}
	else
	{

		//////////////////////////////
		// PREDICTION MODE (KAGGLE) //
		//////////////////////////////

		// You do not have to modify anything from here.

		// Multilayer perceptron object
		MultilayerPerceptron mlp;

		// Initializing the network with the topology vector
		if (!wflag || !mlp.readWeights(wvalue))
		{
			cerr << "Error while reading weights, we can not continue" << endl;
			exit(-1);
		}

		// Reading training and test data: call to mlp.readData(...)
		Dataset *testDataset;
		testDataset = mlp.readData(Tvalue);
		if (testDataset == NULL)
		{
			cerr << "The test file is not valid, we can not continue" << endl;
			exit(-1);
		}

		mlp.predict(testDataset);

		return EXIT_SUCCESS;
	}
}
