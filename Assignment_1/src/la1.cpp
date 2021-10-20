//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

// Example execution:   ./la1 -T ../../datasets/dat/test_parkinsons.dat -t ../../datasets/dat/train_parkinsons.dat  -i 1000 -l 1 -h 10 -e 0.1 -m 0.9 -v 0.0 -d 1.0

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>   // To obtain current time time()
#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>
#include <cstring>

#include "imc/MultilayerPerceptron.h"

using namespace imc;
using namespace std;

int main(int argc, char **argv)
{
    // Process arguments of the command line
    bool Tflag = false, wflag = false, pflag = false, tflag = false, iflag = false, lflag = false, hflag = false, eflag = false, mflag = false;
    char *Tvalue = nullptr, *wvalue = nullptr, *tvalue = nullptr, *ivalue = nullptr, *lvalue = nullptr, *hvalue = nullptr, *evalue = nullptr, *mvalue = nullptr;
    int c;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "T:w:p:t:i:l:h:e:m:")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch (c)
        {
        case 'T':
            Tflag = true;
            Tvalue = optarg;
            break;
        case 't':
            tflag = true;
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
        case 'w':
            wflag = true;
            wvalue = optarg;
            break;
        case 'p':
            pflag = true;
            break;
        case '?':
            if (optopt == 'T' || optopt == 'w' || optopt == 'p' || optopt == 't' || optopt == 'i' || optopt == 'l' || optopt == 'h' || optopt == 'e')
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

    // pflag --> Flag indicating that the program is running in prediction mode.
    if (!pflag)
    {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value;
        int iterations = 1000;
        if (iflag)
            iterations = stoi(ivalue);

        // Read training and test data: call to mlp.readData(...)
        Dataset *trainDataset = mlp.readData(tvalue);
        Dataset *testDataset = mlp.readData(Tvalue);

        // Initialize topology vector
        int layers = 1;
        if (lflag)
            layers = stoi(lvalue);

        int *topology = new int[3];

        // Topology:
        if (hflag)
        {
            int h = stoi(hvalue);
            topology[0] = trainDataset->nOfInputs;
            for (int i = 1; i < layers + 2; i++)
                topology[i] = h;
            topology[layers] = trainDataset->nOfOutputs;
        }

        mlp.initialize(layers + 2, topology);

        // Seed for random numbers
        int seeds[] = {1, 2, 3, 4, 5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = 1;
        for (int i = 0; i < 5; i++)
        {
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations, &(trainErrors[i]), &(testErrors[i]));
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if (wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;

        // Obtain training and test averages and standard deviations

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;
        return EXIT_SUCCESS;
    }
    else
    {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

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
