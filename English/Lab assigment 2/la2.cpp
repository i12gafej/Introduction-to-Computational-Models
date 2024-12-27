//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // To obtain current time time()
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>
#include <float.h>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"

using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv) {
    // Process arguments of the command line (NEW)
    bool Tflag = false, tflag = false, wflag = false, pflag = false, useSoftmax = false,useCrossEntropy = false,useOnline = false;
    char *Tvalue = NULL, *tvalue = NULL, *wvalue = NULL;
    int iterations = 1000;  // Default number of iterations
    int numHiddenLayers = 1;  // Default number of hidden layers
    int numNeuronsHidden = 5;  // Default number of neurons per hidden layer
    double eta = 0.1;  // Default learning rate
    double mu = 0.9;   // Default momentum factor
    int c;

    opterr = 0;

    // Parse command line arguments (NEW)
    while ((c = getopt(argc, argv, "T:t:w:i:l:h:e:m:psf:os")) != -1)
    {
        switch(c){
            case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;
            case 't':
                tflag = true;
                tvalue = optarg;
                break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'i':
                iterations = atoi(optarg);
                break;
            case 'l':
                numHiddenLayers = atoi(optarg);
                break;
            case 'h':
                numNeuronsHidden = atoi(optarg);
                break;
            case 'e':
                eta = atof(optarg);
                break;
            case 'm':
                mu = atof(optarg);
                break;
            case 'p':
                pflag = true;
                break;
            case 'f':
                useCrossEntropy = (atoi(optarg) == 1);
                break;
            case 'o':
                useOnline = true;
                break;
            case 's':
                useSoftmax = true;
                break;  
            case '?':
                if (optopt == 'T' || optopt == 't' || optopt == 'w' || optopt == 'i' || optopt == 'l' || optopt == 'h' || optopt == 'e' || optopt == 'm' || optopt == 'f')
                    fprintf (stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr, "Unknown character `\\x%x'.\n", optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }

    if (!tflag) {
        fprintf(stderr, "The option -t (training file) is required.\n");
        return EXIT_FAILURE;
    }

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Set parameters (NEW)
        mlp.eta = eta;
        mlp.mu = mu;

        // Read training and test data (NEW)
        Dataset *trainDataset = readData(tvalue); // Training file
        Dataset *testDataset = (Tflag) ? readData(Tvalue) : trainDataset; // Test dataset

        // Initialize topology vector (NEW)
        int layers = 2 + numHiddenLayers;
        int *topology = new int[layers];
        topology[0] = trainDataset->nOfInputs;
        for (int i = 1; i <= numHiddenLayers; i++) {
            topology[i] = numNeuronsHidden;
        }
        topology[layers-1] = trainDataset->nOfOutputs;

        // Default configuration of the use of softmax and cross entropy
        if (trainDataset->nOfOutputs > 2) {
            mlp.useSoftmax = true;
            mlp.useCrossEntropy = true;
        }
        else if (trainDataset->nOfOutputs <= 2) {
            mlp.useCrossEntropy = false;
            mlp.useSoftmax = false;
        }

        mlp.useCrossEntropy = useCrossEntropy;
        
        // When using softmax, it's mandatory to use cross entropy
        if (useSoftmax){
            mlp.useCrossEntropy = true;
            mlp.useSoftmax = true;
        } else if (useCrossEntropy){
            mlp.useCrossEntropy = true;
            mlp.useSoftmax = false;
        }


        // Initialize the network using the topology vector
        mlp.initialize(layers, topology);

        // Seed for random numbers
        int seeds[] = {1, 2, 3, 4, 5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = DBL_MAX;

        for(int i=0; i<5; i++){
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            //mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations, &(trainErrors[i]), &(testErrors[i]));
            if (useOnline) {
                cout << "Running Online Backpropagation..." << endl;
                mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations, &(trainErrors[i]), &(testErrors[i]));
            } else {
                cout << "Running Offline Backpropagation..." << endl;
                mlp.runOfflineBackPropagation(trainDataset, testDataset, iterations, &(trainErrors[i]), &(testErrors[i]));
            }
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if(wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        // Obtain training and test averages and standard deviations
        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;

        for (int i = 0; i < 5; i++) {
            averageTestError += testErrors[i];
            averageTrainError += trainErrors[i];
        }
        averageTestError /= 5;
        averageTrainError /= 5;

        for (int i = 0; i < 5; i++) {
            stdTestError += pow(testErrors[i] - averageTestError, 2);
            stdTrainError += pow(trainErrors[i] - averageTrainError, 2);
        }
        stdTestError = sqrt(stdTestError / 5);
        stdTrainError = sqrt(stdTrainError / 5);

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test  error (Mean +- SD): " << averageTestError << " +- " << stdTestError << endl;
        
        return EXIT_SUCCESS;
    }
    else {
        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading test data
        Dataset *testDataset;
        testDataset = readData(Tvalue);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }
}
