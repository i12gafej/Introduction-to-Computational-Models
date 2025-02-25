/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/


#ifndef _MULTILAYERPERCEPTRON_H_
#define _MULTILAYERPERCEPTRON_H_

#include "util.h"

namespace imc{

// Suggested structures
// ---------------------
struct Neuron {
	double  out;            /* Output produced by the neuron (out_j^h)*/
	double  delta;          /* Derivative of the output produced by the neuron (delta_j^h)*/
	double* w;              /* Input weight vector (w_{ji}^h)*/
	double* deltaW;         /* Change to be applied to every weight (\Delta_{ji}^h (t))*/
	double* lastDeltaW;     /* Last change applied to the every weight (\Delta_{ji}^h (t-1))*/
	double* wCopy;          /* Copy of the input weights */
};

struct Layer {
	int     nOfNeurons;   /* Number of neurons of the layer*/
	Neuron* neurons;      /* Vector with the neurons of the layer*/
};

class MultilayerPerceptron {
private:
	int    nOfLayers;     /* Total number of layers in the network */
	Layer* layers;        /* Vector containing every layer */

	// Free memory for the data structures
	void freeMemory();

	// Feel all the weights (w) with random numbers between -1 and +1
	void randomWeights();

	// Feed the input neurons of the network with a vector passed as an argument
	void feedInputs(double* input);

	// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
	void getOutputs(double* output);

	// Make a copy of all the weights (copy w in wCopy)
	void copyWeights();

	// Restore a copy of all the weights (copy wCopy in w)
	void restoreWeights();

	// Apply softmax to the output layer
	void applySoftmax(Layer& layer);

	// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
	void forwardPropagate();

	// Calculate de Cross Entropy error
	double calculateCrossEntropyError(double* target);

	// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
	double obtainError(double* target);

	// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
	void backpropagateError(double* target);

	// Accumulate the changes produced by one pattern and save them in deltaW
	void accumulateChange();

	// Update the network weights, from the first layer to the last one
	void weightAdjustment();

	// Update the weights of the network using batch
	void weightAdjustmentBatch(int numPatterns);

	// Print the network, i.e. all the weight matrices
	void printNetwork();

	// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
	// input is the input vector of the pattern and target is the desired output vector of the pattern
	void performEpochOnline(double* input, double* target);

	// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
	void performEpochOffline(double* input, double* target);


public:
	// Values of the parameters (they are public and can be updated from outside)
	double eta;             // Learning rate
	double mu;              // Momentum factor
    bool useSoftmax;        // Flag to apply softmax to the output layer
    bool useCrossEntropy;	// Flag to use Cross Entropy error

	// Constructor: Default values for all the parameters
	MultilayerPerceptron();

	// DESTRUCTOR: free memory
	~MultilayerPerceptron();

	// Allocate memory for the data structures
    // nl is the number of layers and npl is a vetor containing the number of neurons in every layer
    // Give values to Layer* layers
	int initialize(int nl, int npl[]);

	// Test the network with a dataset and return the MSE
	double test(util::Dataset* dataset);

	// Calculate the CCR
	double calculateCorrectClassifiedPatterns(util::Dataset* dataset);

	// Obtain the predicted outputs for a dataset
	void predict(util::Dataset* testDataset);

	// Perform an online training for a specific dataset
	void trainOnline(util::Dataset* trainDataset);

	// Perform an offline training for a specific dataset
	void trainOffline(util::Dataset* trainDataset);

	// Run the traning algorithm for a given number of epochs, using trainDataset
    // Once finished, check the performance of the network in testDataset
    // Both training and test MSEs should be obtained and stored in errorTrain and errorTest
	void runOnlineBackPropagation(util::Dataset * trainDataset, util::Dataset * testDataset, int maxiter, double *errorTrain, double *errorTest);

	// Run the traning algorithm for a given number of epochs, using trainDataset
	// Once finished, check the performance of the network in testDataset
	// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
	void runOfflineBackPropagation(util::Dataset * trainDataset, util::Dataset * testDataset, int maxiter, double *errorTrain, double *errorTest);

	// Optional Kaggle: Save the model weights in a textfile
	bool saveWeights(const char * archivo);

	// Optional Kaggle: Load the model weights from a textfile
	bool readWeights(const char * archivo);

};

};

#endif
