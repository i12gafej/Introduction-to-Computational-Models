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
	layers = NULL;
    useSoftmax = false;
    useCrossEntropy = false;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {
	nOfLayers = nl;
    layers = new Layer[nOfLayers]; // N number of layers

    for(int i = 0; i < nOfLayers; i++){ //Through all the layers
        layers[i].nOfNeurons = npl[i]; //the number of neurons per layer 
        layers[i].neurons = new Neuron[layers[i].nOfNeurons]; // as neurons as nOfNeurons

        for(int j = 0; j < layers[i].nOfNeurons; j++){// Through the neurons of each layer
            if(i > 0){ // Not the bias
                layers[i].neurons[j].w = new double[layers[i-1].nOfNeurons + 1]; // Vector of the weights of the nOfNeurons of the previous layer
                                                                                // plus one because of the bias
                layers[i].neurons[j].deltaW = new double[layers[i-1].nOfNeurons + 1]; // same with deltaW, lastDeltaW and wCopy
                layers[i].neurons[j].lastDeltaW = new double[layers[i-1].nOfNeurons + 1];
                layers[i].neurons[j].wCopy = new double[layers[i-1].nOfNeurons + 1];
            }
        }
    }

    return 1; // It's all right
}


// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {
	for(int i = 0; i < nOfLayers; i++) {                
        for(int j = 0; j < layers[i].nOfNeurons; j++){ 
            if (i > 0){                           // Not the bias                                   
                delete[] layers[i].neurons[j].w; // Free memory of each vector
                delete[] layers[i].neurons[j].deltaW;
                delete[] layers[i].neurons[j].lastDeltaW;
                delete[] layers[i].neurons[j].wCopy;
            }
        }
        delete[] layers[i].neurons; // The free the neurons
    }
    delete[] layers; // And finally the layers
}

// ------------------------------
// Feel all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {
	for(int i = 1; i < nOfLayers; i++){                          //For each layer since the first hidden layer
        for(int j = 0; j < layers[i].nOfNeurons; j++){           //Go through each neuron
            for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++){ //Go through each weight 
                                                                // of the previous layer + 1 (bias)
                layers[i].neurons[j].w[k] = util::randomDouble(-1.0, 1.0); // initialize the weights with random values
            }
        }
    }
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input) {
	for(int i = 0; i < layers[0].nOfNeurons; i++){ // Layer 0 is the input layer
        layers[0].neurons[i].out = input[i]; // Set input neurons output to the input values
    }
}

// ------------------------------
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output) {
	for(int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++){ // Layer nOfLayers - 1 is the output layer
        output[i] = layers[nOfLayers - 1].neurons[i].out; // Output from the last layer
    }
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {
	for(int i = 1; i < nOfLayers; i++){                          // From each layer since the first hidden one
        for(int j = 0; j < layers[i].nOfNeurons; j++){           // and neuron
            for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++){ // and weight of the previous layer + 1 (bias)
                layers[i].neurons[j].wCopy[k] = layers[i].neurons[j].w[k]; // Copy current weights
            }
        }
    }
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {
	for(int i = 1; i < nOfLayers; i++){                          // From each layer since the first hidden one
        for(int j = 0; j < layers[i].nOfNeurons; j++){           // and neuron
            for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++){ // and weight of the previous layer + 1 (bias)
                layers[i].neurons[j].w[k] = layers[i].neurons[j].wCopy[k]; // Restore weights
            }
        }
    }
}

//----------------------------------
// Apply the softmax function to the output layer
void MultilayerPerceptron::applySoftmax(Layer& layer) {
    double sumExp = 0.0;
    for (int i = 0; i < layer.nOfNeurons; i++) {
        layer.neurons[i].out = exp(layer.neurons[i].out);  // Exponentiation
        sumExp += layer.neurons[i].out;
    }
    for (int i = 0; i < layer.nOfNeurons; i++) {
        layer.neurons[i].out /= sumExp;  // Normalize to create probabilities
    }
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
// Forward propagation with softmax in the output layer
void MultilayerPerceptron::forwardPropagate() {
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            double net = layers[i].neurons[j].w[0]; // Bias
            for (int k = 1; k <= layers[i - 1].nOfNeurons; k++) {
                net += layers[i].neurons[j].w[k] * layers[i - 1].neurons[k - 1].out;
            }
            if (i == nOfLayers - 1 || useSoftmax) { // Output layer
                layers[i].neurons[j].out = net; // Softmax applied later
            } else { // Hidden layers use sigmoid
                layers[i].neurons[j].out = 1.0 / (1.0 + exp(-net));
            }
        }
        if (i == nOfLayers - 1 || useSoftmax) { // Apply softmax in the output layer
            applySoftmax(layers[i]);
        }
    }
}

// ------------------------------
// Calculate de Cross Entropy error
double MultilayerPerceptron::calculateCrossEntropyError(double* target) {
    double loss = 0.0;
    for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++) {
        double output = layers[nOfLayers - 1].neurons[i].out;
        loss -= target[i] * log(output + 1e-8);
    }
    return loss / layers[nOfLayers - 1].nOfNeurons;
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(double* target) {
	if (useCrossEntropy) {
        return calculateCrossEntropyError(target);
    } else {
        double mse = 0.0;
        for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++) {
            double error = target[i] - layers[nOfLayers - 1].neurons[i].out;
            mse += error * error;
        }
        return mse / layers[nOfLayers - 1].nOfNeurons;
    }
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(double* target) {
	// Output layer error (Softmax derivative)
    for (int j = 0; j < layers[nOfLayers - 1].nOfNeurons; j++) {
        double out_j = layers[nOfLayers - 1].neurons[j].out;
        double deltaSum = 0.0;

        for (int i = 0; i < layers[nOfLayers - 1].nOfNeurons; i++) {
            double out_i = layers[nOfLayers - 1].neurons[i].out;
            double derivative_error = useCrossEntropy ? 
                -(target[i] / out_i) :                       // Cross-Entropy derivative
                -2 * (target[i] - out_i) * out_i * (1.0 - out_i); // MSE derivative

            if (i == j) { // Case i == j
                deltaSum += derivative_error * out_j * (1.0 - out_j);
            } else { // Case i != j
                deltaSum += derivative_error * out_j * (-out_i);
            }
        }
        layers[nOfLayers - 1].neurons[j].delta = deltaSum; // Store the computed delta
    }

    // Hidden layers error
    for (int i = nOfLayers - 2; i > 0; i--) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            double deltaSum = 0.0;
            for (int k = 0; k < layers[i + 1].nOfNeurons; k++) {
                deltaSum += layers[i + 1].neurons[k].delta * layers[i + 1].neurons[k].w[j + 1];
            }
            double out = layers[i].neurons[j].out;
            layers[i].neurons[j].delta = deltaSum * out * (1.0 - out); // Sigmoid derivative
        }
    }
}


// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {
	 for(int i = 1; i < nOfLayers; i++){                                            // Start from the first hidden layer
        for(int j = 0; j < layers[i].nOfNeurons; j++){                              // Through each neuron
            for(int k = 0; k <= layers[i - 1].nOfNeurons; k++){ 
                // and through each weight icluding bias with the = sign
                double prev_out = (k == 0) ? 1.0 : layers[i - 1].neurons[k - 1].out;     // bias = 1, otherwise the output of the previous layer
                layers[i].neurons[j].deltaW[k] += layers[i].neurons[j].delta * prev_out; // Accumulate the changes
            }
        }
    }
}

// ------------------------------
// Update the weights of the network using batch
void MultilayerPerceptron::weightAdjustmentBatch(int numPatterns) {
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            for (int k = 0; k <= layers[i - 1].nOfNeurons; k++) {
                double prev_out = (k == 0) ? 1.0 : layers[i - 1].neurons[k - 1].out;
                layers[i].neurons[j].w[k] -= (eta * layers[i].neurons[j].deltaW[k] / numPatterns)
                                            + (mu * layers[i].neurons[j].lastDeltaW[k]);
                layers[i].neurons[j].lastDeltaW[k] = eta * layers[i].neurons[j].deltaW[k] / numPatterns;
                layers[i].neurons[j].deltaW[k] = 0.0;  // Reset for next batch
            }
        }
    }
}
// ------------------------------
// Update the weights with momentum
void MultilayerPerceptron::weightAdjustment() {
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            for (int k = 0; k <= layers[i - 1].nOfNeurons; k++) {
                double prev_out = (k == 0) ? 1.0 : layers[i - 1].neurons[k - 1].out;
                layers[i].neurons[j].w[k] -= eta * layers[i].neurons[j].delta * prev_out
                                            + mu * layers[i].neurons[j].lastDeltaW[k];
                layers[i].neurons[j].lastDeltaW[k] = eta * layers[i].neurons[j].delta * prev_out;
            }
        }
    }
}
// // ------------------------------
// // Update the network weights, from the first layer to the last one
// void MultilayerPerceptron::weightAdjustment() {
// 	for(int i = 1; i < nOfLayers; i++){                         // Start from the first hidden layer
//         for(int j = 0; j < layers[i].nOfNeurons; j++){          // Through each neuron
//             for(int k = 0; k <= layers[i - 1].nOfNeurons; k++){ // and through each weight icluding bias with the = sign
//                 // Apply weight change
//                 layers[i].neurons[j].w[k] -= eta * layers[i].neurons[j].deltaW[k]
//                                              + mu * layers[i].neurons[j].lastDeltaW[k];
//                 // Store this delta for momentum use
//                 layers[i].neurons[j].lastDeltaW[k] = layers[i].neurons[j].deltaW[k];
//                 // Reset deltaW for the next pattern
//                 layers[i].neurons[j].deltaW[k] = 0.0;
//             }
//         }
//     }
// }

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {
	for (int i = 1; i < nOfLayers; i++){
        cout << "Layer " << i << endl;
        for (int j = 0; j < layers[i].nOfNeurons; j++){
            cout << "Neuron " << j << ": ";
            for (int k = 0; k <= layers[i - 1].nOfNeurons; k++){
                cout << layers[i].neurons[j].w[k] << " ";
            }
            cout << endl;
        }
    }
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(double* input, double* target) {
	// Feed input into the network
    feedInputs(input);
    // Forward propagation
    forwardPropagate();
    // Backpropagate the error
    backpropagateError(target);
    // Accumulate changes for the current pattern
    accumulateChange();
    // Adjust the weights
    weightAdjustment();
}

// ------------------------------
// Perform an online training for a specific trainDataset
void MultilayerPerceptron::trainOnline(Dataset* trainDataset) {
	int i;
	for(i=0; i<trainDataset->nOfPatterns; i++){
		performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i]);
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
void MultilayerPerceptron::performEpochOffline(double *input, double *target) {
    // Propagar los valores hacia adelante
    feedInputs(input);
    forwardPropagate();

    // Propagar el error hacia atrás
    backpropagateError(target);

    // Acumular los cambios
    accumulateChange();
}


// ------------------------------
// Perform an offline training for a specific trainDataset
void MultilayerPerceptron::trainOffline(Dataset *trainDataset) {
    // Reset acumuladores de cambios
    for (int i = 1; i < nOfLayers; i++) {
        for (int j = 0; j < layers[i].nOfNeurons; j++) {
            for (int k = 0; k <= layers[i - 1].nOfNeurons; k++) {
                layers[i].neurons[j].deltaW[k] = 0.0;  // Reiniciar acumuladores
            }
        }
    }

    // Procesar cada patrón
    for (int i = 0; i < trainDataset->nOfPatterns; i++) {
        performEpochOffline(trainDataset->inputs[i], trainDataset->outputs[i]);
    }

    // Ajustar los pesos acumulados
    weightAdjustmentBatch(trainDataset->nOfPatterns);
}

// ------------------------------
// Test the network with a dataset and return the MSE
double MultilayerPerceptron::test(Dataset* testDataset) {
	double error = 0.0;
    for (int i = 0; i < testDataset->nOfPatterns; i++) {   // Every pattern
        double* prediction = new double[testDataset->nOfOutputs]; // Prediction vector with the outputs
        // Feed inputs and propagate
        feedInputs(testDataset->inputs[i]);
        forwardPropagate();
        getOutputs(prediction);
        // Calculate the error for this pattern
        if (useCrossEntropy) {
            error += calculateCrossEntropyError(testDataset->outputs[i]);
        } else {
            for (int j = 0; j < testDataset->nOfOutputs; j++) {
                double diff = testDataset->outputs[i][j] - prediction[j];
                error += diff * diff;
            }
        }

        delete[] prediction;
    }
    return error / testDataset->nOfPatterns;
}

double MultilayerPerceptron::calculateCorrectClassifiedPatterns(Dataset* dataset) {
    int correct = 0;  // Contador de patrones correctamente clasificados

    for (int i = 0; i < dataset->nOfPatterns; i++) {
        // Propaga la entrada hacia adelante
        feedInputs(dataset->inputs[i]);
        forwardPropagate();

        // Obtén la salida de la red neuronal
        double* output = new double[layers[nOfLayers - 1].nOfNeurons];
        getOutputs(output);

        // Encuentra la clase predicha (índice con el valor más alto)
        int predicted = 0;
        double maxOutput = output[0];
        for (int j = 1; j < layers[nOfLayers - 1].nOfNeurons; j++) {
            if (output[j] > maxOutput) {
                maxOutput = output[j];
                predicted = j;
            }
        }

        // Encuentra la clase real (índice con el valor más alto en el target)
        int actual = 0;
        double maxTarget = dataset->outputs[i][0];
        for (int j = 1; j < layers[nOfLayers - 1].nOfNeurons; j++) {
            if (dataset->outputs[i][j] > maxTarget) {
                maxTarget = dataset->outputs[i][j];
                actual = j;
            }
        }

        // Si la clase predicha coincide con la real, incrementa el contador
        if (predicted == actual) {
            correct++;
        }

        delete[] output;  // Libera la memoria reservada para la salida
    }

    // Calcula el CCR como la proporción de clasificaciones correctas
    //cout << "Correctly Classified Ratio (CCR): " << ccr << " (" << correct << "/" << dataset->nOfPatterns << ")" << endl;
    return correct;
}


// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset* pDatosTest)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * obtained = new double[numSalidas];
	
	cout << "Id,Predicted" << endl;
	
	for (i=0; i<pDatosTest->nOfPatterns; i++){

		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(obtained);
		
		cout << i;

		for(j = 0; j < numSalidas; j++)
			cout << "," << obtained[j];
		cout << endl;

	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOnlineBackPropagation(Dataset * trainDataset, Dataset * testDataset, int maxiter, double *errorTrain, double *errorTest)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving;
	double testError = 0;

	// Learning
	do {

		trainOnline(trainDataset);
		double trainError = test(trainDataset);
        double testError = test(testDataset);
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if(iterWithoutImproving==50){
			cout << "We exit because the training is not improving!!"<< endl;
			restoreWeights();
			countTrain = maxiter;
		}


		countTrain++;

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Test error: " << testError << endl;
        //printNetwork();
	} while ( countTrain<maxiter );

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<testDataset->nOfPatterns; i++){
		double* prediction = new double[testDataset->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<testDataset->nOfOutputs; j++)
			cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}

	testError = test(testDataset);
	*errorTest=testError;
	*errorTrain=minTrainError;

}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOfflineBackPropagation(Dataset * trainDataset, Dataset * testDataset, int maxiter, double *errorTrain, double *errorTest)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving;
	double testError = 0;

	// Learning
	do {

		trainOffline(trainDataset);
		double trainError = test(trainDataset);
        double testError = test(testDataset);
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if(iterWithoutImproving==50){
			cout << "We exit because the training is not improving!!"<< endl;
			restoreWeights();
			countTrain = maxiter;
		}


		countTrain++;

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Test error: " << testError << endl;

	} while ( countTrain<maxiter );

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<testDataset->nOfPatterns; i++){
		double* prediction = new double[testDataset->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<testDataset->nOfOutputs; j++)
			cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}

	testError = test(testDataset);
	*errorTest=testError;
	*errorTrain=minTrainError;

}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * archivo)
{
	// Object for writing the file
	ofstream f(archivo);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
		f << " " << layers[i].nOfNeurons;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * archivo)
{
	// Object for reading a file
	ifstream f(archivo);

	if (!f.is_open()) {
		cerr << "Error: Unable to open weights file." << endl;
		return false;
	}

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for(int i = 0; i < nl; i++)
		f >> npl[i];

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
