#pragma once
#include<iostream>
#include <string>
#include <vector>
#include "otheUsefulrFun.h"
#include "overwritesHeader.h"

using namespace std;

// set of NN parameters needed to store to passing between NN stages
typedef struct parameters
{
	double loss;
	vector <vector<double>> z2;
	vector <vector<double>> a3;
	vector <vector<double>> theta1_grad;
	vector <vector<double>> theta2_grad;
	vector <vector<double>> theta_1;
	vector <vector<double>> theta_2;

}parameters;

// neuralNetwork .. the main function which will receive the data .. call other NN functions .. and return the results ..
results neuralNetwork(int datasetNum, int featureNum, vector<vector <double>> trainInput, vector<double>trainOutput, vector<vector <double>> testInput, vector<double> testOutput,int hidden_layer_size, double learning_rate, int numofIterations, int printitr, double num1, double num2);

// randInitializeWeights function to initialize theta 1 and theta 2 .. given the in and out dimension of the hidden layer
vector<vector<double>> randInitializeWeights(int divect1, int divect2, double num1, double num2);

// apply the activation function "sigmoid" on the given vector .. uesd in forward propagation 
vector <vector<double>> sigmoid(vector <vector<double>> vect1);

// apply the dervative of sigmoid function ... used in backward propagation 
vector <vector<double>> gradientsigmoid(vector <vector<double>> vect1);


// forward propagation function to calculate propalites of each output class labels for each example
vector < vector <double>> forwardPropagation(vector<vector<double>> features, vector <vector <double>> theta1, vector <vector <double>> theta2);

// function for calculate the cost given aset of parameters 
parameters costFunction(vector <vector<double>> featuresX, vector<double> output, int num_labels, vector <vector <double>> theta1, vector <vector <double>> theta2);

// backpropagation function to calculate dervatives to update weights ..
parameters backpropagation(int num_labels, vector<vector<double>> input, vector<double> output, vector<vector<double>>theta1, vector<vector<double>>theta2);

// update weights based on backpropagation feedback ..
parameters updateParameters(vector<vector<double>> theta1, vector<vector<double>>theta2, vector<vector<double>> theta1_grad, vector<vector<double>>theta2_grad, double learning_rate);

// predict classes using the created NN model .. 
vector<double> predict(int datasetNum,vector <vector<double>> featuresX, vector <vector <double>> theta1, vector <vector <double>> theta2);
