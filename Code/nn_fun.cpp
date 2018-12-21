#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <ppl.h>
#include <windows.h>

#include "concurrent_vector.h"
#include "overwritesHeader.h"
#include "otheUsefulrFun.h"
#include "nn_header.h"

using namespace std;
using namespace concurrency;


results neuralNetwork(int datasetNum, int featureNum, vector<vector <double>> trainInput, vector<double>trainOutput, vector<vector <double>> testInput, vector<double> testOutput, int hidden_layer_size, double learning_rate, int numofIterations, int printitr, double num1, double num2)
{

	//Neural Network paramters settings (1 hidden layer)
	int input_layer_size = featureNum;
	int num_labels = 2;	
	

	//  ... declare weights 
	vector<vector<double>> theta1;
	vector<vector<double>> theta2;

	// initialize weights randomly ...
	theta1 = randInitializeWeights(input_layer_size, hidden_layer_size, num1, num2);
	theta2 = randInitializeWeights(hidden_layer_size, num_labels, num1, num2);

	// for training 
	for (int it = 1; it <= numofIterations; it++)
	{
		// calculate the cost function (which will call forward propagtion inside it)
		parameters pobj1, pobj2, pobj3;
		pobj1 = costFunction(trainInput, trainOutput, num_labels, theta1, theta2);
		double cost = pobj1.loss;

		// for tracing 
		if ((it == 1) || (it % printitr == 0))
		{
			cout << " iterantion # : " << it << " , cost = " << cost << endl;
		}

		// call backpropagation to get feedback on the model and improve it ..
		pobj2 = backpropagation(num_labels, trainInput, trainOutput, theta1, theta2);

		// update parameters based on backpropagation ..
		pobj3 = updateParameters(theta1, theta2, pobj2.theta1_grad, pobj2.theta2_grad, learning_rate);
		theta1 = pobj3.theta_1;
		theta2 = pobj3.theta_2;
	}

	// predict output  for training data
	vector<double> predictedtrainOutput = predict(datasetNum, trainInput, theta1, theta2);

	// measure training accuracy
	results resobj1 = calculateAccuracy( datasetNum, trainOutput, predictedtrainOutput);
	cout << " ----------------------------------------------------------------" << endl << endl;
	printf(" Training Accurcy = %.2f%% \n", resobj1.accurcy * 100);

	
	// confusion matrix
	if (datasetNum == 2)
	{
		cout << " ------------------------------\n";
		cout << " Confusion Matrix for Training Data : \n";
		cout << "\t p \t\t n \n";
		cout << " ------------------------------\n";
		cout << " p \t" << resobj1.pp << "\t\t" << resobj1.pn << "\n";
		cout << " n \t" << resobj1.np << "\t\t" << resobj1.nn << "\n";
		cout << " ------------------------------\n"; 

		if(resobj1.pp !=0 || resobj1.np !=0)
			cout <<  " Precison  =" << static_cast<double>(resobj1.pp / static_cast<double>(resobj1.pp + resobj1.np)) << endl;
		if (resobj1.pp != 0 || resobj1.pn != 0)
			cout << " Recall  =" << static_cast<double> (resobj1.pp / static_cast<double>(resobj1.pp + resobj1.pn)) << endl;
		if (resobj1.pp != 0 || resobj1.np != 0 || resobj1.pn != 0)
			cout << " F1-Meauser  =" << static_cast<double>((2* resobj1.pp) / static_cast<double>((2*resobj1.pp) + resobj1.np+ resobj1.pn)) << endl;

	}

	// predict for testing data..
	vector<double> predictedtestOutput = predict(datasetNum, testInput, theta1, theta2);

	// measure test accuracy  ..
	results resobj2 = calculateAccuracy(datasetNum, testOutput, predictedtestOutput);
	//cout << " ----------------------------------------------------------------" << endl << endl;

	printf(" Testing Accurcy = %.2f%% \n", resobj2.accurcy * 100);


	// confusion matrix
	if (datasetNum == 2)
	{
		cout << " ------------------------------\n";
		cout << " Confusion Matrix for Testing Data : \n";
		cout << "\t p \t\t n \n";
		cout << " ------------------------------\n";
		cout << " p \t" << resobj2.pp << "\t\t" << resobj2.pn << "\n";
		cout << " n \t" << resobj2.np << "\t\t" << resobj2.nn << "\n";
		cout << " ------------------------------\n";

		if (resobj2.pp != 0 || resobj2.np != 0)
			cout << " Precison  =" << static_cast<double>(resobj2.pp / static_cast<double>(resobj2.pp + resobj2.np)) << endl;
		if (resobj2.pp != 0 || resobj2.pn != 0)
			cout << " Recall  =" << static_cast<double> (resobj2.pp / static_cast<double>(resobj2.pp + resobj2.pn)) << endl;
		if (resobj2.pp != 0 || resobj2.np != 0 || resobj2.pn != 0)
			cout << " F1-Meauser  =" << static_cast<double>((2 * resobj2.pp) / static_cast<double>((2 * resobj2.pp) + resobj2.np + resobj2.pn)) << endl;

	}	
	cout << " ----------------------------------------------------------------" << endl << endl;

	results obj;
	obj.trainingaccucy = resobj1.accurcy;
	obj.testingaccurcy = resobj2.accurcy;

	return obj;

}


// randInitializeWeights function to initialize theta 1 and theta 2 .. given the in and out dimension of  theta
vector<vector<double>> randInitializeWeights(int divect1, int divect2, double num1, double num2)
{
	vector <vector <double>> result(divect2, vector <double>(divect1 + 1, 1));
	
	// for parralleization 
	size_t size = divect2;
	concurrency::parallel_for(size_t(0), size, [&](size_t i)
	{
		for (size_t j = 0; j < divect1 + 1; j++)
		{
			// old method
			//result[i][j] = ((((double)rand() / (RAND_MAX)) * 2 * epsilon )- epsilon);

			// rand between -4 & 4 
			//result[i][j] = ((8* (double)rand() / (RAND_MAX)) -4);

			// rand between num1 & num2
			result[i][j] = (((num2-num1)* (double)rand() / (RAND_MAX)) + num1);
		}
	});
	
	return result;
}

// apply the activation function "sigmoid" on the given vector  .. uesd in forward propagation 
vector <vector<double>> sigmoid(vector <vector<double>> vect1)
{
	vector <vector <double>> result(vect1.size(), vector <double>(vect1[0].size(), 1));

	// for parralleization 
	size_t size = vect1.size();
	concurrency::parallel_for(size_t(0), size, [&](size_t i)
	{

		for (size_t j = 0; j < vect1[0].size(); j++)
		{
			result[i][j] = 1 / (1 + exp(-1 * (vect1[i][j])));
		}
	});

	return result;
}


// apply the dervative of sigmoid function ... used in backward propagation 
vector <vector<double>> gradientsigmoid(vector <vector<double>> vect1)
{
	vector <vector <double>> result(vect1.size(), vector <double>(vect1[0].size(), 1));

	// for parralleization 
	size_t size = vect1.size();

	// call sigmoid 
	result = sigmoid(vect1);
	concurrency::parallel_for(size_t(0), size, [&](size_t i)
	{
		for (size_t j = 0; j < vect1[0].size(); j++)
		{
			result[i][j] = result[i][j] * (1 - result[i][j]);
		}
	});

	return result;
}

// forward propagation function to calculate propalites of each output class labels for each example
vector < vector <double>> forwardPropagation(vector<vector<double>> features, vector <vector <double>> theta1, vector <vector <double>> theta2)
{
	// apply neural network steps ..
	vector < vector <double>> tt1(1, vector<double>(features[0].size(), 1));
	tt1[0] = features[0];
	vector <vector <double>> a1(1, vector <double>(1, 1));
	tt1= transpose(tt1);
	a1.insert(a1.end(), tt1.begin(), tt1.end());
	
	vector < vector <double>> z2 = mult(theta1,a1);

	vector < vector <double>> tt2(z2.size(), vector<double>(1, 1));
	tt2 = sigmoid(z2);
	vector <vector <double>> a2(1, vector <double>(1, 1));
	a2.insert(a2.end(), tt2.begin(), tt2.end());

	vector < vector <double>> z3 = mult(theta2,a2);
	vector < vector <double>> a3 = sigmoid(z3);

	// return propabilites ..
	return a3;


}

// function for calculate the cost given a set of parameters 
parameters costFunction(vector <vector<double>> featuresX, vector<double> output, int num_labels, vector <vector <double>> theta1, vector <vector <double>> theta2)
{
	double totalLoss = 0;

	// for parralleization over examples 
	size_t size = output.size();
	Concurrency::critical_section cs2;	
	concurrency::parallel_for(size_t(0), size, [&](size_t m)
	{
		vector < vector <double>> features(1, vector<double>(1, featuresX[m].size()));

		// retrive current example 
		features[0]= featuresX[m];

		// call forwardPropagation to get probability of each output class for this example 
		vector < vector <double>> a3 = forwardPropagation(features, theta1, theta2);		
		
		// resshape output to be in one hot vector shape 
		vector<double> outputvect(num_labels);
		//since datset 1 labeles are 1 &2 .. and dataset2 labels are 0 & 1
		// so outputvect[1]= 1 in dataset1 means the class is 2, where in dataset 2 outputvect[1]= 1 means the class is 0

		if (output[m] == 1)  // label =1 
			outputvect[0] = 1;
		else      // label = 2
			outputvect[1] = 1;

		// for parralleization over num of labels 
		for (size_t j = 0; j < num_labels; j++)
		{ 
			// calculate current loss based on the rule 
			double currentLoss = ( outputvect[j] * log(a3[j][0])) + ((1 - outputvect[j])*log(1 - a3[j][0]));

			// update total loss 			
			cs2.lock();
			totalLoss = totalLoss + currentLoss;
			cs2.unlock();

		}
	});

	// update total loss 		
	totalLoss = -1* totalLoss / output.size();

	// return results
	parameters obj;
	obj.loss = totalLoss;
	
	return obj;

}

// backpropagation function to calculate dervatives to  update weights ..
parameters backpropagation(int num_labels, vector<vector<double>> input, vector<double> output, vector<vector<double>>theta1, vector<vector<double>>theta2)
{
	// define weights .. theta 1 and 2 
	vector<vector<double>> theta1_grad(theta1.size(), vector<double>(theta1[0].size(), 0));
	vector<vector<double>> theta2_grad(theta2.size(), vector<double>(theta2[0].size(), 0));

	// for parralleization  over examples 
	size_t size = output.size();
	// for shared variables 
	Concurrency::critical_section cs;	
	concurrency::parallel_for(size_t(0), size, [&](size_t m)
	{
		// reshape output to be as one hot vector 
		vector<vector<double>> outputvect(num_labels, vector<double>(1, 0));
		//since datset 1 labeles are 1 &2 .. and dataset2 labels are 0 & 1
		// so outputvect[i][1]= 1 in dataset1 means the class is 2, where in dataset 2 outputvect[i][1]= 1 means the class is 0

		if (output[m] == 1)  // label =1 
			outputvect[0][0] = 1;
		else      // label = 2
			outputvect[1][0] = 1;

		// retreive current example 
		vector <vector <double>> example = transpose(getexample(input, m));

		// apply neural networks steps for calculate dervatives .. 
		vector <vector <double>> a1(1, vector <double>(1, 1));
		a1.insert(a1.end(), example.begin(), example.end());

		vector <vector <double>> z2 = mult(theta1, a1);
		vector <vector <double>> tm = sigmoid(z2);
		vector <vector <double>> a2(1, vector <double>(1, 1));
		a2.insert(a2.end(), tm.begin(), tm.end());

		vector <vector <double>> z3 = mult(theta2, a2);
		vector <vector <double>> a3 = sigmoid(z3);

		vector <vector <double>> delta_3 = a3 - outputvect;
		vector <vector <double>> temp1 = mult(transpose(theta2), delta_3);
		vector <vector <double>> tvect2 = gradientsigmoid(z2);
		vector <vector <double>> temp2(1, vector <double>(1, 1));
		temp2.insert(temp2.end(), tvect2.begin(), tvect2.end());
		vector<vector <double>> delta_2 = temp1 * temp2; ;
		delta_2.erase(delta_2.begin());

		// update dervatives ..
		cs.lock();
		theta1_grad = theta1_grad + mult(delta_2, transpose(a1));
		theta2_grad = theta2_grad + mult(delta_3, transpose(a2));
		cs.unlock();

		// clear vectors to use in next example 
		example.clear(); outputvect.clear();
		a1.clear(); a2.clear(); a3.clear();
		temp1.clear(); temp2.clear(); tm.clear(); tvect2.clear();
		z2.clear(); z3.clear();
		delta_2.clear(); delta_3.clear();
		
	});

	// calculate average dervative ..
	theta1_grad = theta1_grad / output.size();
	theta2_grad = theta2_grad / output.size();

	// return results 
	parameters obj;
	obj.theta1_grad = theta1_grad;
	obj.theta2_grad = theta2_grad;

	return obj;

}

// update weights based on backpropagation feedback ..
parameters updateParameters(vector<vector<double>> theta1, vector<vector<double>>theta2, vector<vector<double>> theta1_grad, vector<vector<double>>theta2_grad, double learning_rate)
{
	parameters updatedParam;
	updatedParam.theta_1 = theta1 - (theta1_grad * learning_rate);
	updatedParam.theta_2 = theta2 - (theta2_grad * learning_rate);

	return updatedParam;
}

// predict classes using the created NN model .. 
vector<double> predict(int datasetNum,vector <vector<double>> featuresX, vector <vector <double>> theta1, vector <vector <double>> theta2)
{
	size_t size = featuresX.size();
	vector<double> predictedOutput(size, 1);
	
	// for parralleization over examples ..
	concurrency::parallel_for(size_t(0), size, [&](size_t m)
	{
		vector < vector <double>> features(1, vector<double>( featuresX[m].size(),1));

		// current example 
		features[0]= featuresX[m];

		//  find probabilites using forward propagtion model given the tuned weights ..
		vector < vector <double>> a3 = forwardPropagation(features, theta1, theta2);	

		//take the  index of largest probabilty 
		if (datasetNum == 1)
		{
			if (a3[0][0] > a3[1][0])
				predictedOutput[m] = 1;
			else
				predictedOutput[m] = 2;
		}
		else if (datasetNum == 2)
		{
			if (a3[0][0] > a3[1][0])
				predictedOutput[m] = 0;
			else
				predictedOutput[m] = 1;

		}
	});

	return  predictedOutput;
}

