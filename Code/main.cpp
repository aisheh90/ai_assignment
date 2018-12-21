#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <ppl.h>
#include <windows.h>
#include <cstdio>

#include "concurrent_vector.h"
#include "overwritesHeader.h"
#include "otheUsefulrFun.h"
#include "nn_header.h"
#include "svm_header.h"
#include "dt_header.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace concurrency;



void main()
{
	/************************************************* Reading Data **************************************/

	cout << "\n -------------------- Welcome to my AI project --------------------------" << endl << endl;
	int datasetNum;
	cout << " Please deterimne which dataset you want to work on?" << endl;
	cout << " 1 for Mesothelioma Dataset \n 2 for Diabetic Dataset" << endl;
	cout << " Your choice is : ";
	cin >> datasetNum;
	cout << " \n ------------------------------------------------------------------------" << endl << endl;

	// check required dataset to work 
	dataset obj = readData(datasetNum);

	vector<vector<double>> input = obj.input;
	vector<double> output = obj.output;
	int featureNum = obj.featureNum;


	cout << " Dataset " << datasetNum << "  Info: " << endl;
	cout << " Number of input examples = " << input.size() << endl;
	cout << " Number of input features = " << input[0].size() << endl;
	cout << " \n ------------------------------------------------------------------------" << endl << endl;



	/*******************************************Choose algorithm *****************************************/

	int algoNum;
	bool normalizedataflag;

	cout << " Please deterimne which classification algorithm you want to use?" << endl;
	cout << " 1 for Neural Network \n 2 for Support vector machine \n 3 for Decision tree " << endl;
	cout << " Your choice is : ";
	cin >> algoNum;

	cout << " \n ------------------------------------------------------------------------" << endl << endl;


	//a set of Neural Network paramters to use later (1 hidden layer)
	int	hidden_layer_size; // # i.e. of hidden units 			
	double learning_rate; //1.2;
	double num1, num2; // for rand intilizae weights between num1 and num2
	int numofIterations;
	int printitr;// print after each 10 iterations


	//for svm latter
	int svmkernalType;
	if (algoNum == 2)
	{
		cout << " Please choose  SVM kernal type: " << endl;
		cout << " 1 for linear kernal \n";
		cout << " 2 for Radial basis function (RBF) kernal \n";
		cout << " 3 for Sigmoid kernal \n";
		cout << " Your choice : ";
		cin >> svmkernalType;
		cout << " \n ------------------------------------------------------------------------" << endl << endl;


	}
	
	

	// for DT latter
	bool dtfailedsplitbranchmethod;

	/******************************************* preprocess data  *****************************************/
	
	// normalizedata for NN & SVM
	if (algoNum == 1|| algoNum == 2) // normalization improve results much.. to cancel effect of fetaures high values on other features small values
	{

		cout << "With Data normalization? 1 For yes  ... 0 for NO \n Your choice: ";
		cin >> normalizedataflag;
		cout << endl;

		if (normalizedataflag)
		{
			input = normalizedata(input);
		}



	}
	if (algoNum == 3)// for DT .. since its not efficent and hard to work on continous data in split features values 
	{
		input = descretizeData(datasetNum,input,output);

	}

	int numOfFolds;
	if (datasetNum == 1)
	{
		cout << " Please deterimne number of folds for this dataset (must >1 )" << endl;
		cout << " Number of folds = ";
		cin >> numOfFolds;
		
		
	}

	//************ shuffle data (i.e. randomize examples places)*************
	vector <int> indexes =  suffledataIndices(output);
	vector < vector<double>> newInput;
	vector<double> newOutput;

	for (int ind = 0; ind < indexes.size(); ind++)
	{
		newInput.push_back(input[indexes[ind]]);
		newOutput.push_back(output[indexes[ind]]);
	}


	
	/******************************************* split data into training & test  ... then apply required algoriothm *****************************************/
	
	vector<vector<double>> trainInput;
	vector<vector<double>> testInput;
	vector<double> trainOutput;
	vector<double> testOutput;


	if (datasetNum == 1 ) // divide it in folds
	{
		double averageTrainAccurcy = 0;
		double averageTestAccurcy = 0;
		
		// 5-folds
		
		int foldsize = floor(newInput.size()/ numOfFolds);
		
		for (int f = 0; f < numOfFolds; f++) 
		{
			//clear data from previous folds
			trainInput.erase(trainInput.begin(), trainInput.end());
			trainOutput.erase(trainOutput.begin(), trainOutput.end());
			testInput.erase(testInput.begin(), testInput.end());
			testOutput.erase(testOutput.begin(), testOutput.end());

			//test data
			for (int j = f * foldsize; j < (f*foldsize + foldsize); j++)
			{
				 testInput.push_back(newInput[j]);
				 testOutput.push_back(newOutput[j]);
			}


			//train data
			for (int j = 0; j< (f * foldsize); j++)
			{
				trainInput.push_back(newInput[j]);
				trainOutput.push_back(newOutput[j]);
			}
			for (int j = (f*foldsize + foldsize); j < newOutput.size(); j++)
			{
				trainInput.push_back(newInput[j]);
				trainOutput.push_back(newOutput[j]);
			}

			
			// apply neural network for current fold 
			if (algoNum == 1)
			{ 
				

				if (f == 0)
				{
					cout << " Please set the following Neural network parameters : " << endl;
					cout << " Number of hidden units = ";
					cin >> hidden_layer_size;
					cout << " Learning rate = ";
					cin >> learning_rate;
					cout << " Initialize weights between two numbers :" << endl;
					cout << " Num1 = "; cin >> num1;
					cout << " Num2 = "; cin >> num2;
					cout << " Number of iterations = ";					
					cin >> numofIterations;
					cout << " Print results after what number of iterations? ";
					cin >> printitr;
					cout << "-------------------------------------------------" << endl;

				}


				//cout <<" ------------------------------------------------------------ \n\n";
				cout << "fold #" << f+1 << endl;

				
				// call main neural network function to run on cuurent data fold
				results obj = neuralNetwork(datasetNum, featureNum, trainInput, trainOutput, testInput, testOutput, hidden_layer_size, learning_rate, numofIterations, printitr, num1, num2);

				// retreive training and testing accurcies for current fold 
				// avergae results print statements below (common for all algorithms)
				averageTrainAccurcy = averageTrainAccurcy + obj.trainingaccucy*100;
				averageTestAccurcy = averageTestAccurcy + obj.testingaccurcy*100;


			}
			else if(algoNum == 2) // apply SVM for current fold
			{
				cout << "Fold #" << f+1 << endl;

				//convert train data from vector to mat
				cv::Mat trainmat(trainInput.size(), trainInput[0].size(), CV_32F);
				trainmat = convert2dVecttoMat(trainInput);
				
				cv::Mat trainlabelsmat(trainOutput.size(), 1, CV_32F);
				trainlabelsmat = convert1dVecttoMat(trainOutput);
					
				// convert data type since opencv only accept 32 float or scalar
				trainmat.convertTo(trainmat, CV_32F);
				trainlabelsmat.convertTo(trainlabelsmat, CV_32S);

			    //  create SVM object ..
				Ptr<SVM> svm = SVM::create();
				svm->setType(SVM::C_SVC);

				// specify some parameters
				if (svmkernalType == 1)
				{
					svm->setKernel(SVM::LINEAR);
				}
				else if (svmkernalType == 2)
				{
					svm->setKernel(SVM::RBF);// Radial basis function kernel (most common used)
				}
				else if (svmkernalType == 3)
				{
					svm->setKernel(SVM::SIGMOID);
				}
				
				// this to terminate SVM and don't stuck until it find the hyperlane that seprate exactly 
				// i.e. not necessary the optimal solution
				svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6)); 

				// Train the SVM ...
				svm->train(trainmat, ROW_SAMPLE, trainlabelsmat);
				
				// predict using svm 
				cv::Mat predicted(trainmat.rows, 1, CV_32S);
				predicted = svmPredict(svm, trainmat);
				
				// find train accuracy of current fold 
				
				svmRes currentTrainobj = calcSVMaccurcy(trainlabelsmat, predicted,datasetNum);
				printf(" Training Accurcy = %.2f%% \n", currentTrainobj.acc);

				// update average train accuacry 
				averageTrainAccurcy = averageTrainAccurcy + currentTrainobj.acc;

				// test data 
				//convert test data from vector to mat
				cv::Mat testmat(testInput.size(), testInput[0].size(), CV_32F);
				testmat = convert2dVecttoMat(testInput);

				cv::Mat testlabelsmat(testOutput.size(), 1, CV_32F);
				testlabelsmat = convert1dVecttoMat(testOutput);

				// convert data type since opencv only accept 32 float or scalar
				testmat.convertTo(testmat, CV_32F);
				testlabelsmat.convertTo(testlabelsmat, CV_32S);

				// predict using svm 
				cv::Mat predictedtest(testmat.rows, 1, CV_32S);
				predictedtest = svmPredict(svm, testmat);

				// find test accuracy of current fold 
				svmRes currentTestobj = calcSVMaccurcy(testlabelsmat, predictedtest, datasetNum);
				printf(" Testing Accurcy = %.2f%% \n", currentTestobj.acc);

				// update average test accuacry 
				averageTestAccurcy = averageTestAccurcy + currentTestobj.acc;

			}
			else if (algoNum == 3) //apply DT for current fold 
			{
				if (f == 0)
				{
					// some DT settings
					cout << " When reach in a branch to a point that can not split, what the method you want to use to give that branch a class value? " << endl;
					cout << " 1 fo method 1: give it the value of the positive class \n";
					cout << " 2 for method 2: give it random class value \n";
					cout << " 3 for method 3: give it the valus of the class with most examples in that branch \n";
					cout << " Your choice : ";
					cin >> dtfailedsplitbranchmethod;
					cout << " ------------------------------\n";

				}
							   				 			  

				cout << " Fold #: " << f+1 << endl;

				// prepare needed variables and data to DT 
				int numOftotalFeatures = trainInput[0].size();
				int numOfUsedFeatures = 0;
				string str = ""; // to store path gradually in it
				int prevFeatureLoc = 0;
				// the following just to keep track of fetaures used in splitting
				vector<int>fetauresNumsVect(trainInput[0].size());
				for (int i = 0; i < trainInput[0].size(); i++)
				{
					fetauresNumsVect[i] = i;
				}
				double curFrVal = NAN;

				// create DT 
				createTree(datasetNum, trainInput, trainOutput, str, numOftotalFeatures, numOfUsedFeatures, fetauresNumsVect, curFrVal, dtfailedsplitbranchmethod);

				// predict for training data 
				vector<double> predictedOutput = predictDT(datasetNum, trainInput);

				// measure train accuracy
				results resobj1 = calculateAccuracy(datasetNum, trainOutput, predictedOutput);

				printf(" Training Accurcy = %.2f%% \n", resobj1.accurcy * 100); 
				averageTrainAccurcy = averageTrainAccurcy + resobj1.accurcy*100; 


				// for testing 
				// predict for test data 
				vector<double> predictedtestOutput = predictDT(datasetNum, testInput);

				// measure test accuracy
				results resobj2 = calculateAccuracy(datasetNum, testOutput, predictedtestOutput);

				printf(" Testing Accurcy = %.2f%% \n", resobj2.accurcy * 100);
				averageTestAccurcy= averageTestAccurcy+ resobj2.accurcy*100; 

				DTPaths.erase(DTPaths.begin(), DTPaths.end());
				cout << " ----------------------------------------------------------------" << endl << endl;

			}
			else 
			{
				cout << " Invalid algorithm number -_- " << endl;
			}
		}

		// average results for all folds
		cout << " ------------------------------------------------------------ \n\n";
		averageTrainAccurcy = averageTrainAccurcy / numOfFolds;
		averageTestAccurcy = averageTestAccurcy / numOfFolds;
		printf(" Average Training Accurcy = %.2f %% \n", averageTrainAccurcy);
		printf(" Average Testing Accurcy = %.2f %% \n", averageTestAccurcy);

	}
	else if(datasetNum == 2) //datasetNum =2
	{
		// 80% as training data .. 20% as testing data
		int trainPercent =  floor(0.80 * newInput.size());
		
		trainInput = newInput;
		trainOutput = output;

		// train data ..
		for (int t = 0; t < trainPercent; t++)
		{
			trainInput.push_back(newInput[t]);
			trainOutput.push_back(output[t]);
		}

		// test data ..
		for (int t = trainPercent; t < newInput.size(); t++)
		{
			testInput.push_back(newInput[t]);
			testOutput.push_back(output[t]);
		}

		if (algoNum == 1) //neural network
		{			
					
			cout << " Please set the following Neural network parameters : " << endl;
			cout << " Number of hidden units = ";
			cin >> hidden_layer_size;
			cout << " Learning rate = ";
			cin >> learning_rate;
			cout << " Initialize weights between two numbers :" << endl;
			cout << " Num1 = "; cin >> num1;
			cout << " Num2 = "; cin >> num2;
			cout << " Number of iterations = ";
			cin >> numofIterations;
			cout << " Print results after what number of iterations? ";
			cin >> printitr;
			
			cout << "-------------------------------------------------" << endl;

			// call main neural network function to run on the data				
			results obj = neuralNetwork(datasetNum, featureNum, trainInput, trainOutput, testInput, testOutput,	hidden_layer_size, learning_rate, numofIterations, printitr, num1, num2);


			// training and testing accurcies are stored in obj 
			// results print statements already in the same nn function
		}
		else if (algoNum == 2) // apply SVM on dataset 2
		{

			//convert train data from vector to mat
			cv::Mat trainmat(trainInput.size(), trainInput[0].size(), CV_32F);
			trainmat = convert2dVecttoMat(trainInput);

			cv::Mat trainlabelsmat(trainOutput.size(), 1, CV_32F);
			trainlabelsmat = convert1dVecttoMat(trainOutput);

			// convert data type since opencv only accept 32 float or scalar
			trainmat.convertTo(trainmat, CV_32F);
			trainlabelsmat.convertTo(trainlabelsmat, CV_32S);

			//  create SVM object ..
			Ptr<SVM> svm = SVM::create();
			svm->setType(SVM::C_SVC);

			// set some parameters
			if (svmkernalType == 1)
			{
				svm->setKernel(SVM::LINEAR);
			}
			else if (svmkernalType == 2)
			{
				svm->setKernel(SVM::RBF);// Radial basis function kernel (most common used)
			}
			else if (svmkernalType == 3)
			{
				svm->setKernel(SVM::SIGMOID);
			}
			

			// this to terminate SVM and don't stuck until it find the hyperlane that seprate exactly 
			// i.e. not necessary the optimal solution
			svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
						
			// Train the SVM ...
			svm->train(trainmat, ROW_SAMPLE, trainlabelsmat);

			// predict using svm 
			cv::Mat predicted(trainmat.rows, 1, CV_32S);
			predicted = svmPredict(svm, trainmat);

			// find train accuracy 
			svmRes svmresl = calcSVMaccurcy(trainlabelsmat, predicted ,datasetNum);
			printf(" Training accurcy = %.2f%% \n", svmresl.acc);

			// print confusion matrix for training data
			cout << " ------------------------------\n";
			cout << " Confusion Mtraix for Training Data : \n";
			cout << "\t p \t\t n \n";
			cout << " ------------------------------\n";
			cout << " p \t" << svmresl.pp << "\t\t" << svmresl.pn << "\n";
			cout << " n \t" << svmresl.np << "\t\t" << svmresl.nn << "\n";
			cout << " ------------------------------\n";

			if (svmresl.pp != 0 || svmresl.np != 0)
				cout << " Precison  =" << static_cast<double>(svmresl.pp / static_cast<double>(svmresl.pp + svmresl.np)) << endl;
			if (svmresl.pp != 0 || svmresl.pn != 0)
				cout << " Recall  =" << static_cast<double> (svmresl.pp / static_cast<double>(svmresl.pp + svmresl.pn)) << endl;
			if (svmresl.pp != 0 || svmresl.np != 0 || svmresl.pn != 0)
				cout << " F1-Meauser  =" << static_cast<double>((2 * svmresl.pp) / static_cast<double> ((2 * svmresl.pp) + svmresl.np + svmresl.pn)) << endl;


			// test data 
			//convert test data from vector to mat
			cv::Mat testmat(testInput.size(), testInput[0].size(), CV_32F);
			testmat = convert2dVecttoMat(testInput);

			cv::Mat testlabelsmat(testOutput.size(), 1, CV_32F);
			testlabelsmat = convert1dVecttoMat(testOutput);

			// convert data type since opencv only accept 32 float or scalar
			testmat.convertTo(testmat, CV_32F);
			testlabelsmat.convertTo(testlabelsmat, CV_32S);

			// predict using svm 
			cv::Mat predictedtest(testmat.rows, 1, CV_32S);
			predictedtest = svmPredict(svm, testmat);

			// find test accuracy 
			svmRes svresult = calcSVMaccurcy(testlabelsmat, predictedtest, datasetNum);
			cout << " ----------------------------------------------------------------" << endl << endl;
			printf(" Testing accurcy = %.2f%% \n", svresult.acc);

			// print confusion matrix for testing data
			cout << " ------------------------------\n";
			cout << " Confusion Mtraix for Testing Data : \n";
			cout << "\t p \t\t n \n";
			cout << " ------------------------------\n";
			cout << " p \t" << svresult.pp << "\t\t" << svresult.pn << "\n";
			cout << " n \t" << svresult.np << "\t\t" << svresult.nn << "\n";
			cout << " ------------------------------\n";

			if (svresult.pp != 0 || svresult.np != 0)
				cout << " Precison  =" << static_cast<double>(svresult.pp / static_cast<double>(svresult.pp + svresult.np)) << endl;
			if (svresult.pp != 0 || svresult.pn != 0)
				cout << " Recall  =" << static_cast<double> (svresult.pp / static_cast<double>(svresult.pp + svresult.pn)) << endl;
			if (svresult.pp != 0 || svresult.np != 0 || svresult.pn != 0)
				cout << " F1-Meauser  =" << static_cast<double>((2 * svresult.pp) / static_cast<double> ((2 * svresult.pp) + svresult.np + svresult.pn)) << endl;

		}
		else if (algoNum == 3) // apply  DT on dataset 2
		{
			// some DT settings
			cout << " When reach in a branch to a point that can not split, what the method you want to use to give that branch a class value? " << endl;
			cout << " 1 fo method 1: give it the value of the positive class \n";
			cout << " 2 for method 2: give it random class value \n";
			cout << " 3 for method 3: give it the valus of the class with most examples in that branch \n";
			cout << " Your choice : ";
			cin >> dtfailedsplitbranchmethod;

			cout << " ------------------------------\n";


			// prepare needed variables and data to DT 
			int numOftotalFeatures = trainInput[0].size();
			int numOfUsedFeatures = 0;
			string str = ""; // to store path gradually in it
			int prevFeatureLoc = 0;
			// the following just to keep track of fetaures used in splitting
			vector<int>fetauresNumsVect(trainInput[0].size());
			for (int i = 0; i < trainInput[0].size(); i++)
			{
				fetauresNumsVect[i] = i;
			}
			double curFrVal = NAN;

			// create DT 
			createTree(datasetNum, trainInput, trainOutput, str, numOftotalFeatures, numOfUsedFeatures, fetauresNumsVect, curFrVal, dtfailedsplitbranchmethod);
			
			// predict for training data 
			vector<double> predictedOutput = predictDT(datasetNum, trainInput);

			// measure train accuracy
			results resobj1 = calculateAccuracy(datasetNum, trainOutput, predictedOutput);
			cout << " ----------------------------------------------------------------" << endl << endl;
			printf(" Training accurcy = %.2f%% \n", resobj1.accurcy * 100);

			
			// confusion matrix for train 			
			cout << " ------------------------------\n";
			cout << " Confusion Mtraix for Training Data : \n";
			cout << "\t p \t\t n \n";
			cout << " ------------------------------\n";
			cout << " p \t" << resobj1.pp << "\t\t" << resobj1.pn << "\n";
			cout << " n \t" << resobj1.np << "\t\t" << resobj1.nn << "\n";
			cout << " ------------------------------\n";

			if (resobj1.pp != 0 || resobj1.np != 0)
				cout << " Precison  =" << static_cast<double>(resobj1.pp / static_cast<double>(resobj1.pp + resobj1.np)) << endl;
			if (resobj1.pp != 0 || resobj1.pn != 0)
				cout << " Recall  =" << static_cast<double> (resobj1.pp / static_cast<double>(resobj1.pp + resobj1.pn)) << endl;
			if (resobj1.pp != 0 || resobj1.np != 0 || resobj1.pn != 0)
				cout << " F1-Meauser  =" << static_cast<double>((2 * resobj1.pp) / static_cast<double>((2 * resobj1.pp) + resobj1.np + resobj1.pn)) << endl;
			
		

			// for testing 
			// predict for test data 
		    vector<double> predictedtestOutput = predictDT(datasetNum, testInput);

			// measure test accuracy
			results resobj2 = calculateAccuracy(datasetNum, testOutput, predictedtestOutput);

			cout << " ----------------------------------------------------------------" << endl << endl;
			printf(" Testing accurcy = %.2f%% \n", resobj2.accurcy * 100);

			// confusion matrix for test 			
			cout << " ------------------------------\n";
			cout << " Confusion Mtraix for Testing Data : \n";
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
		else
		{
			cout << " Invalid algorithm number -_- " << endl;
		}

	}
	
	system("pause");

}