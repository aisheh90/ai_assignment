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

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;

#include "svm_header.h"

// convert 2d vector data into MAT type to be suitable for SVM in opencv library 
Mat convert2dVecttoMat(vector<vector<double>> vect)
{
	Mat mat(vect.size(), vect[0].size(), CV_64F);
	for (size_t i = 0; i < vect.size(); i++)
	{
		for (size_t j = 0; j < vect[0].size(); j++)
		{
			mat.at<double>(i, j) = vect[i][j];
		}
	}

	return mat;
}

// convert 1d vector data into MAT type to be suitable for SVM in opencv library 
Mat convert1dVecttoMat(vector<double> vect)
{
	Mat mat(vect.size(), 1, CV_64F);
	for (size_t i = 0; i < vect.size(); i++)
	{
		
		mat.at<double>(i) = vect[i];
		
	}

	return mat;
}

// predict class of each example in the data using SVM
Mat svmPredict(Ptr<SVM> svm, Mat datatmat)
{
	// MAT type to store predictions call it predictedtest
	cv::Mat predictedtest(datatmat.rows, 1, CV_32S);  // CV_32S means datatype 32 scalar .. opencv accept only 32 scalar of float (CV_32F)

	// predict for each example
	for (int i = 0; i < datatmat.rows; i++)
	{		
		// current example 
		cv::Mat sampletest = datatmat.row(i);

		// define datatype to store current prediction in it
		cv::Mat responsetest(1, 1, CV_32S);

		// prediction
		responsetest = svm->predict(sampletest);

		// store current prediction in predictedtest
		predictedtest.at<int>(i, 0) = responsetest.at<int>(0, 0);

	}
	
	return predictedtest;

}


// Calcualte accuarcy of SVM prediction 
svmRes calcSVMaccurcy(Mat actuallabelsmat,  Mat predicted, int datasetNum)
{
	// different measures to calcuate  
	// # of correct predictions, # of wrong predictions 
	// & confusion matrix components pp ,nn, pn, np
	int correct = 0, wrong = 0;
	int pp = 0, nn = 0, pn = 0, np = 0;
	
	// for eacn example ,, compare actual class with predicted class
	for (int i = 0; i < actuallabelsmat.rows; i++)
	{
		int p = predicted.at<int>(i, 0);
		int a = actuallabelsmat.at<int>(i, 0);

		if (p == a)
		{
			correct++;
		}
		else
		{
			wrong++;
		}

		// for confusion matrix 
		if (datasetNum == 2) // classes 0 &1 
		{

			if (p == a && p == 1) pp++;
			else if (p == a && p == 0) nn++;
			else if (p != a && a == 1) pn++;
			else if (p != a && a == 0) np++;

		}
	}

	double accu = 100 * correct / (correct + wrong);
	svmRes res;
	res.acc = accu;
	res.pp = pp; res.nn = nn; res.pn = pn; res.np = np;
	return res;
}


