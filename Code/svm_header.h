#pragma once
#include<iostream>
#include <string>
#include <vector>

using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;

// strudct as object for storing svm results 
typedef struct svmRes
{
	double acc;
	int pp;
	int pn;
	int np;
	int nn;

}svmRes;

// convert 2d vector data into MAT type to be suitable for SVM in opencv library 
cv::Mat convert2dVecttoMat(vector<vector<double>> vect);

// convert 1d vector data into MAT type to be suitable for SVM in opencv library
cv::Mat convert1dVecttoMat(vector<double>vect);

// predict class of each example in the data using SVM
cv::Mat svmPredict(Ptr<SVM> svm, Mat testmat);

// Calcualte accuarcy of SVM prediction 
svmRes calcSVMaccurcy(Mat actuallabelsmat, Mat predicted, int datasetNum);

