#pragma once
#include<iostream>
#include <string>
#include <vector>
#include "otheUsefulrFun.h"
#include "overwritesHeader.h"

using namespace std;


// DTPaths to store created paths by the DT
extern vector<vector<double>> DTPaths;

// struct to store info of each feature: unique values of the feature,
// and number of class 1 and class 2 for associated with each value
typedef struct featureValueInfo
{
	double uniquVal;
	double numofclass1;
	double numofclass2;

	featureValueInfo()
	{
		numofclass1 = 0;
		numofclass2 = 0;
	}


}featureValueInfo;

// struct to store info of all features: number of unique values of each feature,
// and number of class 1 and class 2 for associated with each value of each feature
typedef struct featuresInformation
{
	int numofDistinctValues;
	int tnumofclass1;
	int tnumofclass2;
	vector <featureValueInfo> uniquValsInfo;

	featuresInformation()
	{
		 numofDistinctValues =0;
		 tnumofclass1= 0;
		 tnumofclass2= 0;
	}

}featuresInformation;


// functions needed for DT algorithm 

// descretizeData fucntion used to descretize feature values when its continous data into 2-ways based on midpoint with heighest gain
vector <vector<double>> descretizeData(int datasetNum, vector <vector<double>> inputdata, vector<double>outputdata);

// calcGainForOneFeature function used in descretizeData function for calculating each midpoint gain
double calcGainForOneFeature(int datasetNum, vector<double>rowdata, vector<double>outdata);

// findfeaturesinfo function to find information for each feature 
vector <featuresInformation> findfeaturesinfo(int datasetNum, vector<vector <double>> trainInput, vector<double>trainOutput);

// simpleGain function calculate gain of single value of a feature 
double simpleGain(int numofclass1, int numofclass2);

// calcGain fucntion calculate gain for each feature by calling simpleGain function  for each single value of that feature 
vector <double> calcGain(int datasetNum, vector<double> data, vector <featuresInformation> featuresInfoVect);

// findHighestGain function return index of the feature with heighest gain
int findHighestGain(vector <double> featuresGain);

// createTree function create the decison tree (DT) 
void createTree(int datasetNum, vector<vector<double>> data, vector<double>output, string str, int numOftotalFeatures, int numOfUsedFeatures, vector<int> fetauresNumsVect, double curFrVal, bool dtfailedsplitbranchmethod);

// addpathtotree fucntion add new path to the DT 
void addpathtotree(string str);

// predictDT function used to predict class of each exampel in the data usign the created decsion tree 
vector<double> predictDT(int datasetNum, vector<vector<double>> input);

