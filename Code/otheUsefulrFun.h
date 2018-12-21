#pragma once
#include<iostream>
#include <string>
#include <vector>

using namespace std;

// used in store datadset info
typedef struct dataset
{
	vector <vector<double > > input;
	vector <double > output;
	int featureNum;

}dataset;

// used in store each algorithm results
typedef struct results
{
	double trainingaccucy;
	double testingaccurcy;
	double accurcy;

	// for confusion matrix
	int pp;
	int nn;
	int pn;
	int np;

}results;


// ready code for read data from csv file 
class CSVRow
{
public:
	string const& operator[](size_t index) const
	{
		return m_data[index];
	}
	size_t size() const
	{
		return m_data.size();
	}
	//read whole file lines
	void readNextRow(istream& str)
	{
		// read lines by line 
		string         line;
		getline(str, line);

		stringstream   lineStream(line);
		string         cell;

		m_data.clear();
		// split based on "," delimter 
		while (getline(lineStream, cell, ','))
		{
			m_data.push_back(cell);
		}
		// reach end of the file
		if (!lineStream && cell.empty())
		{

			m_data.push_back("");
		}
	}
private:
	vector<string>    m_data;
};

// readData function to read specific file based on the given dataset number 
dataset readData(int datasetNum);

// transpose matrix function 
vector <vector <double>> transpose(vector<vector<double>> vect1);

// add column of ones to the matrix in the begining 
vector <vector<double>> vectWithOnes(vector <vector<double>> vect1);

// retreive specific example form the data based on given index
vector <vector <double>> getexample(vector <vector <double>> input, int index);

// shuffle data indices to randomize examples places 
vector <int> suffledataIndices( vector <double> vect);

// normalize data to make all values between 0 and  1
vector<vector<double>> normalizedata(vector<vector<double>> data);

// calculate Accuracy function used in NN and DT 
results calculateAccuracy(int datasetNum, vector <double> actualOutput, vector <double> predictedOutput);