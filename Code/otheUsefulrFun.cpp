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

using namespace std;
using namespace concurrency;

// to read row by row
istream& operator>>(istream& str, CSVRow& data)
{
	data.readNextRow(str);
	return str;
}


// readData function to read specific file based on the given dataset number 
dataset readData(int datasetNum)
{
	ifstream  file;
	CSVRow   row;
	int featureNum = 0;
	int counter = 0;

	vector<vector<double>> input;
	vector<double> output;

	// check required dataset to work 
	if (datasetNum == 1)
	{
		file.open("dataset1.csv");
	}
	else if (datasetNum == 2)
	{
		file.open("dataset2.csv");

	}
	else
	{
		cout << " Sorry.. invalid dataset number " << endl;
	
	}


	// read the required file 
	while (file >> row)
	{
		counter++;
		vector<double> currentRow;

		if (counter == 1)
		{
			featureNum = row.size() - 1;
			if (datasetNum == 1)
				continue; // ignore header row if dataset 1 only .. dataset2 hasent header

		}

		// store rows 
		for (int i = 0; i < featureNum; i++)
		{
			currentRow.push_back(stod(row[i]));
		}

		// split data into input & output
		input.push_back(currentRow);
		output.push_back(stod(row[featureNum]));

	}

	// return datset info
	dataset obj;
	obj.input = input;
	obj.output = output;
	obj.featureNum = featureNum;
	return obj;
}

// transpose matrix function 
vector <vector <double>> transpose(vector<vector<double>> vect1)
{

	vector <vector <double>> result(vect1[0].size(), vector <double>(vect1.size(), 1));
	// for paralleziation 
	size_t size = vect1.size();
	concurrency::parallel_for(size_t(0), size, [&](size_t i)
	{

		for (size_t j = 0; j < vect1[0].size(); j++)
		{

			result[j][i] = vect1[i][j];
		}
	});

	return result;
}

// add column of ones to the matrix in the begining 
vector <vector<double>> vectWithOnes(vector <vector<double>> vect1)
{
	// intilize a new vect with suitable size all to ones 
	vector <vector <double>> result(vect1.size(), vector <double>(1 + vect1[0].size(), 1));

	// for paralleziation 
	size_t size = vect1.size();
	// override columns by original vect values starting from column 2 (i.e. co,umn 1 remains ones)
	concurrency::parallel_for(size_t(0), size, [&](size_t i)
	{

		for (size_t j = 0; j < vect1[0].size(); j++)
		{

			result[i][j + 1] = vect1[i][j];
		}
	});

	return result;
}

// retreive specific example form the data based on given index
vector <vector <double>> getexample(vector <vector <double>> input, int index)
{
	vector <vector <double>> exa(1, vector <double>(input[index].size(), 1));
	for (int j = 0; j < input[index].size(); j++)
	{
		exa[0][j] = input[index][j];
	}

	return exa;
}

// shuffle data indices to randomize examples places 
vector <int> suffledataIndices(vector <double> vect)
{
	vector<int> indexes;
	indexes.reserve(vect.size());
	for (int i = 0; i < vect.size(); i++)
	{
		indexes.push_back(i);
	}
	// ready function in c++ 
	random_shuffle(indexes.begin(), indexes.end());

	return indexes;

}

// normalize data to make all values between 0 and  1
vector<vector<double>> normalizedata(vector<vector<double>> data)
{
	vector<vector<double>> newdata(data.size(), vector<double>(data[0].size(), 1));
	vector<double> minvect;
	vector<double> maxvect;

	// work on each feature .. find its min and max 
	for (int i = 0; i < data[0].size(); i++)
	{
		double min = data[0][i];
		double max = data[0][i];

		for (int j = 0; j < data.size(); j++)
		{
			if (data[j][i] < min)
				min = data[j][i];
			else if (data[j][i] > max)
				max= data[j][i];


		}

		minvect.push_back(min);
		maxvect.push_back(max);

	}

	// normalize values of each feature to be between 0 and 1
	// new value = (old value - min value)/(max value - min value) 
	for (int i = 0; i < data[0].size(); i++)
	{

		for (int j = 0; j < data.size(); j++)
		{
			double minn = minvect[i];
			double maxx = maxvect[i];

			double normalizedValue = (data[j][i] - minn) / (maxx - minn);

			newdata[j][i] = normalizedValue;

		}

	}

	return newdata;
}

// calculate Accuracy function used in NN and DT 
results calculateAccuracy(int datasetNum, vector <double> actualOutput, vector <double> predictedOutput)
{

	double accurcy = 0;
	concurrency::critical_section cs3;
	size_t size = actualOutput.size();
	int pp = 0, nn = 0, pn = 0, np = 0;

	// for paralleziation 
	concurrency::parallel_for(size_t(0), size, [&](size_t m)
	{
		// compare actual class with predicted class.. 
		if (actualOutput[m] == predictedOutput[m])
		{
			cs3.lock();
			accurcy = accurcy + 1;  // since its a shared variable
			cs3.unlock();

		}

		// for confusion matrix 
		if (datasetNum == 2) // clasess are  0 (negative) & 1 (positive)
		{
			if ((actualOutput[m] == predictedOutput[m]) && (actualOutput[m] == 1))  pp++;
			else if ((actualOutput[m] == predictedOutput[m]) && (actualOutput[m] == 0))  nn++;
			else if ((actualOutput[m] != predictedOutput[m]) && (actualOutput[m] == 1))  pn++;
			else if ((actualOutput[m] != predictedOutput[m]) && (actualOutput[m] == 0))  np++;

		}
	});


	accurcy = accurcy / actualOutput.size();
	results obj;
	obj.accurcy = accurcy;

	// for confusion matrix  dataset2
	obj.pp = pp;  obj.nn = nn; obj.pn = pn; obj.np = np;
	return obj;
}