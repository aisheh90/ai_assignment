#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <ppl.h>
#include <windows.h>

#include "concurrent_vector.h"
#include"overwritesHeader.h"

using namespace std;
using namespace concurrency;


// override for "-" operator to deal with matrix 
vector <vector <double>> operator-(vector<vector<double>> vect1, vector<vector<double>> vect2)
{

	vector <vector <double>> result(vect1.size(), vector <double>(vect1[0].size(), 0));
	// for paralleization
	size_t size = vect1.size();
	concurrency::parallel_for(size_t(0), size, [&](size_t i)
	{

		for (size_t j = 0; j < vect1[0].size(); j++)
		{
			result[i][j] = vect1[i][j] - vect2[i][j];

		}
	});

	return result;
}

// override for "+" operator to deal with matrix 
vector <vector <double>> operator+(vector<vector<double>> vect1, vector<vector<double>> vect2)
{

	vector <vector <double>> result(vect1.size(), vector <double>(vect1[0].size(), 0));
	// for paralleization
	size_t size = vect1.size();
	concurrency::parallel_for(size_t(0), size, [&](size_t i)
	{

		for (size_t j = 0; j < vect1[0].size(); j++)
		{
			result[i][j] = vect1[i][j] + vect2[i][j];

		}
	});

	return result;
}


// override for "/" operator to deal with matrix 
vector <vector <double>> operator/(vector<vector<double>> vect1, double value)
{

	vector <vector <double>> result(vect1.size(), vector <double>(vect1[0].size(), 0));
	// for paralleization
	size_t size = vect1.size();
	concurrency::parallel_for(size_t(0), size, [&](size_t i)
	{

		for (size_t j = 0; j < vect1[0].size(); j++)
		{
			result[i][j] = vect1[i][j] / value;

		}
	});

	return result;
}


// override for "*" operator to deal with matrix 
vector <vector <double>> operator*(vector<vector<double>> vect1, vector<vector<double>> vect2)
{

	vector <vector <double>> result(vect1.size(), vector <double>(vect1[0].size(), 0));
	// for paralleization
	size_t size = vect1.size();
	concurrency::parallel_for(size_t(0), size, [&](size_t i)
	{

		for (size_t j = 0; j < vect1[0].size(); j++)
		{
			result[i][j] = vect1[i][j] * vect2[i][j];

		}
	});

	return result;
}


// override for "*" operator to deal with matrix multiplcation with value
vector <vector <double>> operator*(vector<vector<double>> vect1, double value)
{

	vector <vector <double>> result(vect1.size(), vector <double>(vect1[0].size(), 0));
	// for paralleization
	size_t size = vect1.size();
	concurrency::parallel_for(size_t(0), size, [&](size_t i)
	{

		for (size_t j = 0; j < vect1[0].size(); j++)
		{
			result[i][j] = vect1[i][j] * value;

		}
	});

	return result;
}


// multiply matrix function .. (dot product)
vector<vector<double>> mult(vector<vector<double>> vect1, vector<vector<double>> vect2)
{
	vector<vector<double>> result(vect1.size(), vector<double>(vect2[0].size(), 0));
	// for paralleization
	size_t size = vect1.size();
	concurrency::parallel_for(size_t(0), size, [&](size_t i)
	{
		for (size_t j = 0; j < vect2[0].size(); j++)
		{
			double temp = 0;
			for (int k = 0; k < vect1[0].size(); k++)
			{
				temp += vect1[i][k] * vect2[k][j];
			}
			result[i][j] = temp;
		}
	});

	return result;
}

