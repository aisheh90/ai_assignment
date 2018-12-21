#pragma once
#include<iostream>
#include <string>
#include <vector>

using namespace std;

// override for +,-,*, /  to deal with matrix 
vector <vector <double>> operator-(vector<vector<double>> vect1, vector<vector<double>> vect2);
vector <vector <double>> operator+(vector<vector<double>> vect1, vector<vector<double>> vect2);
vector <vector <double>> operator/(vector<vector<double>> vect1, double value);
vector <vector <double>> operator*(vector<vector<double>> vect1, vector<vector<double>> vect2);
vector <vector <double>> operator*(vector<vector<double>> vect1, double value);

// multiply matrix function .. (dot product)
vector<vector<double>> mult(vector<vector<double>> vect1, vector<vector<double>> vect2);

