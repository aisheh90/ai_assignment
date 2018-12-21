#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <ppl.h>
#include <windows.h>
#include <math.h>       
#include <numeric>   

#include "concurrent_vector.h"
#include "overwritesHeader.h"
#include "otheUsefulrFun.h"
#include "nn_header.h"
#include "dt_header.h"                                                           

using namespace std;


// DTPaths to store created paths by the DT
 vector<vector<double>> DTPaths;

// calcGainForOneFeature function used in descretizeData function for calculating each midpoint gain
// row data contains all examples feature values ( <= mid point, > mid point)
double calcGainForOneFeature(int datasetNum,vector<double>rowdata, vector<double>outdata)
{
	
	int calss1count = 0;
	int calss2count = 0;

	// count of each class for the given value
	if (datasetNum == 1)
	{
		calss1count = count(outdata.begin(), outdata.end(), 1);
		calss2count = count(outdata.begin(), outdata.end(), 2);
	}
	else
	{
		calss1count = count(outdata.begin(), outdata.end(), 0);
		calss2count = count(outdata.begin(), outdata.end(), 1);
	}

	//basic gain
	double basicgain = simpleGain(calss1count, calss2count);
	int total = calss1count + calss2count;

	
	double midpointgain = 0;

	// gain for each distinct value 0 ,1 
	for (int j = 0; j < 2; j++)
	{
		int numofclass1 = 0, numofclass2 = 0;
		for (int i = 0; i < rowdata.size(); i++)
		{
			if (datasetNum == 1)
			{
				if (rowdata[i] == j && outdata[i] == 1)
					numofclass1++;
				else
					numofclass2++;
			}
			else if (datasetNum == 2)
			{
				if (rowdata[i] == j && outdata[i] == 0)
					numofclass1++;
				else
					numofclass2++;
			}
		}
		
		// calculations for gain
		int valuecount = numofclass1 + numofclass2;
		double valuegain = simpleGain(numofclass1, numofclass2);
		midpointgain = midpointgain + ((static_cast<double>(valuecount) / static_cast<double>(total)) * valuegain);
		
	}

	// calculations for gain
	midpointgain = basicgain - midpointgain;
				
	return midpointgain;
}


// descretizeData fucntion used to descretize feature values when its continous data into 2-ways based on midpoint with heighest gain
vector <vector<double>> descretizeData(int datasetNum,vector <vector<double>> inputdata, vector<double> outputdata)
{
	//  for simplicity and since all the code assume working with numeric data
	//  when choose a mid poitn based on a specific criteria .. <= midpoint , > midpoint 
	//  I used 0 as a value for values <= midpoint && 1 as a value for values > midpoint

	vector <vector<double>> newinputdata;
	
	// I've implemnted two methods to descrertize 
	// .. the common way which is based on heighest gain
	// the secon way based on average value ... (commented)

	// Method 1: if we choose midpoint to spilt based on heighest gain value .. this the best method to split based on it 
	vector <vector<double>> tempdata=transpose(inputdata);
	for (int f = 0; f < tempdata.size(); f++)
	{
		
		// find unique values in current feature 		   
		vector<double> currentUniquVals = tempdata[f];
		sort(currentUniquVals.begin(), currentUniquVals.end());
		currentUniquVals.erase(unique(currentUniquVals.begin(), currentUniquVals.end()), currentUniquVals.end());
			
			
		// num of distinct values  > 3
		if (currentUniquVals.size() > 3)
		{
			// find mid points
			vector<double> midpoints;
			for (int m = 0; m < currentUniquVals.size()-1; m++)
			{
				double currentmidpoint = (currentUniquVals[m] + currentUniquVals[m + 1]) / 2;
				midpoints.push_back(currentmidpoint);
			}


			// calcualte gain for each midpoint
			vector<double> tempgains;

			// iterate at all examples and replace values with zeros and ones based on <= midpoint or >midpoint 
			for (int m = 0; m < midpoints.size(); m++)
			{

				vector<double> rowdata;
				vector <double> outdata;
				
				// any value <= midpoints make it zero , and if > midpoints make it 1
				for (int j = 0; j < tempdata[f].size(); j++)
				{
					if (tempdata[f][j] <= midpoints[m])
					{
							rowdata.push_back(0);
					}
					else
					{
						rowdata.push_back(1);
					}

					outdata.push_back(outputdata[j]);

				}

				// calculate gain for current midpoint 
				double midpointgain = calcGainForOneFeature(datasetNum,rowdata, outdata);

				// push it into all midpoints gains vector 
				tempgains.push_back(midpointgain);
			}

			// find index of the midpoint with heighest gain
			int highestg = findHighestGain(tempgains);

			// now descretize based on the chosen midpoint
			vector<double> chosenrowdata;
			for (int j = 0; j < tempdata[f].size(); j++)
			{
				if (tempdata[f][j] <= midpoints[highestg])
					chosenrowdata.push_back(0);
				else
					chosenrowdata.push_back(1);
					
			}
			newinputdata.push_back(chosenrowdata);

					
		}
		else //  when feature unique values <= 3 .. then no need for descretization .. take feature data as is  
		{
			newinputdata.push_back(tempdata[f]);
		}

	}

	
	// Method -2 : if we choose midpoint to spilt based on average value
	/*
	vector <vector<double>> tempdata = transpose(inputdata);
	for (int f = 0; f < tempdata.size(); f++)
	{
		// find unique values in current feature 
		vector<double> currentUniquVals = tempdata[f];
		sort(currentUniquVals.begin(), currentUniquVals.end());
		currentUniquVals.erase(unique(currentUniquVals.begin(), currentUniquVals.end()), currentUniquVals.end());

		vector<double> rowdata;

		//find the avg of the values 
		double mean = accumulate(currentUniquVals.begin(), currentUniquVals.end(), 0.0) / currentUniquVals.size();

		// any value <= avg make it zero , and if > avg make it 1
		for (int j = 0; j < tempdata[f].size(); j++)
		{
			if (tempdata[f][j] <= mean)
				rowdata.push_back(0);
			else
				rowdata.push_back(1);

		}

		newinputdata.push_back(rowdata);
	}
	*/
	
	// return the readu descretized data 
	newinputdata = transpose(newinputdata);
	return newinputdata;

}

// findfeaturesinfo function to find information for each feature 
vector <featuresInformation> findfeaturesinfo(int datasetNum, vector<vector <double>> dataInput, vector<double>dataOutput)
{

	vector < vector <double>> input(dataInput[0].size(), vector<double>(dataInput.size(), 1));
	// transpose data to iterate over features not examples
	input = transpose(dataInput);

	vector <featuresInformation> featuresInfo;

	// iterate over features 
	for (int i = 0; i < input.size(); i++)
	{
		// find distinct values of current feature 
		vector<double> currentUniquVals = input[i];
		sort(currentUniquVals.begin(), currentUniquVals.end());
		currentUniquVals.erase(unique(currentUniquVals.begin(), currentUniquVals.end()), currentUniquVals.end());

		// count distinct values of current feature 
		featuresInformation currntFeature;
		currntFeature.numofDistinctValues = currentUniquVals.size();

		// stire  the  feature and number of its distinct value and store unique values also
		for (int j = 0; j < currentUniquVals.size(); j++)
		{
			featureValueInfo currentvalInfo;
			currentvalInfo.uniquVal = currentUniquVals[j];
			currntFeature.uniquValsInfo.push_back(currentvalInfo);
		}

		featuresInfo.push_back(currntFeature);

	}


	// find number of class 1 and class 2 with each distinct value of each feature 
	// iterate over features 
	for (int i = 0; i < input.size(); i++)
	{
		// iterate over examples 
		for (int j = 0; j < input[i].size(); j++)
		{
			// iterate over distinct values of feature i
			for (int k = 0; k < featuresInfo[i].uniquValsInfo.size(); k++)
			{
				
				if (input[i][j] == featuresInfo[i].uniquValsInfo[k].uniquVal)
				{

					if (datasetNum == 1)
					{
						//class1 = 1 class2 = 2
						if (dataOutput[j] == 1)
						{
							featuresInfo[i].uniquValsInfo[k].numofclass1++;
							featuresInfo[i].tnumofclass1++;
						}
						else
						{
							featuresInfo[i].uniquValsInfo[k].numofclass2++;
							featuresInfo[i].tnumofclass2++;
						}

					}
					else if (datasetNum == 2)
					{
						//class1 = 0 class2 = 1
						if (dataOutput[j] == 0)
						{
							featuresInfo[i].uniquValsInfo[k].numofclass1++;
							featuresInfo[i].tnumofclass1++;
						}
						else
						{
							featuresInfo[i].uniquValsInfo[k].numofclass2++;
							featuresInfo[i].tnumofclass2++;
						}
					}


					break;


				}


			}

		}

	}

		

	 // print for debugging 
	 /*
		for (int i = 0; i < featuresInfo.size(); i++)
		{
			cout << " feature #: " << i << endl;
			cout << " num of distinct values =  " << featuresInfo[i].numofDistinctValues<< endl;
			cout << " total num of class1  #: " << featuresInfo[i].tnumofclass1 << endl;
			cout << " total num of class2  #: " << featuresInfo[i].tnumofclass2 << endl;

			for (int j = 0; j < featuresInfo[i].uniquValsInfo.size(); j++)
			{
				cout << " distinct value # :  " << j << " value = " << featuresInfo[i].uniquValsInfo[j].uniquVal;
				cout << " class1val = " << featuresInfo[i].uniquValsInfo[j].numofclass1;
				cout << " class1va2 = " << featuresInfo[i].uniquValsInfo[j].numofclass2 << endl;

			}
			cout << " --------------";
		}
		*/
		
		return	featuresInfo;
	
}


// simpleGain function calculate gain of single value of a feature
double simpleGain(int numofclass1, int numofclass2)
{
	double g = 0;
	int total = numofclass1 + numofclass2;
	double	part1 = 0;
	double	part2 = 0;
	if(numofclass1 >0)
		part1 = (static_cast<double> (numofclass1) / static_cast<double>(total)) * (log2((static_cast<double>(numofclass1) / static_cast<double>(total))));
	if(numofclass2 >0)
		part2 = (static_cast<double> (numofclass2) / static_cast<double>(total)) * (log2((static_cast<double>(numofclass2) / static_cast<double>(total))));
	
	g = -1 * (part1 + part2);

	return g;

}

// calcGain fucntion calculate gain for all features by calling simpleGain function  for each single value of that feature 
vector <double> calcGain(int datasetNum, vector<double> outdata, vector <featuresInformation> featuresInfoVect)
{
	vector <double> gainInfo;

	// count of each class in the whole data 
	int calss1count = 0;
	int calss2count = 0;

	if (datasetNum == 1)
	{
		calss1count = count(outdata.begin(), outdata.end(), 1);
		calss2count = count(outdata.begin(), outdata.end(), 2);
	}
	else
	{
		calss1count = count(outdata.begin(), outdata.end(), 0);
		calss2count = count(outdata.begin(), outdata.end(), 1);
	}

	//basic gain
	double basicgain = simpleGain(calss1count, calss2count);
	int total = calss1count + calss2count;

	// gain for each feature
	for (int i = 0; i < featuresInfoVect.size(); i++)
	{
		double featuregain = 0;

		// gain for each distinct value 
		for (int j = 0; j < featuresInfoVect[i].numofDistinctValues; j++)
		{
			int valuecount = featuresInfoVect[i].uniquValsInfo[j].numofclass1 + featuresInfoVect[i].uniquValsInfo[j].numofclass2;
			double valuegain = simpleGain(featuresInfoVect[i].uniquValsInfo[j].numofclass1, featuresInfoVect[i].uniquValsInfo[j].numofclass2);
			featuregain = featuregain + ((static_cast<double>(valuecount) / static_cast<double>(total)) * valuegain);
			
		}

		// continue calculations for gain
		featuregain = basicgain - featuregain;
		gainInfo.push_back(featuregain);
	
 	}

	return gainInfo;
}

// findHighestGain function return index of the feature with heighest gain
int findHighestGain(vector <double> featuresGain)
{
	
	int maxElementIndex = std::max_element(featuresGain.begin(), featuresGain.end()) - featuresGain.begin();
	//cout << " feature # with highest gain = " << maxElementIndex << endl;
	return maxElementIndex;
}

// createTree function create the decison tree (DT) 
void createTree(int datasetNum, vector<vector<double>> data, vector<double>output, string str, int numOftotalFeatures, int numOfUsedFeatures, vector<int>fetauresNumsVect, double curFrVal, bool dtfailedsplitbranchmethod)
{
	// stop is a flag becomes true in 2 situations (all examples have the same class in that branch, or no more features to split)
	bool stop = false;

	// branch of a current value of a feature 
	int currentNode = 0;

	// find information and gain of avilable features for current branch 
	vector <featuresInformation> featuresInfoVect = findfeaturesinfo(datasetNum, data, output);
	vector <double> featuresGain = calcGain(datasetNum, output, featuresInfoVect);

	// check if all avilable features have the same gain
	if (adjacent_find(featuresGain.begin(), featuresGain.end(), not_equal_to<double>()) == featuresGain.end())
	{
		//cout << " all gain  same " << endl;

		// if all gains zeros  .. i.e stop work no benifit 
		if (featuresGain[0] == 0 )
		{
			stop = true;

			//to enforce stop 
			numOfUsedFeatures = numOftotalFeatures + 1;
		}
		else // all gain same but not zero .. then give feature with lowest number of ditinct values a higher priorty 
		{
			for (int i = 1; i < featuresInfoVect.size(); i++)
			{
				if (featuresInfoVect[i].numofDistinctValues < featuresInfoVect[currentNode].numofDistinctValues)
				{
					currentNode = i;
				}
			}
		}

		//cout << " feature # with lowest number of distinct values = " << currentNode << endl;
	}
	else  // differnt gains .. so continue work as expected and take the value with heighest gain
	{
		currentNode = findHighestGain(featuresGain);
	}

	// since we chose a new feature index ... increase number of used features 

	numOfUsedFeatures = numOfUsedFeatures + 1;

	
	// for storing path info 
	if(curFrVal == curFrVal)
	{
		str = str + ", Val: " + to_string(curFrVal) +",";
	}

	// value of the class for current path 
	double classVal;

	//stop split or not
	// rule 1: check if all samples in this datasubset have the same clas value 
	if (adjacent_find(output.begin(), output.end(), not_equal_to<double>()) == output.end())
	{
		//All elements are equal each other
		stop = true;
		classVal = output[0]; // since all have the same calss value
	}
	// rule 2: no remaining fetaures to split on (or no benifit to split if all gains = 0)
	// i.e. in this case there examples with class 1 and examples with class 2
	// so can't split .. but what class to give??
	// method 1: give it the more importnat class and assign ot to the path .. but this bias to class over the other class
	// method 2: randomly choose class and assign ot to the path 
	// method 3: the class with more samples in that branch
	else if (numOfUsedFeatures > numOftotalFeatures)
	{
		stop = true;
	

		if(dtfailedsplitbranchmethod==1)
		{
			// method 1
			// based on important we bias to the positive class .. maybe based on the effect of the dangerous of the dieases 
			if (datasetNum == 1)
			{
				classVal = 2;
			}
			else if (datasetNum == 2)
			{
				classVal = 1;			
			}
		}
		else if (dtfailedsplitbranchmethod == 2)
		{
			// method 2:
			// randomly choose
			if (datasetNum == 1)
			{
				int arr[2] = { 1,2 };
				classVal = arr[rand() % 2];
			}
			else if (datasetNum == 2)
			{
				int arr[2] = { 0,1 };
				classVal = arr[rand() % 2];
			}

		}
		else if (dtfailedsplitbranchmethod == 3)
		{

			// method -3 
			// consider the class with more samples 

			int calss1count = 0;
			int calss2count = 0;

			if (datasetNum == 1)
			{
				calss1count = count(output.begin(), output.end(), 1);
				calss2count = count(output.begin(), output.end(), 2);
			}
			else
			{
				calss1count = count(output.begin(), output.end(), 0);
				calss2count = count(output.begin(), output.end(), 1);
			}

			if (calss1count >= calss2count)
			{
				classVal = calss1count;
			}
			else
			{
				classVal = calss2count;
			}
		}
			

	}


	if (stop== true) // yes stop .. store path
	{
		// we store the path nodes gradually in a string .. when finish extract path info from the string and store the path in the DTPaths vector 
		str = str + " class: " + to_string(classVal);
		cout << " Path # " << DTPaths.size() << " :	" ;
		cout << str << endl;
		//cout << " -----------------------------------------"<<endl;

		// add path to tree
		 addpathtotree(str);
		
	}
	else // continue build the path 
	{
		// store current feature 
		int actualfeatureNum = fetauresNumsVect[currentNode];
		str = str + "F: " + to_string(actualfeatureNum);

		// each distinct value is a subrtree
		for (int i = 0; i < featuresInfoVect[currentNode].numofDistinctValues; i++)
		{
			// current val to work on 
			curFrVal = featuresInfoVect[currentNode].uniquValsInfo[i].uniquVal;
		
			// take section of data where the value of the currrent feature = current value 
			vector<vector<double>> currentsubtreedata;
			vector <double> currentoutput;

			for (int j = 0; j < data.size(); j++)
			{
				if (data[j][currentNode] == featuresInfoVect[currentNode].uniquValsInfo[i].uniquVal)
				{
					currentsubtreedata.push_back(data[j]);
					currentoutput.push_back(output[j]);
				}

			}

			//delete that feature from data
			currentsubtreedata = transpose(currentsubtreedata);
			currentsubtreedata.erase( currentsubtreedata.begin() + currentNode);
			currentsubtreedata = transpose(currentsubtreedata);

			fetauresNumsVect.erase( fetauresNumsVect.begin() + currentNode );

			// call createTree fucntion recursively for the current section of data 
			createTree(datasetNum, currentsubtreedata, currentoutput, str, numOftotalFeatures, numOfUsedFeatures, fetauresNumsVect,  curFrVal, dtfailedsplitbranchmethod);

			// this is to keep track what features used so far 
			fetauresNumsVect.erase(fetauresNumsVect.begin(), fetauresNumsVect.end());
			for (int i = 0; i < data[0].size(); i++)
			{
				fetauresNumsVect.push_back(i);
			}
		}
	}
}

// addpathtotree fucntion add new path to the DT by parsing the received string 
// the string format is such the following example:
//string str = "F: 6, Val: 30.000, F: 18, Val: 1, class: 0";
// where F: 6, means feature with index 6 ,,, Val: 30.00 is the current value of feature with index 6 ...and so on 
// and the last part is the class value of that path
void addpathtotree(string str)
{
	// the extracted path info will be stored here 
	vector<double> path;
	   
	
	// a falg to tell the end of the path  
	bool contSearch = true;

	while (contSearch)
	{		
		int temp1 = (int)str.find(",");

		// reach the last part of the path (class value part)
		if (temp1 == -1)
		{
			contSearch = false;
		}
		else
		{
			string part1 = str.substr(0, temp1);
			str = str.substr(temp1 + 1, str.length());

			int temp2 = (int)part1.find("F: ");
			int temp3 = (int)part1.find("Val: ");
			if (temp2 != -1) // fetaure number 
			{
				double featurenum = stod(part1.substr(temp2 + 3, part1.length()));
				path.push_back(featurenum);
			}
			if (temp3 != -1) // fetaure value 
			{
				double value = stod(part1.substr(temp3 + 5, part1.length()));
				path.push_back(value);
			}

		}
	}

	// end of the path get the class value 
	double calssval = stod(str.substr(7, str.length()));
	path.push_back(calssval);
		
	// store the path into DTPaths
	DTPaths.push_back(path);

}


// predictDT function used to predict class of each exampel in the data usign the created decsion tree 
vector<double> predictDT(int datasetNum,vector<vector<double>> input)
{
	vector<double> predicted;

	// iterate on examples
	for (int i = 0; i < input.size(); i++) 
	{
		// a flag to track if path applied or not 
		bool flag = true; 

		// find suitbale path .. iterate over paths
		for (int j = 0; j < DTPaths.size(); j++) 
		{
			// iterate over current path nodes (features and values)
			for (int k = 0; k < DTPaths[j].size() - 1;)
			{
				if (input[i][DTPaths[j][k]] == DTPaths[j][k + 1])
				{
					//cout <<" so far satisfied " <<endl;
					flag = true;

				}
				else /// this path not suitable 
				{					
					flag = false;
					break;
				}

				// this due to the format of the path I stored, fetaure index then value 
				k = k + 2;
			}

			// all the path components applied 
			if (flag == true)
			{
				// take that path class value as the predicted value of the current example
				predicted.push_back(DTPaths[j][DTPaths[j].size() - 1]);
				break;
			}						

		}

	}


	

	return predicted;
}


