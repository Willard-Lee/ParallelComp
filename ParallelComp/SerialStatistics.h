#pragma once

//Includes
#include <iostream>
#include <string>
#include <vector>

//Sort Type ENUM
typedef enum SORT { ASCENDING, DECENDING };
//Define MyType
typedef float myType;

class SerialStatistics
{
public:
	//Empty Constructor and Destructor
	SerialStatistics();
	~SerialStatistics();

	//Sorts a input vector using the bubble sort algorithm
	std::vector<myType>& Sort(std::vector<myType>& Values, SORT Mode);
	//Finds the sum of an input vector
	myType Sum(std::vector<myType>& Values);
	//Finds the Min or Max of an input vector
	myType MinMax(std::vector<myType>& Values, bool MinMax);
	//Displays a vector 
	void Display(std::vector<myType>& Values);
	//Gets the median value from an input vector
	myType GetMedianValue(std::vector<myType>& Values);
	//Gets the mean value from an input vector
	myType Mean(std::vector<myType>& Values);
	//Gets the Standard Deviation value from an input vector
	myType StandardDeviation(std::vector<myType>& Values);
	//Gets the First Quartile value from an input vector
	myType FirstQuartile(std::vector<myType>& Values);
	//Gets the Third Quartile value from an input vector
	myType ThirdQuartile(std::vector<myType>& Values);
};