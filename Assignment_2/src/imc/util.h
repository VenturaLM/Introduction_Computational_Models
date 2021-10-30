/*
 * util.h
 *
 *  Created on: 06/03/2015
 *      Author: pedroa
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <cstdlib> // To establish the seed srand() and generate pseudorandom numbers rand()
#include <iostream>
#include <cmath>

namespace util
{
	static int *integerRandomVectorWithoutRepeating(int min, int max, int howMany)
	{
		int total = max - min + 1;
		int *numbersToBeSelected = new int[total];
		int *numbersSelected = new int[howMany];
		// Initialize the list of possible selections
		for (int i = 0; i < total; i++)
			numbersToBeSelected[i] = min + i;

		for (int i = 0; i < howMany; i++)
		{
			int selectedNumber = rand() % (total - i);
			// Store the selected number
			numbersSelected[i] = numbersToBeSelected[selectedNumber];
			// We include the last valid number in numbersToBeSelected, in this way
			// all numbers are valid until total-i-1
			numbersToBeSelected[selectedNumber] = numbersToBeSelected[total - i - 1];
		}
		delete[] numbersToBeSelected;
		return numbersSelected;
	};

	static void computeAverageErrors(double *testErrors, double *trainErrors, double &averageTestError, double &averageTrainError)
	{
		for (auto i = 0; i < 5; i++)
		{
			averageTestError += testErrors[i];
			averageTrainError += trainErrors[i];
		}
		averageTestError /= 5;
		averageTrainError /= 5;
	};

	static void computeStdErrors(double *testErrors, double *trainErrors, double &stdTestError, double &stdTrainError, double averageTestError, double averageTrainError)
	{
		for (auto i = 0; i < 5; i++)
		{
			stdTestError += pow((testErrors[i] - averageTestError), 2);
			stdTrainError += pow((trainErrors[i] - averageTrainError), 2);
		}
		stdTestError = sqrt(stdTestError / 5);
		stdTrainError = sqrt(stdTrainError / 5);
	};
}

#endif /* UTIL_H_ */
