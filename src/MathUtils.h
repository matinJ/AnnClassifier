#ifndef __MATH_UTILS_H__
#define __MATH_UTILS_H__
#include <opencv2/opencv.hpp>

class MathUtils
{
public:
	int ConvertToInt(std::string bnum);
	std::string ConvertToBinary(int num,int n);
	
};

#endif