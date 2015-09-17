#include "MathUtils.h"
#include <bitset>
using namespace cv;
using namespace std;


int MathUtils::ConvertToInt(std::string bnum)
{
	int num = 0;
	stringstream ss;
	for (int i = 0; i < bnum.length(); i++)
	{
		int index = bnum.length() - i - 1;
		int numbit = 0;
		ss << bnum[index];
		ss >> numbit;
		num += numbit*pow(2, i);
		ss.clear();
	}
	return num;
}

std::string MathUtils::ConvertToBinary(int num, int n)
{
	stringstream ss;
	string bnumStr;
	ss << bitset<32>(num);
	ss >> bnumStr;
	int ii = bnumStr.length() - n > 0 ? bnumStr.length() - n : 0;
	int nn = min(n, static_cast<int>(bnumStr.length()));
	return bnumStr.substr(ii,nn);
}


// int main(int argc, char** argv)
// {
//  	MathUtils mutils;
// // 	cout << mutils.ConvertToInt("1111010010001101");
// 	cout << mutils.ConvertToBinary(14, 10);
// 	system("pause");
// }
