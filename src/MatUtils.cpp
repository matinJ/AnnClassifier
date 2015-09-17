#include "MatUtils.h"
using namespace cv;
using namespace std;


void MatUtils::CopyMatTo(std::vector<cv::Mat>& _from, std::vector<cv::Mat>& _to)
{
	_to.clear();
	for (Mat mat : _from)
	{
		_to.push_back(mat.clone());
	}
}

void MatUtils::AddMatRow(cv::Mat& _mat1, cv::Mat& _mat2, cv::Mat& _result)
{
	int cols = max(_mat1.cols, _mat2.cols);
	int rows = _mat1.rows + _mat2.rows;
	Mat result(rows, cols, CV_32F);
	for (int r = 0; r < _mat1.rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			if (c < _mat1.cols)
			{
				result.at<float>(r, c) = _mat1.at<float>(r, c);
			}
			else
			{
				result.at<float>(r, c) = 0.f;
			}
		}
	}

	for (int r = _mat1.rows; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			if (c < _mat2.cols)
			{
				result.at<float>(r, c) = _mat2.at<float>(r - _mat1.rows, c);
			}
			else
			{
				result.at<float>(r, c) = 0.f;
			}
		}
	}
	_result = result;
}

MatUtils::MatUtils()
{

}

cv::Mat MatUtils::ToMat(std::vector<std::vector<float>> _data)
{
	int cols = 0;
	//寻找最大列
	for (vector<float> colvec : _data)
	{
		if (colvec.size() > cols)
		{
			cols = colvec.size();
		}
	}
	//赋值
	Mat mat(_data.size(), cols, CV_32F);
	for (int r = 0; r < _data.size(); r++)
	{
		for (int c = 0; c < _data.size(); c++)
		{
			mat.at<float>(r, c) = _data[r][c];
		}
	}
	return mat;
}

cv::Mat MatUtils::ToMat(std::string matStr)
{
	stringstream ss;
	vector<vector<float>> matVec;

	int row = -1;
	for (int i = 0; i < matStr.length(); i++)
	{
		if (matStr[i] == '[')
		{
			//新增一行
			matVec.push_back(vector<float>());
			row++;
			continue;
		}
		else if (matStr[i] == ';')
		{
			//赋值
			float num;
			ss >> num;
			matVec[row].push_back(num);
			ss.clear();
			//新增一行
			matVec.push_back(vector<float>());
			row++;
			continue;
		}
		else if (matStr[i] == ']' || matStr[i] == ',')
		{
			//赋值
			float num;
			ss >> num;
			matVec[row].push_back(num);
			ss.clear();
			continue;
		}
		ss << matStr[i];
	}
	int cols = 0;
	for (int r = 0; r < matVec.size(); r++)
	{
		int tmpcols = matVec[r].size();
		if (tmpcols >= cols)
		{
			cols = tmpcols;
		}
	}

	Mat mat(matVec.size(), cols, CV_32F);
	for (int r = 0; r < matVec.size(); r++)
	{
		for (int c = 0; c < cols; c++)
		{
			if (c >= matVec[r].size())
			{
				mat.at<float>(r, c) = 0.0;
			}
			else
			{
				mat.at<float>(r, c) = matVec[r][c];
			}
		}
	}
	return mat;
}
