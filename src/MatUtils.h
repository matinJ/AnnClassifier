#ifndef __MAT_UTILS_H__
#define __MAT_UTILS_H__
#include <opencv2/opencv.hpp>

class MatUtils
{
public:
	MatUtils();
	void CopyMatTo(std::vector<cv::Mat>& _from, std::vector<cv::Mat>& _to);
	//������ƴ��
	void AddMatRow(cv::Mat& _mat1, cv::Mat& _mat2, cv::Mat& _result);
	//��[mat]�ַ����ع���mat
	cv::Mat ToMat(std::string matStr);

	cv::Mat ToMat(std::vector<std::vector<float>> _data);
};

#endif