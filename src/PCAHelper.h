#ifndef __PCA_HELPER_H__
#define __PCA_HELPER_H__
#include <opencv2/opencv.hpp>
#include <fstream>

class PCAHelper
{
private:
	cv::PCA m_pca;
	cv::Mat FormatImagesForPCA(const std::vector<cv::Mat> &data);
	void ResizeImgsForPCA(std::vector<cv::Mat> &data, cv::Size _size = cv::Size(54, 128));
	cv::Mat ToGrayscale(cv::InputArray _src);
public:
	PCAHelper();
	cv::PCA CreatePCA(std::vector<cv::Mat> &imgs, int maxComponents ,cv::Size _size = cv::Size(54, 128));
	void SavePCA(std::string _path);
	cv::PCA LoadPCA(std::string _path);
	
	

	void Project(cv::Mat& _img, cv::Mat& _point, cv::Size _size = cv::Size(54, 128));
	void Project(std::vector<cv::Mat>& _imgs, cv::Mat& _point, cv::Size _size=cv::Size(54,128));
	void BackProject(cv::Mat& _point, cv::Mat& _img, cv::Size _size = cv::Size(54, 128), int channels=1);
};
#endif