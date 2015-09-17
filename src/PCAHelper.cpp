#include "PCAHelper.h"
#include "FileUtils.h"
#include "MatUtils.h"
using namespace cv;
using namespace std;

PCAHelper::PCAHelper()
{

}



cv::Mat PCAHelper::FormatImagesForPCA(const std::vector<cv::Mat> &data)
{
	Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_32F);
	for (unsigned int i = 0; i < data.size(); i++)
	{
		Mat image_row = data[i].clone().reshape(1, 1);
		Mat row_i = dst.row(i);
		image_row.convertTo(row_i, CV_32F);
	}
	return dst;
}

void PCAHelper::ResizeImgsForPCA(std::vector<cv::Mat> &data, cv::Size _size /*= cv::Size(54, 128)*/)
{
	for (int i = 0; i < data.size(); i++)
	{
		Mat tmp;
		resize(data[i], tmp, _size);
		data[i] = tmp;
	}
}

cv::Mat PCAHelper::ToGrayscale(cv::InputArray _src)
{
	Mat src = _src.getMat();
	// only allow one channel
	if (src.channels() != 1) {
		CV_Error(CV_StsBadArg, "Only Matrices with one channel are supported");
	}
	// create and return normalized image
	Mat dst;
	cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

cv::PCA PCAHelper::CreatePCA(std::vector<cv::Mat> &imgs, int maxComponents, cv::Size _size /*= cv::Size(54, 128)*/)
{
	ResizeImgsForPCA(imgs, _size);
	Mat pcaMat = FormatImagesForPCA(imgs);
	// perform PCA
	PCA pca(pcaMat, cv::Mat(), CV_PCA_DATA_AS_ROW, maxComponents);
	m_pca = pca;
	return pca;
}

void PCAHelper::SavePCA(std::string path)
{
	FileUtils futils;
	ofstream outfile;
	outfile.open(path);
	futils.OutStreamMat(outfile, m_pca.eigenvalues, "eigenvalues");
	outfile << endl;
	futils.OutStreamMat(outfile, m_pca.eigenvectors, "eigenvectors");
	outfile << endl;
	futils.OutStreamMat(outfile, m_pca.mean, "mean");
	outfile.close();
}

cv::PCA PCAHelper::LoadPCA(std::string _path)
{
	FileUtils futils;
	MatUtils matutils;
	PCA pca;
	std::vector<std::string> matStrs;
	std::vector<std::string> matNames;
	futils.ReadMatFile(matStrs, matNames, _path);
	pca.eigenvalues = matutils.ToMat(matStrs[0]);
	pca.eigenvectors = matutils.ToMat(matStrs[1]);
	pca.mean = matutils.ToMat(matStrs[2]);
	m_pca = pca;
	return pca;
}

void PCAHelper::BackProject(cv::Mat& _point, cv::Mat& _img, cv::Size _size /*= cv::Size(54, 128)*/, int channels/*=1*/)
{
	Mat reconstruction = m_pca.backProject(_point); // re-create the image from the "point"
	reconstruction = reconstruction.reshape(channels, _size.width*_size.height); // reshape from a row vector into image shape
	reconstruction = ToGrayscale(reconstruction); // re-scale for displaying purposes
	_img = reconstruction;
}


void PCAHelper::Project(cv::Mat& _img, cv::Mat& _point, cv::Size _size/*=cv::Size(54,128)*/)
{
	vector<Mat> imgs;
	imgs.push_back(_img);
	Project(imgs, _point, _size);
}

void PCAHelper::Project(std::vector<cv::Mat>& _imgs, cv::Mat& _point, cv::Size _size/*=cv::Size(54,128)*/)
{
	ResizeImgsForPCA(_imgs, _size);
	Mat pcaMat = FormatImagesForPCA(_imgs);
	_point=m_pca.project(pcaMat);
}






