#ifndef __ANN_HELPER_H__
#define __ANN_HELPER_H__
#include <opencv2/opencv.hpp>

class AnnHelper
{
private:
	CvANN_MLP m_net;//神经网络
	CvANN_MLP_TrainParams m_params;
private:
	void CreateLayers(cv::Mat& _layerMat, int _inNum, int _outNum,int _layers=3);//构造层
public:
	AnnHelper();
	void CreateAnn(int _inNum, int _outNum, int _layers=3,
		int _activateFunc = CvANN_MLP::SIGMOID_SYM);
	void Save(std::string _path);
	void Load(std::string _path);
	void Train(cv::Mat& _label,cv::Mat& _response);
	void SetParams();
	void SetParams(CvANN_MLP_TrainParams params); //训练参数

	void Predict(cv::Mat& _inputs,cv::Mat& _outputs);


	void LoadResponsesForTraining(std::vector<std::string> _files,cv::Mat& _response);
	void LoadResponsesForTraining_new(std::vector<std::string> _files, cv::Mat& _response);
	
	void ConvertMatForAnn(cv::Mat& _mat);
	



};

#endif