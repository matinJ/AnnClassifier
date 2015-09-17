#include "AnnHelper.h"
#include "CBrowseDir.h"
#include "PCAHelper.h"
#include "MathUtils.h"
#include "MatUtils.h"
#include "FileUtils.h"
// #include <windows.h>   
// #include <iostream>   
// #include <Shlwapi.h>  
using namespace cv;
using namespace std;

void AnnHelper::CreateLayers(cv::Mat& _layerMat, int _inNum, int _outNum, int _layers/*=3*/)
{
	int centreNum = sqrt(_inNum*_outNum);
	int layers[3] = { _inNum, centreNum, _outNum };
	Mat_<int> layerSize(1, 3);
	memcpy(layerSize.data, layers, sizeof(int)* 3);
	_layerMat = layerSize;
}

void AnnHelper::CreateAnn(int _inNum, int _outNum, int _layers/*=3*/, int _activateFunc /*= CvANN_MLP::SIGMOID_SYM*/)
{
	//设置中间层
	Mat layerSizes;
	CreateLayers(layerSizes, _inNum, _outNum, _layers);
	m_net.create(layerSizes, _activateFunc);//CvANN_MLP::SIGMOID_SYM  
	//CvANN_MLP::GAUSSIAN  
	//CvANN_MLP::IDENTITY 
}

void AnnHelper::Save(std::string _path)
{
	m_net.save(_path.c_str());
}

AnnHelper::AnnHelper()
{

}

void AnnHelper::SetParams()
{
	//设置结束条件
	m_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 0.001);
	//设置权值矫正算法
	m_params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	//官方解释：bp_dw_scale（只用于bp网络），该系数乘以计算出的权值梯度，推荐值为0.1。
	//该参数可通过构造函数的param1设置。
	//目测应该是学习速率
	m_params.bp_dw_scale = 0.1;
	//动量参数
	m_params.bp_moment_scale = 0.1;
}

void AnnHelper::SetParams(CvANN_MLP_TrainParams params)
{
	m_params = params;
}

void AnnHelper::Train(cv::Mat& _label, cv::Mat& _response)
{
	m_net.train(_label, _response, Mat(), Mat(), m_params);
}

void AnnHelper::LoadResponsesForTraining(std::vector<string> _files, cv::Mat& _response)
{
	FileUtils futils;
	MatUtils matutils;
	Mat responses;

	for (string filename : _files)
	{
		//获取图片信息文件路径
		string fileinfo = filename.substr(0, filename.find_last_of('\\'));//目录
		fileinfo += "\\training_data.txt";
		//读取结果矩阵
		vector<string> matNames;
		vector<string> matStrs;
		futils.ReadMatFile(matStrs, matNames, fileinfo);
		assert(matStrs.size()>0);
		
		Mat response = matutils.ToMat(matStrs[0]);
		matutils.AddMatRow(responses, response, responses);

	}
	_response = responses;
}


void AnnHelper::LoadResponsesForTraining_new(std::vector<std::string> _files, cv::Mat& _response)
{
	FileUtils futils;
	MathUtils mathUtils;
	MatUtils matutils;
	Mat responses;

	for (string filename : _files)
	{
		//获取图片信息文件路径
		string fileinfo = filename.substr(0, filename.find_last_of('\\'));//目录
		fileinfo += "\\training_data_new.txt";
		//读取结果矩阵
		vector<string> matNames;
		vector<string> matStrs;
		futils.ReadMatFile(matStrs, matNames, fileinfo);
		assert(matStrs.size() > 0);

		int bcodeIndex = 0;//数值矩阵
		int bitIndex = 0;//位数矩阵

		for (int i = 0; i < matNames.size(); i++)
		{
			if (matNames[i] == "code")
			{
				bcodeIndex = i;
			}
			if (matNames[i] == "bit")
			{
				bitIndex = i;
			}
		}

		string bcode = "";
		Mat decCodeMat = matutils.ToMat(matStrs[bcodeIndex]);
		Mat bitMat = matutils.ToMat(matStrs[bitIndex]);
		assert(decCodeMat.cols == bitMat.cols);
		int bitNum=0;
		int bit = 0;
		for (int i = 0; i < decCodeMat.cols; i++)
		{
			bitNum = decCodeMat.at<float>(0, i);
			bit = bitMat.at<float>(0, i);
			bcode += mathUtils.ConvertToBinary(bitNum, bit);
		}
		stringstream ss;
		Mat resMat(1, bcode.length(), CV_32F);
		for (int i = 0; i < resMat.cols; i++)
		{
			ss << bcode[i];
			float ib = 0.;
			ss >> ib;
			resMat.at<float>(0, i) = ib;
			ss.clear();
		}
		matutils.AddMatRow(responses, resMat, responses);
	}
	_response = responses;
}


void AnnHelper::ConvertMatForAnn(cv::Mat& _mat)
{
	cv::Mat mat(_mat.rows, _mat.cols, CV_32FC1);
	for (int r = 0; r < _mat.rows; r++)
	{
		for (int c = 0; c < _mat.cols; c++)
		{
			mat.at<float>(r, c) = _mat.at<float>(r, c);
		}
	}
	_mat = mat;
}

void AnnHelper::Predict(cv::Mat& _inputs, cv::Mat& _outputs)
{
	m_net.predict(_inputs, _outputs);
}

void AnnHelper::Load(std::string _path)
{
	m_net.load(_path.c_str());
}

//神经网络保存测试
// int main(int argc, char** argv)
// {
// 	FileUtils futils;
// 	AnnHelper annHp;
// 	annHp.SetParams();
// 	annHp.CreateAnn(12, 10, 3);
// 	//获取图片
// 	CStatDir m_statdir;
// 	m_statdir.SetInitDir("D:\\Codes\\opencv\\AnnClassifier\\Resource\\背光源瓶身\\背光源");
// 	vector<string> fileNames = m_statdir.BeginBrowseFilenames("*.*");
// 	futils.ChoosingImageFiles(fileNames);
// 	vector<Mat> imgs;
// 	for (string fname : fileNames)
// 	{
// 		imgs.push_back(imread(fname, CV_8U));
// 	}
// 	//pca
// 	PCAHelper pcaHp;
// 	pcaHp.LoadPCA("D:\\Codes\\opencv\\AnnClassifier\\Resource\\pca.txt");
// 	Mat pcaPoints;
// 	pcaHp.Project(imgs, pcaPoints);
// 	//train
// 	Mat responses;
// 	annHp.LoadResponsesForTraining(fileNames, responses);
// 	annHp.Train(pcaPoints, responses);
// 	annHp.Save("D:\\Codes\\opencv\\AnnClassifier\\Resource\\ann.xml");
// 	Mat test = pcaPoints(Rect(0,59,pcaPoints.cols,1));
// 	Mat testResult;
// 	//成功
// // 	CvANN_MLP bp;
// // 	bp.load("D:\\Codes\\opencv\\AnnClassifier\\Resource\\ann.xml");
// // 	bp.predict(test, testResult);
// 	//成功
// // 	AnnHelper annHp2;
// // 	annHp2.Load("D:\\Codes\\opencv\\AnnClassifier\\Resource\\ann.xml");
// // 	annHp2.Predict(test, testResult);
// 
// 	cout << "test"<<endl << test<<endl;
// 	cout << "result" << endl << testResult;
// 	system("pause");
// 
// }
// 
