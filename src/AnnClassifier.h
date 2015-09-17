#ifndef __ANN_CLASSIFIER_H__
#define __ANN_CLASSIFIER_H__
#include <opencv2/opencv.hpp>
#include "MDTHelper1.h"
#include "PCAHelper.h"
#include <vector>
#include <map>
struct AnnResult
{
	int liq;//液化 0=无液化，1=液化
	//底部冻干量
	//000=很少的量,001=较少的量,010=合适的量,011=较多的量,100=很多的量
	int bottomQuantity;
	//底部冻干表面凹凸不平程度
	//00=规则,01=不太规则,10=非常不规则
	int lumpLevel;
	//连续的飞溅块的级别
	//00=无飞溅,01=有一些飞溅,02=有大量飞溅（一般超过瓶身的一半）
	int splash1;
	//粉末状飞溅的级别
	//00=无飞溅,01=有一些飞溅,02=有大量飞溅（一般超过瓶身的一半）
	int splash2;
	//将神经网络输出结果矩阵转换到AnnResult的数组
	static std::vector<AnnResult> TranslateResult(cv::Mat& _resultMat);
	//TranslateResult函数的逆操做
	static void TranslateToMat(std::vector<AnnResult>& _annResults,cv::Mat& _mat);
	//区别于上面的static函数，该函数只讲对象自己转化为mat
	//言下之意该mat只有一行
	cv::Mat ToMat();
	//将AnnResult解释为结果字符串,类似于toString()方法
	std::string PrintResult();
	//将AnnResult中的某个分量解释为字符串
	std::string translateLiq();
	std::string translateBottomQuantity();
	std::string translateLumpLevel();
	std::string translateSplash1();
	std::string translateSplash2();
};

struct AnnProcData
{
	cv::Mat pcaResult;//pca结果
	cv::Mat annResult;//神经网络的输入矩阵
};

struct AnnData
{
	cv::Mat testImg;//图片
	AnnProcData procData;//过程变量
	AnnResult imgResult;//图片识别结果
	//构造函数
	AnnData(cv::Mat _testImg
		, AnnResult _imgResult){
		testImg = _testImg.clone(), imgResult = _imgResult;
	};
	AnnData(){};
	//检查训练样本是否有效
	//0=齐全，1=无法训练，2=无法预测，3=过程变量不全
	//这里没有实现
	int check();
};

class IAnnClassifier
{
public:
	virtual void Predict(cv::Mat& _img,AnnData& _result) = 0;
	virtual void Predict(std::vector<cv::Mat>& _imgs, std::vector<AnnData>& _results) = 0;
	virtual bool Train(std::string _imgRootPath, std::string _savingPath
		, std::map<std::string, std::string> _params = std::map<std::string, std::string>()) = 0;
};

enum ClassifierTypes
{
	PCA_ANN,
	MDT_PCA_ANN
};

class AnnClassifier :public IAnnClassifier
{
private:
	IAnnClassifier* m_classifier;
private:
public:
	AnnClassifier();
	~AnnClassifier();
	bool init(ClassifierTypes _type, std::map<std::string, std::string> _params = std::map<std::string, std::string>());
	virtual void Predict(std::vector<cv::Mat>& _imgs, std::vector<AnnData>& _results);
	virtual void Predict(cv::Mat& _img, AnnData& _result);
	//目前参数_param可能的取值为<"path_pca",path>,<"path_ann",path>,<"mdt_layer",5>
	virtual bool Train(std::string _imgRootPath, std::string _savingPath
		, std::map<std::string, std::string> _params = std::map<std::string, std::string>());
};
#endif