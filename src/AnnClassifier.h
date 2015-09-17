#ifndef __ANN_CLASSIFIER_H__
#define __ANN_CLASSIFIER_H__
#include <opencv2/opencv.hpp>
#include "MDTHelper1.h"
#include "PCAHelper.h"
#include <vector>
#include <map>
struct AnnResult
{
	int liq;//Һ�� 0=��Һ����1=Һ��
	//�ײ�������
	//000=���ٵ���,001=���ٵ���,010=���ʵ���,011=�϶����,100=�ܶ����
	int bottomQuantity;
	//�ײ����ɱ��氼͹��ƽ�̶�
	//00=����,01=��̫����,10=�ǳ�������
	int lumpLevel;
	//�����ķɽ���ļ���
	//00=�޷ɽ�,01=��һЩ�ɽ�,02=�д����ɽ���һ�㳬��ƿ���һ�룩
	int splash1;
	//��ĩ״�ɽ��ļ���
	//00=�޷ɽ�,01=��һЩ�ɽ�,02=�д����ɽ���һ�㳬��ƿ���һ�룩
	int splash2;
	//������������������ת����AnnResult������
	static std::vector<AnnResult> TranslateResult(cv::Mat& _resultMat);
	//TranslateResult�����������
	static void TranslateToMat(std::vector<AnnResult>& _annResults,cv::Mat& _mat);
	//�����������static�������ú���ֻ�������Լ�ת��Ϊmat
	//����֮���matֻ��һ��
	cv::Mat ToMat();
	//��AnnResult����Ϊ����ַ���,������toString()����
	std::string PrintResult();
	//��AnnResult�е�ĳ����������Ϊ�ַ���
	std::string translateLiq();
	std::string translateBottomQuantity();
	std::string translateLumpLevel();
	std::string translateSplash1();
	std::string translateSplash2();
};

struct AnnProcData
{
	cv::Mat pcaResult;//pca���
	cv::Mat annResult;//��������������
};

struct AnnData
{
	cv::Mat testImg;//ͼƬ
	AnnProcData procData;//���̱���
	AnnResult imgResult;//ͼƬʶ����
	//���캯��
	AnnData(cv::Mat _testImg
		, AnnResult _imgResult){
		testImg = _testImg.clone(), imgResult = _imgResult;
	};
	AnnData(){};
	//���ѵ�������Ƿ���Ч
	//0=��ȫ��1=�޷�ѵ����2=�޷�Ԥ�⣬3=���̱�����ȫ
	//����û��ʵ��
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
	//Ŀǰ����_param���ܵ�ȡֵΪ<"path_pca",path>,<"path_ann",path>,<"mdt_layer",5>
	virtual bool Train(std::string _imgRootPath, std::string _savingPath
		, std::map<std::string, std::string> _params = std::map<std::string, std::string>());
};
#endif