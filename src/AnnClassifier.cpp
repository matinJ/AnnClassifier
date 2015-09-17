#include "AnnClassifier.h"
#include "CBrowseDir.h"
#include "PCAHelper.h"
#include "AnnHelper.h"
#include "FileUtils.h"
#include "PCAHelper.h"
#include "AnnHelper.h"
#include "MathUtils.h"
#include "MatUtils.h"
#include "MDTHelper1.h"
using namespace std;
using namespace cv;


class AnnClassifier_Pca_Ann :public IAnnClassifier
{
private:
	PCAHelper m_pcaHp;
	AnnHelper m_annHp;
public:
	virtual void Predict(cv::Mat& _img, AnnData& _result);
	virtual void Predict(std::vector<cv::Mat>& _imgs, std::vector<AnnData>& _results);
	virtual bool Train(std::string _imgRootPath, std::string _savingPath
		, std::map<std::string, std::string> _params = std::map<std::string, std::string>());

	AnnClassifier_Pca_Ann();
	AnnClassifier_Pca_Ann(std::string _pcaPath, std::string _annPath);
	bool LoadPCA(std::string _path);
	bool LoadAnn(std::string _path);
};

class AnnClassifier_MDT_PCA_ANN :public IAnnClassifier
{
private:
	AnnClassifier_Pca_Ann m_parent;
	int m_MDTLayer;
public:
	AnnClassifier_MDT_PCA_ANN(int _mdtLayer = 2);
	AnnClassifier_MDT_PCA_ANN(std::string _pcaPath, std::string _annPath,int _mdtLayer=2);
	bool LoadPCA(std::string _path);
	bool LoadAnn(std::string _path);
	void SetMDTLayers(int _mdtLayer);

	virtual void Predict(cv::Mat& _img, AnnData& _result);
	virtual void Predict(std::vector<cv::Mat>& _imgs, std::vector<AnnData>& _results);
	virtual bool Train(std::string _imgRootPath, std::string _savingPath
		, std::map<std::string, std::string> _params = std::map<std::string, std::string>());
};

AnnClassifier_MDT_PCA_ANN::AnnClassifier_MDT_PCA_ANN(int _mdtLayer /*= 2*/)
{
	SetMDTLayers(_mdtLayer);
}

AnnClassifier_MDT_PCA_ANN::AnnClassifier_MDT_PCA_ANN(std::string _pcaPath, std::string _annPath, int _mdtLayer/*=2*/)
{
	LoadPCA(_pcaPath);
	LoadAnn(_annPath);
	SetMDTLayers(_mdtLayer);
}



bool AnnClassifier_MDT_PCA_ANN::LoadPCA(std::string _path)
{
	return m_parent.LoadPCA(_path);
}

bool AnnClassifier_MDT_PCA_ANN::LoadAnn(std::string _path)
{
	return m_parent.LoadAnn(_path);
}

void AnnClassifier_MDT_PCA_ANN::SetMDTLayers(int _mdtLayer)
{
	m_MDTLayer = _mdtLayer;
}


void AnnClassifier_MDT_PCA_ANN::Predict(cv::Mat& _img, AnnData& _result)
{
	MDTHelper1 mdtHp;
	Mat img = _img.clone();
	mdtHp.ColorDWT(img, m_MDTLayer);
	mdtHp.GetPartOfDWT(img, img, 0, m_MDTLayer);
	m_parent.Predict(img, _result);
	_result.testImg = _img;
}

void AnnClassifier_MDT_PCA_ANN::Predict(std::vector<cv::Mat>& _imgs, std::vector<AnnData>& _results)
{
	MDTHelper1 mdtHp;
	MatUtils matutils;

	_results.clear();

	vector<Mat> imgs;
	matutils.CopyMatTo(_imgs, imgs);
	for (int i = 0; i < imgs.size(); i++)
	{
		mdtHp.ColorDWT(imgs[i], m_MDTLayer);
		mdtHp.GetPartOfDWT(imgs[i], imgs[i], 0, m_MDTLayer);
	}
	m_parent.Predict(imgs, _results);

	for (int i = 0; i < _imgs.size(); i++)
	{
		_results[i].testImg = _imgs[i];
	}
}


bool AnnClassifier_MDT_PCA_ANN::Train(std::string _imgRootPath, std::string _savingPath, std::map<std::string, std::string> _params /*= std::map<std::string, std::string>()*/)
{
	try
	{
		FileUtils futils;
		vector<string> filenames;
		vector<Mat> imgs;

		futils.FindImgs(_imgRootPath, filenames);
		futils.LoadImgs(filenames, imgs);
		//mdt
		MDTHelper1 mdtHp;
		map<string, string>::iterator it = _params.find("mdt_layer");
		////外部赋予mdt_layer值
		string mdtLayer = it == _params.end() ? "" : _params["mdt_layer"];
		if (mdtLayer != "")
		{
			stringstream ss;
			ss << mdtLayer;
			ss>> m_MDTLayer;
		}
		for (int i = 0; i < imgs.size(); i++)
		{
			mdtHp.ColorDWT(imgs[i], m_MDTLayer);
			mdtHp.GetPartOfDWT(imgs[i], imgs[i], 0, m_MDTLayer);
		}
		//pca
		PCAHelper pcaHp;
		Mat pcaPoints;
		pcaHp.CreatePCA(imgs, 12);
		pcaHp.Project(imgs, pcaPoints);
		pcaHp.SavePCA(_savingPath + "//pca.txt");
		//ann
		AnnHelper annHp;
		Mat responses;
		annHp.CreateAnn(12, 10);
		annHp.SetParams();
		annHp.LoadResponsesForTraining(filenames, responses);
		annHp.Train(pcaPoints, responses);
		annHp.Save(_savingPath + "//ann.xml");
		return true;
	}
	catch (exception e)
	{
		return false;
	}
}

bool AnnClassifier_Pca_Ann::Train(std::string _imgRootPath, std::string _savingPath, std::map<std::string, std::string> _params /*= std::map<std::string, std::string>()*/)
{
	try
	{
		FileUtils futils;
		vector<string> filenames;
		vector<Mat> imgs;

		futils.FindImgs(_imgRootPath, filenames);
		futils.LoadImgs(filenames, imgs);
		//pca
		PCAHelper pcaHp;
		Mat pcaPoints;
		pcaHp.CreatePCA(imgs, 12);
		pcaHp.Project(imgs, pcaPoints);
		pcaHp.SavePCA(_savingPath + "//pca.txt");
		//ann
		AnnHelper annHp;
		Mat responses;
		annHp.CreateAnn(12, 10);
		annHp.SetParams();
		annHp.LoadResponsesForTraining(filenames, responses);
		annHp.Train(pcaPoints, responses);
		annHp.Save(_savingPath + "//ann.xml");
		return true;
	}
	catch (exception e)
	{
		return false;
	}
}

bool AnnClassifier_Pca_Ann::LoadPCA(std::string _path)
{
	try
	{
		m_pcaHp.LoadPCA(_path);
		return true;
	}
	catch (exception e)
	{
		return false;
	}
	
}

bool AnnClassifier_Pca_Ann::LoadAnn(std::string _path)
{
	try
	{
		m_annHp.Load(_path);
		return true;
	}
	catch (exception e)
	{
		return false;
	}
	
}

AnnClassifier_Pca_Ann::AnnClassifier_Pca_Ann(std::string _pcaPath, std::string _annPath)
{
	LoadPCA(_pcaPath);
	LoadAnn(_annPath);
}

AnnClassifier_Pca_Ann::AnnClassifier_Pca_Ann()
{

}

void AnnClassifier_Pca_Ann::Predict(cv::Mat& _img, AnnData& _result)
{
	vector<Mat> imgs;
	vector<AnnData> results;
	imgs.push_back(_img.clone());
	results.push_back(_result);
	Predict(imgs, results);
	if (imgs.size() > 0 && results.size() > 0)
	{
		_result = results[0];
	}
}

void AnnClassifier_Pca_Ann::Predict(std::vector<cv::Mat>& _imgs, std::vector<AnnData>& _results)
{
	MathUtils mutils;
	MatUtils matutils;

	_results.clear();

	vector<Mat> imgs;
	matutils.CopyMatTo(_imgs, imgs);
	//pca处理
	Mat pcaPoints;
	m_pcaHp.Project(imgs, pcaPoints);
	//ann处理
	Mat annOutputs;
	m_annHp.Predict(pcaPoints, annOutputs);
	//整理结果
	vector<AnnResult> annResVec=AnnResult::TranslateResult(annOutputs);
	for (int i = 0; i < imgs.size(); i++)
	{
		AnnData data;
		data.testImg = _imgs[i].clone();
		data.procData.pcaResult = pcaPoints(Rect(0,i,pcaPoints.cols,1));
		data.procData.annResult = annOutputs(Rect(0, i, annOutputs.cols, 1));
		data.imgResult = annResVec[i];
		_results.push_back(data);
	}
}

AnnClassifier::AnnClassifier()
{

}


AnnClassifier::~AnnClassifier()
{
	if (!m_classifier)
	{
		delete m_classifier;
	}
}

bool AnnClassifier::init(ClassifierTypes _type, std::map<std::string, std::string> _params /*= std::map<std::string, std::string>()*/)
{
	stringstream ss;
	std::map<std::string, std::string>::iterator it;
	string pcaPath = _params.find("path_pca") == _params.end() ? "" : _params["path_pca"];
	string annPath = _params.find("path_ann") == _params.end() ? "" : _params["path_ann"];
	string str_mdt_layer = _params.find("mdt_layer") == _params.end() ? "" : _params["mdt_layer"];
	try
	{
		switch (_type)
		{
		case PCA_ANN:
			if (_params.size() == 2 && pcaPath != ""&&annPath != "")
			{
				m_classifier = new AnnClassifier_Pca_Ann(pcaPath, annPath);
			}
			else if (_params.size() == 0)
			{
				m_classifier = new AnnClassifier_Pca_Ann();
			}
			break;
		case MDT_PCA_ANN:
			if (_params.size() == 3 && pcaPath != ""&&annPath != ""&&str_mdt_layer!="")
			{
				int mdtLayers = 0;
				ss << str_mdt_layer;
				ss >> mdtLayers;
				m_classifier = new AnnClassifier_MDT_PCA_ANN(pcaPath, annPath, mdtLayers);
			}
			else if (_params.size() == 2)
			{
				m_classifier = new AnnClassifier_MDT_PCA_ANN(pcaPath, annPath);
			}
			else if (_params.size() == 0)
			{
				m_classifier = new AnnClassifier_MDT_PCA_ANN();
			}
			break;

		default:
			break;
		}
		return true;
	}
	catch (exception e)
	{
		return false;
	}
}

void AnnClassifier::Predict(std::vector<cv::Mat>& _imgs, std::vector<AnnData>& _results)
{
	if (!m_classifier) return;
	m_classifier->Predict(_imgs, _results);
}

void AnnClassifier::Predict(cv::Mat& _img, AnnData& _result)
{
	if (!m_classifier) return;
	m_classifier->Predict(_img, _result);
}

bool AnnClassifier::Train(std::string _imgRootPath, std::string _savingPath, std::map<std::string, std::string> _params /*= std::map<std::string, std::string>()*/)
{
	return m_classifier->Train(_imgRootPath, _savingPath, _params);
}

std::vector<AnnResult> AnnResult::TranslateResult(cv::Mat& _resultMat)
{
	MathUtils mutils;
	vector<AnnResult> annRes;
	stringstream ss;
	try
	{
		for (int r = 0; r < _resultMat.rows; r++)
		{
			AnnResult annr;
			for (int c = 0; c < _resultMat.cols; c++)
			{
				float bit = _resultMat.at<float>(r, c);
				int ibit = bit>0.5 ? 1 : 0;
				ss << ibit;
			}
			string bcode = "";
			ss >> bcode;
			annr.liq = mutils.ConvertToInt(bcode.substr(0, 1));
			annr.bottomQuantity = min(mutils.ConvertToInt(bcode.substr(1, 3)), 4);
			annr.lumpLevel = min(mutils.ConvertToInt(bcode.substr(4, 2)), 2);
			annr.splash1 = min(mutils.ConvertToInt(bcode.substr(6, 2)), 2);
			annr.splash2 = min(mutils.ConvertToInt(bcode.substr(8, 2)), 2);
			annRes.push_back(annr);
			ss.clear();
		}
	}
	catch (exception e)
	{

	}
	return annRes;

}

cv::Mat AnnResult::ToMat()
{
	MathUtils mutils;
	MatUtils matutils;

	string numStr = mutils.ConvertToBinary(liq, 1)
		+ mutils.ConvertToBinary(bottomQuantity, 3)
		+ mutils.ConvertToBinary(lumpLevel, 2)
		+ mutils.ConvertToBinary(splash1, 2)
		+ mutils.ConvertToBinary(splash2, 2);

	stringstream ss;
	Mat resMat(1, 10, CV_32F);
	for (int i = 0; i < 10; i++)
	{
		ss << numStr[i];
		ss >> resMat.at<float>(0, i);
		ss.clear();
	}
	return resMat;
}

std::string AnnResult::PrintResult()
{
	stringstream ss;
	string printStr = "";

	Mat bcodeMat = ToMat();
	ss << "code=";
	for (int i = 0; i < bcodeMat.cols; i++)
	{
		ss << bcodeMat.at<float>(0, i);
	}

	ss << "结果:"
		<< "是否液化：" << translateLiq()
		<< ";底部冻干量:" << translateBottomQuantity()
		<< ";底部规则度：" << translateLumpLevel()
		<< ";块状飞溅：" << translateSplash1()
		<< ";粉状飞溅：" << translateSplash2();
	ss >> printStr;
	return printStr;

}

std::string AnnResult::translateLiq()
{
	string mean = "";
	switch (liq)
	{
	case 0:
		mean = "非液化";
		break;
	case 1:
		mean = "液化";
		break;
	default:
		break;
	}
	return mean;
}

std::string AnnResult::translateBottomQuantity()
{
	string mean = "";
	switch (bottomQuantity)
	{
	case 0:
		mean = "严重少量";
		break;
	case 1:
		mean = "较少";
		break;
	case 2:
		mean = "正常";
		break;
	case 3:
		mean = "较多";
		break;
	case 4:
		mean = "严重过量";
		break;
	default:
		break;
	}
	return mean;
}

std::string AnnResult::translateLumpLevel()
{
	string mean = "";
	switch (lumpLevel)
	{
	case 0:
		mean = "规则";
		break;
	case 1:
		mean = "比较不规则";
		break;
	case 2:
		mean = "非常不规则";
		break;
	default:
		break;
	}
	return mean;
}

std::string AnnResult::translateSplash1()
{
	string mean = "";
	switch (splash1)
	{
	case 0:
		mean = "无块状飞溅";
		break;
	case 1:
		mean = "有块状飞溅";
		break;
	case 2:
		mean = "大量块状飞溅";
		break;
	default:
		break;
	}
	return mean;
}

std::string AnnResult::translateSplash2()
{
	string mean = "";
	switch (splash2)
	{
	case 0:
		mean = "无粉状飞溅";
		break;
	case 1:
		mean = "有粉状飞溅";
		break;
	case 2:
		mean = "大量粉状飞溅";
		break;
	default:
		break;
	}
	return mean;
}

void AnnResult::TranslateToMat(std::vector<AnnResult>& _annResults, cv::Mat& _mat)
{
	MathUtils mutils;
	MatUtils matutils;

	Mat mat;
	for (int i = 0; i < _annResults.size(); i++)
	{
		matutils.AddMatRow(mat, _annResults[i].ToMat(), mat);
	}
	_mat = mat;
}

//AnnClassifier_MDT_Pca_Ann测试
 //int main(int argc, char** argv)
 //{
 //	FileUtils futils;
 //	vector<Mat> imgs;
 //	vector<AnnData> datas;
 //	futils.LoadImgs("E:\\瓶子\\AnnClassifier\\Resource", imgs);
 // 	AnnClassifier_Pca_Ann classifier("E:\\瓶子\\AnnClassifier\\Resource\\pca.txt"
 // 		, "E:\\瓶子\\AnnClassifier\\Resource\\ann.xml");
 //	//vector<string> params;
 //	//params.push_back("D:\\Codes\\opencv\\AnnClassifier\\Resource\\pca.txt");
 //	//params.push_back("D:\\Codes\\opencv\\AnnClassifier\\Resource\\ann.xml");
 //	//AnnClassifier classifier;
 //	//classifier.init(ClassifierTypes::MDT_PCA_ANN, params);
 // 	classifier.Predict(imgs, datas);
 //	cout << datas[3].imgResult.PrintResult();
 //	namedWindow("3");
 //	imshow("3", datas[3].testImg);
 //	waitKey(0);
 //	system("pause");
 //
 //}

//AnnClassifier测试 成功
// int main(int argc, char** argv)
// {
// 	AnnClassifier classifier;
// 	vector<string> paths;
// 	paths.push_back("D:\\Codes\\opencv\\AnnClassifier\\Resource\\pca.txt");
// 	paths.push_back("D:\\Codes\\opencv\\AnnClassifier\\Resource\\ann.xml");
// 	classifier.init(ClassifierTypes::PCA_ANN, paths);
// 	Mat lisa = imread("D:\\Codes\\opencv\\Opencv2Tests\\OpencvTests\\TestResource\\lisa1.jpg", CV_8U);
// 	AnnData data;
// 	classifier.Predict(lisa, data);
// 	cout << data.imgResult.PrintResult();
// 	namedWindow("lisa");
// 	imshow("lisa",lisa);
// 	waitKey(0);
// 	system("pause");
// 
// }

//AnnClassifier_Pca_Ann train函数测试 成功
// int main(int argc, char** argv)
// {
// // 	AnnClassifier_Pca_Ann trainer;
// // 	trainer.Train("D:\\Codes\\opencv\\AnnClassifier\\Resource", "D:\\Codes\\opencv\\AnnClassifier\\Resource");
// 	AnnClassifier_MDT_PCA_ANN trainer;
// 	trainer.Train("D:\\Codes\\opencv\\AnnClassifier\\Resource", "D:\\Codes\\opencv\\AnnClassifier\\Resource");
// }

// int main(int argc, char** argv)
// {
// 	Mat lisa = imread("D:\\Codes\\opencv\\Opencv2Tests\\OpencvTests\\TestResource\\lisa1.jpg",CV_8U);
// 	Mat lisaBackup = lisa.clone();
// 	AnnData result;
// 	AnnClassifier_Pca_Ann classifier("D:\\Codes\\opencv\\AnnClassifier\\Resource\\pca.txt"
// 		, "D:\\Codes\\opencv\\AnnClassifier\\Resource\\ann.xml");
// 	classifier.Predict(lisa, result);
// 	cout << result.imgResult.PrintResult();
// 	namedWindow("lisa");
// 	imshow("lisa", lisaBackup);
// 
// 	waitKey(0);
// 	system("pause");
// }

// AnnClassifier_Pca_Ann predict函数测试 成功
// int main(int argc, char** argv)
// {
// 	FileUtils futils;
// 	MatUtils matutils;
// 	vector<Mat> imgs;
// 	vector<Mat> imgsBackup;
// 	futils.LoadImgs("D:\\Codes\\opencv\\AnnClassifier\\Resource", imgs);
// 	matutils.CopyMatTo(imgs, imgsBackup);
// 	AnnClassifier_Pca_Ann classifier("D:\\Codes\\opencv\\AnnClassifier\\Resource\\pca.txt"
// 		,"D:\\Codes\\opencv\\AnnClassifier\\Resource\\ann.xml");
// 	CvANN_MLP bp;
// 	bp.load("D:\\Codes\\opencv\\AnnClassifier\\Resource\\ann.xml");
// 	vector<AnnData> datas;
// 	classifier.Predict(imgs, datas);
// 	namedWindow("31");
// 	cout << datas[31].imgResult.PrintResult();
// 	imshow("31",imgsBackup[31]);
// 	waitKey(0);
// 	system("pause");
// 	
// }


