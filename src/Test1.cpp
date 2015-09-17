#include <opencv2/opencv.hpp>
#include <fstream>
#include "FileUtils.h"
#include "AnnHelper.h"
#include "AnnClassifier.h"
using namespace std;
using namespace cv;

//测试FileUtils的LoadImgs方法
// int main(int argc, char** argv)
// {
// 	vector<Mat> imgs;
// 	FileUtils futils;
// 	futils.LoadImgs("D:\\Codes\\opencv\\AnnClassifier\\Resource", imgs);
// 	//图片都装载到了imgs中。
// }

//测试FileUtils的LoadResponsesForTraining_new函数
 //int main(int argc, char** argv)
 //{
 //	vector<string> files;
 //	FileUtils futils;
 //	futils.FindImgs("E:\\Bottle\\FY\\二", files);
 //	AnnHelper annHp;
 //	Mat response;
 //	annHp.LoadResponsesForTraining_new(files, response);
 //	cout << response;
 //	system("pause");
 //}

//AnnClassifier train函数测试（pca_ann）
// int main(int argc, char** argv)
// {
// 	AnnClassifier classifier;
// 	classifier.init(PCA_ANN);
// 	classifier.Train("D:\\Codes\\opencv\\AnnClassifier\\Resource", "D:\\Codes\\opencv\\AnnClassifier\\Resource");
// }

//AnnClassifier train函数测试（mdt_pca_ann）
// int main(int argc, char** argv)
// {
// 	AnnClassifier classifier;
// 	classifier.init(MDT_PCA_ANN);
// 	map<string, string> mdtLayer;
// 	mdtLayer.insert(pair<string,string>("mdt_layer", "3"));
// 	classifier.Train("D:\\Codes\\opencv\\AnnClassifier\\Resource", "D:\\Codes\\opencv\\AnnClassifier\\Resource",mdtLayer);
// }

//AnnClassifier predict测试（pca_ann）
// int main(int argc, char** argv)
// {
// 	FileUtils futils;
// 	AnnClassifier classifier;
// 	map<string, string> params;
// 	params.insert(pair<string, string>("path_pca", "D:\\Codes\\opencv\\AnnClassifier\\Resource\\pca.txt"));
// 	params.insert(pair<string, string>("path_ann", "D:\\Codes\\opencv\\AnnClassifier\\Resource\\ann.xml"));
// 	classifier.init(ClassifierTypes::PCA_ANN, params);
// 
// 	vector<Mat> imgs;
// 	futils.LoadImgs("D:\\Codes\\opencv\\AnnClassifier\\Resource", imgs);
// 
// 	AnnData data;
// 	classifier.Predict(imgs[3], data);
// 	cout << data.imgResult.PrintResult();
// 	namedWindow("3");
// 	imshow("3", imgs[3]);
// 	waitKey(0);
// 	system("pause");
// }

//AnnClassifier predict测试（mdt_pca_ann）
// int main(int argc, char** argv)
// {
// 	FileUtils futils;
// 	AnnClassifier classifier;
// 	map<string, string> params;
// 	params.insert(pair<string, string>("path_pca", "D:\\Codes\\opencv\\AnnClassifier\\Resource\\pca.txt"));
// 	params.insert(pair<string, string>("path_ann", "D:\\Codes\\opencv\\AnnClassifier\\Resource\\ann.xml"));
// 	params.insert(pair<string, string>("mdt_layer", "3"));
// 	classifier.init(ClassifierTypes::MDT_PCA_ANN, params);
// 
// 	vector<Mat> imgs;
// 	futils.LoadImgs("D:\\Codes\\opencv\\AnnClassifier\\Resource", imgs);
// 
// 	AnnData data;
// 	classifier.Predict(imgs[14], data);
// 	cout << data.imgResult.PrintResult();
// 	namedWindow("14");
// 	imshow("14", imgs[14]);
// 	waitKey(0);
// 	system("pause");
// }
// 
