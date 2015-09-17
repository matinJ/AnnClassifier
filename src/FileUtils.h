#ifndef __FILE_UTILS_H__
#define __FILE_UTILS_H__
#include <opencv2/opencv.hpp>
#include <windows.h>
class FileUtils
{
public:
	//按格式输入矩阵到文件
	void OutStreamMat(std::ofstream& _outfile, cv::Mat& _mat, std::string _matname);
	//从文件中读取矩阵字符串，文件格式:{标签}[Mat]
	void ReadMatFile(std::vector<std::string>& matStrs, std::vector<std::string>& matNames, std::string _path);
	//获取所有包含图片的目录	
	void FindImgDirs(std::vector<std::string>& _imgs, std::vector<std::string>& _dirs);
	//挑选出图片文件
	void ChoosingImageFiles(std::vector<std::string>& fileList);
	//查找文件目录（以及子目录）下的所有图片的绝对路径
	void FindImgs(std::string _root,std::vector<std::string>& _imgPathList);
	//装载图片
	void LoadImgs(std::string _root,std::vector<cv::Mat>& _imgs,int _type=CV_8U);
	void LoadImgs(std::vector<std::string>& _imgPathList, std::vector<cv::Mat>& _imgs, int _type = CV_8U);

	//字符串处理，本来想做路径转换的，但是现在没用
	LPWSTR ConvertCharToLPWSTR(const char * szString);
	char* WstrToAstr(WCHAR *wstr);

};
#endif