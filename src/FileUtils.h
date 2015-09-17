#ifndef __FILE_UTILS_H__
#define __FILE_UTILS_H__
#include <opencv2/opencv.hpp>
#include <windows.h>
class FileUtils
{
public:
	//����ʽ��������ļ�
	void OutStreamMat(std::ofstream& _outfile, cv::Mat& _mat, std::string _matname);
	//���ļ��ж�ȡ�����ַ������ļ���ʽ:{��ǩ}[Mat]
	void ReadMatFile(std::vector<std::string>& matStrs, std::vector<std::string>& matNames, std::string _path);
	//��ȡ���а���ͼƬ��Ŀ¼	
	void FindImgDirs(std::vector<std::string>& _imgs, std::vector<std::string>& _dirs);
	//��ѡ��ͼƬ�ļ�
	void ChoosingImageFiles(std::vector<std::string>& fileList);
	//�����ļ�Ŀ¼���Լ���Ŀ¼���µ�����ͼƬ�ľ���·��
	void FindImgs(std::string _root,std::vector<std::string>& _imgPathList);
	//װ��ͼƬ
	void LoadImgs(std::string _root,std::vector<cv::Mat>& _imgs,int _type=CV_8U);
	void LoadImgs(std::vector<std::string>& _imgPathList, std::vector<cv::Mat>& _imgs, int _type = CV_8U);

	//�ַ���������������·��ת���ģ���������û��
	LPWSTR ConvertCharToLPWSTR(const char * szString);
	char* WstrToAstr(WCHAR *wstr);

};
#endif