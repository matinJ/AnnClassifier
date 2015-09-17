#include "FileUtils.h"
#include "CBrowseDir.h"
#include <fstream>
using namespace cv;
using namespace std;

void FileUtils::ReadMatFile(std::vector<std::string>& matStrs, std::vector<std::string>& matNames, std::string _path)
{
	ifstream istream;
	istream.open(_path, ios::in);
	char tmp[50];
	stringstream ss;
	while (!istream.eof())
	{
		istream >> tmp;
		string stmp(tmp);
		if (stmp[0] == '{'&&stmp[stmp.length() - 1] == '}')//标签
		{
			matNames.push_back(stmp.substr(1, stmp.length() - 2));
			ss.clear();
			continue;
		}
		ss << stmp;
		if (stmp[stmp.length() - 1] == ']')//mat结束
		{
			string matStr;
			ss >> matStr;
			matStrs.push_back(matStr);
		}

	}
}

void FileUtils::FindImgDirs(std::vector<std::string>& _imgs, std::vector<std::string>& _dirs)
{
	vector<string> dirs;
	for (string img : _imgs)
	{
		string tmpdir = img.substr(0, img.find_last_of('\\'));
		bool isExist = false;
		for (string dir : dirs)
		{
			if (dir == tmpdir)
			{
				isExist = true;
				break;
			}
		}
		if (!isExist)
		{
			dirs.push_back(tmpdir);
		}
	}
	_dirs = dirs;
}

// void FileUtils::AddMatRow(cv::Mat& _mat1, cv::Mat& _mat2, cv::Mat& _result)
// {
// 	int cols = max(_mat1.cols, _mat2.cols);
// 	int rows = _mat1.rows + _mat2.rows;
// 	Mat result(rows, cols, CV_32F);
// 	for (int r = 0; r < _mat1.rows; r++)
// 	{
// 		for (int c = 0; c < cols; c++)
// 		{
// 			if (c < _mat1.cols)
// 			{
// 				result.at<float>(r, c) = _mat1.at<float>(r, c);
// 			}
// 			else
// 			{
// 				result.at<float>(r, c) = 0.f;
// 			}
// 		}
// 	}
// 
// 	for (int r = _mat1.rows; r < rows; r++)
// 	{
// 		for (int c = 0; c < cols; c++)
// 		{
// 			if (c < _mat2.cols)
// 			{
// 				result.at<float>(r, c) = _mat2.at<float>(r - _mat1.rows, c);
// 			}
// 			else
// 			{
// 				result.at<float>(r, c) = 0.f;
// 			}
// 		}
// 	}
// 	_result = result;
// }

void FileUtils::ChoosingImageFiles(std::vector<std::string>& fileList)
{
	vector<string> files;
	for (string file : fileList)
	{
		int pointIndex = file.find_last_of(".");
		string tail = file.substr(pointIndex + 1, file.length() - pointIndex - 1);
		if (tail == "png" || tail == "jpg" || tail == "bmp")
		{
			files.push_back(file);
		}
	}
	fileList = files;
}

void FileUtils::OutStreamMat(std::ofstream& _outfile, cv::Mat& _mat, std::string _matname)
{
	_outfile << "{" << _matname << "}" << endl;
	_outfile << _mat;
}

LPWSTR FileUtils::ConvertCharToLPWSTR(const char * szString)
{
	int dwLen = strlen(szString) + 1;
	int nwLen = MultiByteToWideChar(CP_ACP, 0, szString, dwLen, NULL, 0);//算出合适的长度
	LPWSTR lpszPath = new WCHAR[dwLen];
	MultiByteToWideChar(CP_ACP, 0, szString, dwLen, lpszPath, nwLen);
	return lpszPath;
}

char* FileUtils::WstrToAstr(WCHAR *wstr)
{
	unsigned long i = lstrlen(wstr);
	char *astr;
	astr = (char*)malloc(i + 1);
	if (astr == NULL)
		return NULL;
	else
		memset(astr, '\0', i + 1);

	wcstombs(astr, wstr, i + 1);
	return astr;
}

void FileUtils::FindImgs(std::string _root, std::vector<std::string>& _imgPathList)
{
	
	CStatDir m_statdir;
	m_statdir.SetInitDir(_root.c_str());
	vector<string> fileNames = m_statdir.BeginBrowseFilenames("*.*");
	ChoosingImageFiles(fileNames);
	_imgPathList = fileNames;
}

void FileUtils::LoadImgs(std::string _root, std::vector<cv::Mat>& _imgs, int _type/*=CV_8U*/)
{
	vector<string> fileNames;
	FindImgs(_root, fileNames);
	LoadImgs(fileNames, _imgs, _type);
}

void FileUtils::LoadImgs(std::vector<std::string>& _imgPathList, std::vector<cv::Mat>& _imgs, int _type /*= CV_8U*/)
{
	vector<Mat> imgs;
	for (string file : _imgPathList)
	{
		imgs.push_back(imread(file, _type));
	}
	_imgs = imgs;
}

