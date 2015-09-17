#ifndef __MDT_HELPER_1_H__
#define __MDT_HELPER_1_H__
#include <opencv2/opencv.hpp>

class MDTHelper1
{
public:
	MDTHelper1();
	// ��ά��ɢС���任����ͨ������ͼ��
	void DWT(IplImage *pImage, int nLayer);
	// ��ά��ɢС���ָ�����ͨ������ͼ��
	void IDWT(IplImage *pImage, int nLayer);

	void ColorDWT(IplImage *pSrc, int nLayer);
	void ColorDWT(cv::Mat& _src,int nLayer);
	void ColorIDWT(IplImage *pSrc, int nLayer);
	void ColorIDWT(cv::Mat& _src, int nLayer);

	void GetPartOfDWT(cv::Mat& inputMat,cv::Mat& outputMat,int part,int nLayer);

};
#endif