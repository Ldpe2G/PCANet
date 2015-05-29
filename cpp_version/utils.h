#include <opencv2\opencv.hpp>
#include <vector>
#include <math.h>
using namespace std;



const int ROW_DIM = 0;
const int COL_DIM = 1;

cv::Mat im2colstep(cv::Mat& InImg, vector<int>& blockSize, vector<int>& stepSize);

cv::Mat im2col_general(cv::Mat& InImg, vector<int>& blockSize, vector<int>& stepSize);

cv::Mat PCA_FilterBank(vector<cv::Mat>& InImg, int PatchSize, int NumFilters);

typedef struct {
	vector<cv::Mat> OutImg;
	vector<int> OutImgIdx;
} PCA_Out_Result;

PCA_Out_Result* PCA_output(vector<cv::Mat>& InImg, vector<int>& InImgIdx, int PatchSize, int NumFilters, cv::Mat& Filters, int threadnum);

typedef struct {
	//SparseMat Features;
	cv::Mat Features;
	vector<int> feature_idx;
	vector<cv::Mat> Filters;
	vector<int> BlkIdx;
} PCA_Train_Result;

typedef struct {
	int NumStages;
	int PatchSize;
	vector<int> NumFilters;
	vector<int> HistBlockSize;
	double BlkOverLapRatio;
} PCANet;

PCA_Train_Result* PCANet_train(vector<cv::Mat>& InImg, PCANet* PcaNet, bool is_extract_feature);

typedef struct {
	//SparseMat Features;
	cv::Mat Features;
	vector<int> BlkIdx;
} Hashing_Result;

Hashing_Result* HashingHist(PCANet* PcaNet, vector<int>& ImgIdx, vector<cv::Mat>& Imgs);

cv::Mat Heaviside(cv::Mat& X);

/**
*	the range of the histogram is 0 to Range
*/
cv::Mat Hist(cv::Mat& mat, int Range);

cv::Mat bsxfun_times(cv::Mat& BHist, int NumFilters);
//SparseMat bsxfun_times(SparseMat BHist, int cols, int NumFilters);

cv::Mat Mat_mul_MT(cv::Mat& mat);
