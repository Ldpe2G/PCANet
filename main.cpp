#include <opencv2\opencv.hpp>
#include "utils.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include <string>
using namespace std;
using namespace cv;


int main(int argc, char** argv){
	const int DIR_LENGTH = 256;
	const int DIR_NUM = 7;
	//input image size height: 60, width: 48
	// 路径根据自己的情况来修改即可
	const char *dir[DIR_NUM] = {
		"..\\datas\\train\\1\\1_",
		"..\\datas\\train\\2\\2_",
		"..\\datas\\train\\3\\3_",
		"..\\datas\\train\\4\\4_",
		"..\\datas\\train\\5\\5_",
		"..\\datas\\train\\6\\6_",
		"..\\datas\\train\\7\\7_"
	};

	const char *test_dir[DIR_NUM] = {
		"..\\datas\\test\\1\\1_",
		"..\\datas\\test\\2\\2_",
		"..\\datas\\test\\3\\3_",
		"..\\datas\\test\\4\\4_",
		"..\\datas\\test\\5\\5_",
		"..\\datas\\test\\6\\6_",
		"..\\datas\\test\\7\\7_"
	};
	
	char path[DIR_LENGTH];
	IplImage* img;
	IplImage *change;
	vector<cv::Mat> InImgs;
	cv::Mat* bmtx;
	cv::Mat* histo; // histogram equalizaion  
	
	const int train_num = 40;
	const int NUM = DIR_NUM * train_num;

	
	float *labels = new float[NUM]; 
	int x = 0;
	for(int i=1; i<train_num + 1; i++){
		for(int j=1; j<=DIR_NUM; j++){
			sprintf(path, "%s%d%s", dir[j-1], i, ".jpg");
			img = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);
			
			change = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, img->nChannels);
			cvConvertScale(img, change, 1.0/255, 0);
			bmtx = new cv::Mat(change);
			InImgs.push_back(*bmtx);
			labels[x] = (float)j;
			x++;
		}
	}
	
	
	vector<int> NumFilters;
	NumFilters.push_back(8);
	NumFilters.push_back(8);
	vector<int> blockSize;
	blockSize.push_back(12);   //  height / 4
	blockSize.push_back(10);    //  width / 4
	

	PCANet pcaNet = {
		2,
		7,
		NumFilters,
		blockSize,
		0.5
	};
	
	cout <<"\n ====== PCANet Training ======= \n"<<endl;
	int64 e1 = cv::getTickCount();
	PCA_Train_Result* result = PCANet_train(InImgs, &pcaNet, true);
	int64 e2 = cv::getTickCount();
	double time = (e2 - e1)/ cv::getTickFrequency();
	cout <<" PCANet Training time: "<<time<<endl;

	
	FileStorage fs("..\\model\\all_age_filters.xml", FileStorage::WRITE);  
	fs<<"filter1"<<result->Filters[0]<<"filter2"<<result->Filters[1];  
	fs.release();  

	///  svm  train  //////////
	cout <<"\n ====== Training Linear SVM Classifier ======= \n"<<endl;

	float *new_labels = new float[NUM];
	int size = result->feature_idx.size();
	for(int i=0; i<size; i++)
		new_labels[i] = labels[result->feature_idx[i]];
	
	

	Mat labelsMat(NUM, 1, CV_32FC1, new_labels);  

	result->Features.convertTo(result->Features, CV_32F);

    //设置支持向量机的参数  
    CvSVMParams params;  
	params.svm_type    = CvSVM::C_SVC;//SVM类型
	params.C = 1;
	//params.nu = 0.8;
	params.kernel_type = CvSVM::LINEAR;//核函数类型
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);//终止准则函数：当迭代次数达到最大值时终止  
  
    //训练SVM  
    //建立一个SVM类的实例  
    CvSVM SVM;  

	e1 = cv::getTickCount();
    //训练模型，参数为：输入数据、响应、XX、XX、参数（前面设置过）  
	SVM.train(result->Features, labelsMat, Mat(), Mat(), params);  
	///  svm  train    /////
	e2 = cv::getTickCount();
	time = (e2 - e1)/ cv::getTickFrequency();
	cout <<" svm training complete, time usage: "<<time<<endl;
	
	SVM.save("..\\model\\all_age_svm.xml");

	cout <<"\n ====== PCANet Testing ======= \n"<<endl;

	vector<Mat> testImg;
	vector<int> testLabel;
	vector<string> names;
	string *t;

	int testNum = 24;

	for(int i=41; i<65; i++){
		for(int j=0; j<DIR_NUM; j++){
			sprintf(path, "%s%d%s", test_dir[j], i, ".jpg");
			img = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);
			t = new string(path);
			names.push_back(*t);
			change = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, img->nChannels);
			cvConvertScale(img, change, 1.0/255, 0);
			bmtx = new cv::Mat(change);
			testImg.push_back(*bmtx);
			testLabel.push_back(j + 1);
		}
	}
	int testSIze = testImg.size();
	Hashing_Result* hashing_r;
	PCA_Out_Result *out;

	float all = DIR_NUM * testNum;
	float correct = 0;
	int coreNum = omp_get_num_procs();//获得处理器个数

	float *corrs = new float[DIR_NUM];
	for(int i=0; i<DIR_NUM; i++)
		corrs[i] = 0;



	e1 = cv::getTickCount();
# pragma omp parallel for default(none) num_threads(coreNum) private(out, hashing_r) shared(names, corrs, correct, testLabel, SVM, pcaNet, testSIze, testImg, result)
	for(int i=0; i<testSIze; i++){
		out = new PCA_Out_Result;
		out->OutImgIdx.push_back(0);
		out->OutImg.push_back(testImg[i]);
		out = PCA_output(out->OutImg, out->OutImgIdx, pcaNet.PatchSize, 
			pcaNet.NumFilters[0], result->Filters[0], 2);
		for(int j=1; j<pcaNet.NumFilters[1]; j++)
			out->OutImgIdx.push_back(j);

		out = PCA_output(out->OutImg, out->OutImgIdx, pcaNet.PatchSize, 
			pcaNet.NumFilters[1], result->Filters[1], 2);		
		hashing_r = HashingHist(&pcaNet, out->OutImgIdx, out->OutImg);	
		hashing_r->Features.convertTo(hashing_r->Features, CV_32F);
		float pred = SVM.predict(hashing_r->Features);
#pragma omp critical 
{		
		//printf("predict: %f, testLabel: %d\n", pred, testLabel[i]);
		if(pred == testLabel[i]){
			corrs[testLabel[i]-1]++;
			correct ++;
		}
}
		delete out;
	}


	e2 = cv::getTickCount();
	time = (e2 - e1)/ cv::getTickFrequency();
	cout <<" test time usage: "<<time<<endl;
	cout <<"all precise: "<<correct / all<<endl;
	for(int i=0; i<DIR_NUM; i++)
		cout << "person" <<i+1<<" precise: "<<corrs[i] / testNum<<endl;
	cout <<"test images num for each class: "<<testNum<<endl;
	
	return 0;
}
