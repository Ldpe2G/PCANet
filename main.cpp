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
	const char *dir[DIR_NUM] = {
		"F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\train\\children1to5\\1_",
		"F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\train\\children4to8\\1_",
		"F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\train\\children7to10\\1_",
		"F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\train\\teens\\1_",
		"F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\train\\youth\\1_",
		"F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\train\\middleage\\1_",
		"F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\train\\elderly\\1_"
	};
	
	char path[DIR_LENGTH];
	IplImage* img;
	IplImage *change;
	vector<cv::Mat> InImgs;
	cv::Mat* bmtx;
	cv::Mat* histo; // histogram equalizaion  
	
	const int train_num = 40;
	const int NUM = DIR_NUM * train_num;

	const int e_t_num = 10;
	const int E_NUM = e_t_num * 3;

	
	float *labels = new float[NUM + E_NUM]; 
	int x = 0;
	for(int i=1; i<train_num + 1; i++){
		for(int j=1; j<=DIR_NUM; j++){
			if(i < 10) sprintf(path, "%s%d%d%s", dir[j-1], 0, i, ".jpg");
			else sprintf(path, "%s%d%s", dir[j-1], i, ".jpg");
			img = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);
			
			change = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, img->nChannels);
			cvConvertScale(img, change, 1.0/255, 0);
			bmtx = new cv::Mat(change);
			InImgs.push_back(*bmtx);
			labels[x] = (float)j;
			x++;
		}
	}
	
	for(int i=train_num+1; i<train_num + e_t_num + 1; i++){
		for(int j=1; j<=3; j++){
			if(i < 10) sprintf(path, "%s%d%d%s", dir[j-1], 0, i, ".jpg");
			else sprintf(path, "%s%d%s", dir[j-1], i, ".jpg");
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
	//blockSize.push_back(25);
	//blockSize.push_back(20);
	blockSize.push_back(15);   //  height / 4
	blockSize.push_back(12);    //  width / 4
	//blockSize.push_back(14);
	//blockSize.push_back(9);

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

	
	FileStorage fs("F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\train\\all_age_filters_60x48_2.xml", FileStorage::WRITE);  
	fs<<"filter1"<<result->Filters[0]<<"filter2"<<result->Filters[1];  
	fs.release();  

	///  svm  train  //////////
	cout <<"\n ====== Training Linear SVM Classifier ======= \n"<<endl;

	float *new_labels = new float[NUM + E_NUM];
	int size = result->feature_idx.size();
	for(int i=0; i<size; i++)
		new_labels[i] = labels[result->feature_idx[i]];
	
	

	Mat labelsMat(E_NUM + NUM, 1, CV_32FC1, new_labels);  

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
	
	SVM.save("F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\train\\all_age_svm_60x48_2.xml");

	result->Features.deallocate();
	
	
	cout <<"\n ====== PCANet Testing ======= \n"<<endl;

	vector<Mat> testImg;
	vector<int> testLabel;
	vector<string> names;
	string *t;

	int testNum = 29;

	for(int i=31; i<31 + testNum; i++){
		for(int j=0; j<DIR_NUM; j++){
			if(i < 10) sprintf(path, "%s%d%d%s", dir[j], 0, i, ".jpg");
			else sprintf(path, "%s%d%s", dir[j], i, ".jpg");
			img = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);
			t = new string(path);
			names.push_back(*t);
			change = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, img->nChannels);
			cvConvertScale(img, change, 1.0/255, 0);
			bmtx = new cv::Mat(change);
			testImg.push_back(*bmtx);
			//testLabel.push_back(1);
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
		}else printf("%d, pred: %f , label:%d \n", i/DIR_NUM+31, pred, testLabel[i]);
}
		delete out;
	}


	e2 = cv::getTickCount();
	time = (e2 - e1)/ cv::getTickFrequency();
	cout <<" test time usage: "<<time<<endl;
	cout <<"all precise: "<<correct / all<<endl;
	for(int i=0; i<DIR_NUM; i++)
		cout <<"baby"<<i+1<<" precise: "<<corrs[i] / testNum<<endl;
	cout <<"test images num for each class: "<<testNum<<endl;
	
	
	//face detect test
	/*
	CvSVM Child_Adult_svm;
	vector<Mat> Child_Adult_filters;

	Child_Adult_svm.load("F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\train\\all_age_svm_60x48.xml");
	FileStorage fs2("F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\train\\all_age_filters_60x48.xml", FileStorage::READ);
	Mat Filter1;
	Mat Filter2;
	fs2["filter1"] >> Filter1;
	fs2["filter2"] >> Filter2;
	Child_Adult_filters.push_back(Filter1);
	Child_Adult_filters.push_back(Filter2);

	vector<Mat> testImg;
	vector<IplImage> fakeImg;
	for(int i=1; i<375; i++){
		//sprintf(path, "%s%d%s", "F:\\lwf\\AllPicture\\AllFaces2\\Stars\\1", i, ".jpg");
		if(i < 10) sprintf(path, "%s%d%d%s",  "F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\test\\children4to8\\1_", 0, i, ".jpg");
		else sprintf(path, "%s%d%s",  "F:\\lwf\\AllPicture\\Final\\925New\\notRotate\\test\\children4to8\\1_", i, ".jpg");

		img = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);
		
		fakeImg.push_back(*img);

		change = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, img->nChannels);
		cvConvertScale(img, change, 1.0/255, 0);
		bmtx = new cv::Mat(change);
		testImg.push_back(*bmtx);

	}
	int testSIze = testImg.size();
	Hashing_Result* hashing_r;
	PCA_Out_Result *out;

	float all = 1487;
	float correct = 0;
	int *corrects = new int[DIR_NUM];
	for(int i=0; i<DIR_NUM; i++)
		corrects[i] = 0;

	int coreNum = omp_get_num_procs();//获得处理器个数
	int testclass = 1;

	int64 e1 = cv::getTickCount();

# pragma omp parallel for default(none) num_threads(coreNum) private(out, hashing_r) shared(corrects, testclass, correct, SVM, pcaNet, testSIze, testImg, Child_Adult_svm, Child_Adult_filters, fakeImg)
	for(int i=0; i<testSIze; i++){
		out = new PCA_Out_Result;
		out->OutImgIdx.push_back(0);
		out->OutImg.push_back(testImg[i]);
		out = PCA_output(out->OutImg, out->OutImgIdx, pcaNet.PatchSize, 
			pcaNet.NumFilters[0], Child_Adult_filters[0], 2);
		for(int j=1; j<pcaNet.NumFilters[1]; j++)
			out->OutImgIdx.push_back(j);

		out = PCA_output(out->OutImg, out->OutImgIdx, pcaNet.PatchSize, 
			pcaNet.NumFilters[1], Child_Adult_filters[1], 2);		
		hashing_r = HashingHist(&pcaNet, out->OutImgIdx, out->OutImg);	
		hashing_r->Features.convertTo(hashing_r->Features, CV_32F);
		float pred = Child_Adult_svm.predict(hashing_r->Features);
#pragma omp critical 
{		

		corrects[(int)pred - 1] ++;
		//if(pred < 3){
			//printf("predict: %f, num: %d\n", pred, i+1);
		//	correct ++;
		//}else{
			char* path = new char[256];
			
			//printf("predict: %f, num: %d\n", pred, i+1);
			if(pred == 1)
				sprintf(path, "%s%d%s%d.jpg", "F:\\lwf\\AllPicture\\Final\\NewTest48\\Error_40x28\\youth\\children1to4\\", testclass, "_", i+1);
			if(pred == 2)
				sprintf(path, "%s%d%s%d.jpg", "F:\\lwf\\AllPicture\\Final\\NewTest48\\Error_40x28\\youth\\children4to7\\", testclass, "_", i+1);
			if(pred == 3)
				sprintf(path, "%s%d%s%d.jpg", "F:\\lwf\\AllPicture\\Final\\NewTest48\\Error_40x28\\youth\\children7to10\\", testclass, "_", i+1);
			if(pred == 4)
				sprintf(path, "%s%d%s%d.jpg", "F:\\lwf\\AllPicture\\Final\\NewTest48\\Error_40x28\\youth\\teens\\", testclass, "_", i+1);
			if(pred == 5)
				sprintf(path, "%s%d%s%d.jpg", "F:\\lwf\\AllPicture\\Final\\NewTest48\\Error_40x28\\youth\\youth\\", testclass, "_", i+1);
			if(pred == 6)
				sprintf(path, "%s%d%s%d.jpg", "F:\\lwf\\AllPicture\\Final\\NewTest48\\Error_40x28\\youth\\middle\\", testclass, "_", i+1);
			if(pred == 7)
				sprintf(path, "%s%d%s%d.jpg", "F:\\lwf\\AllPicture\\Final\\NewTest48\\Error_40x28\\youth\\elderly\\", testclass, "_", i+1);
			
			//cvSaveImage(path, &fakeImg[i]);
		//}//if(pred > 0)
		//	correct ++;
}
		delete out;
	}
	int64 e2 = cv::getTickCount();
	double time = (e2 - e1)/ cv::getTickFrequency();
	
	for(int i=0; i<DIR_NUM; i++)
		cout <<corrects[i]<<endl;
	//cout <<" precise: "<<correct <<endl;
	cout <<"Testing time: "<<time<<endl;
	
	int t;
	cin >>t;*/
	return 0;
}
