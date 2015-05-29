#include "utils.h"
#include <fstream>
#include <omp.h>

cv::Mat im2colstep(cv::Mat& InImg, vector<int>& blockSize, vector<int>& stepSize){
	
	int r_row = blockSize[ROW_DIM] * blockSize[COL_DIM];
	int row_diff = InImg.rows - blockSize[ROW_DIM];
	int col_diff = InImg.cols - blockSize[COL_DIM];
	int r_col = (row_diff / stepSize[ROW_DIM] + 1) * (col_diff / stepSize[COL_DIM] + 1);
	cv::Mat OutBlocks(r_col, r_row, InImg.depth());

	double *p_InImg, *p_OutImg;
	int blocknum = 0;

	for(int j=0; j<=col_diff; j+=stepSize[COL_DIM]){
		for(int i=0; i<=row_diff; i+=stepSize[ROW_DIM]){

			p_OutImg = OutBlocks.ptr<double>(blocknum);
			
			for(int m=0; m<blockSize[ROW_DIM]; m++){
				
				p_InImg = InImg.ptr<double>(i + m);

				for(int l=0; l<blockSize[COL_DIM]; l++){
					p_OutImg[blockSize[ROW_DIM] * l + m] = p_InImg[j + l];
					//p_OutImg[blockSize[COL_DIM] * l + m] = p_InImg[j + l];
				}

			}
			blocknum ++;
		}
	}

	return OutBlocks;
}

cv::Mat im2col_general(cv::Mat& InImg, vector<int>& blockSize, vector<int>& stepSize){
	assert(blockSize.size() == 2 && stepSize.size() == 2);

	int channels = InImg.channels();
	
	vector<cv::Mat> layers;
	if(channels > 1)
		split(InImg, layers);
	else
		layers.push_back(InImg);
	
	cv::Mat AllBlocks = im2colstep(layers[0], blockSize, stepSize);

	int size = layers.size();
	if(size > 1){
		swap(layers[0], layers.back());
		layers.pop_back();
		for(int i=1; i<size; i++){
			hconcat(AllBlocks, im2colstep(layers[i], blockSize, stepSize), AllBlocks);	
		}
	}
	return AllBlocks.t();
}

int* getRandom(int size){
	int* rand_idx = new int[size];
	int* idx = new int[size];
	for (int i=0; i<size; i++) {
        idx[i] = i;
    }
	int temp;
	int j=0;
	for(int i=size; i>0; i--){
		temp = rand() % i;
		rand_idx[j] = idx[temp];
		j++;
		idx[temp] = idx[i - 1];
	}
	return rand_idx;
}

cv::Mat PCA_FilterBank(vector<cv::Mat>& InImg, int PatchSize, int NumFilters){
	int channels = InImg[0].channels();
	int InImg_Size = InImg.size();

	int* randIdx = getRandom(InImg_Size);

	int size = channels * PatchSize * PatchSize;
	int img_depth = InImg[0].depth();
	cv::Mat Rx = cv::Mat::zeros(size, size, img_depth);
	
	vector<int> blockSize;
	vector<int> stepSize;

	for(int i=0; i<2; i++){
		blockSize.push_back(PatchSize);
		stepSize.push_back(1);
	}
	
	cv::Mat temp;
	cv::Mat mean;
	cv::Mat temp2;
	cv::Mat temp3;
	
	int coreNum = omp_get_num_procs();//»ñµÃŽŠÀíÆ÷žöÊý
	int cols = 0;
# pragma omp parallel for default(none) num_threads(coreNum) private(temp, temp2, temp3, mean) shared(cols, Rx, InImg_Size, InImg, randIdx, blockSize, stepSize)
	for(int j=0; j<InImg_Size; j++){
	
		temp = im2col_general(InImg[randIdx[j]], blockSize, stepSize);
	
		cv::reduce(temp, mean, 0, CV_REDUCE_AVG);
		temp3.create(0, temp.cols, temp.type());
		cols = temp.cols;
		for(int i=0;i<temp.rows;i++){
			temp2 = (temp.row(i) - mean.row(0));
			temp3.push_back(temp2.row(0));
		}
	
		temp = temp3 * temp3.t();
# pragma omp critical
		Rx = Rx + temp;
	}
	Rx = Rx / (double)(InImg_Size * cols);

	cv::Mat eValuesMat;  
    cv::Mat eVectorsMat;  

	eigen(Rx, eValuesMat, eVectorsMat);  
	
	cv::Mat Filters(0, Rx.cols, Rx.depth());
	
	for(int i=0; i<NumFilters; i++)
		Filters.push_back(eVectorsMat.row(i));
	return Filters;
}	

PCA_Out_Result* PCA_output(vector<cv::Mat>& InImg, vector<int>& InImgIdx, int PatchSize, int NumFilters, cv::Mat& Filters, int threadnum){
	
	PCA_Out_Result* result = new PCA_Out_Result;
	
	int img_length = InImg.size();
	int mag = (PatchSize - 1) / 2; 
	int channels = InImg[0].channels();
	
	
	cv::Mat img;
	
	vector<int> blockSize;
	vector<int> stepSize;

	for(int i=0; i<2; i++){
		blockSize.push_back(PatchSize);
		stepSize.push_back(1);
	}

	cv::Mat temp;
	cv::Mat mean;
	cv::Mat temp2;
	cv::Mat temp3;

	int coreNum = omp_get_num_procs();//»ñµÃŽŠÀíÆ÷žöÊý
	coreNum = coreNum > threadnum ? threadnum : coreNum;
	cv::Scalar s = cv::Scalar(0);

# pragma omp parallel for default(none) num_threads(coreNum) private(img, temp, temp2, temp3, mean) shared(InImgIdx, s, blockSize, stepSize, mag, img_length, InImg, result, Filters, NumFilters)
	for(int i=0; i<img_length; i++){
		
		cv::copyMakeBorder(InImg[i], img, mag, mag, mag, mag, cv::BORDER_CONSTANT, s);
		
		temp = im2col_general(img, blockSize, stepSize);

		cv::reduce(temp, mean, 0, CV_REDUCE_AVG);
		
		temp3.create(0, temp.cols, temp.type());

		for(int i=0;i<temp.rows;i++){
			temp2 = (temp.row(i) - mean.row(0));
			temp3.push_back(temp2.row(0));
		}
# pragma omp critical 
{
		result->OutImgIdx.push_back(InImgIdx[i]);
		for(int j=0; j<NumFilters; j++){
			temp = Filters.row(j) * temp3;		
			temp = temp.reshape(0, InImg[i].cols);
			result->OutImg.push_back(temp.t());
		}
}
	}
	/*
	int size = InImgIdx.size();
	for(int i=0; i<size; i++)
		for(int j=0; j<NumFilters; j++)
			result->OutImgIdx.push_back(InImgIdx[i]);*/
	return result;
}

PCA_Train_Result* PCANet_train(vector<cv::Mat>& InImg, PCANet* PcaNet, bool is_extract_feature){
	assert(PcaNet->NumFilters.size() == PcaNet->NumStages);
	
	PCA_Train_Result* train_result = new PCA_Train_Result;
	PCA_Out_Result* out_result = new PCA_Out_Result; 
	PCA_Out_Result* temp;

	out_result->OutImg = InImg;
	int img_length = InImg.size();
	for(int i=0; i<img_length; i++)
		out_result->OutImgIdx.push_back(i);

	int64 e1 = cv::getTickCount();
	int64 eo1, eo2, eb1, eb2;
	
	for(int s=0; s<PcaNet->NumStages; s++){
		eb1 = cv::getTickCount();
		cout << " Computing PCA filter bank and its outputs at stage " << s << "..." << endl;
		train_result->Filters.push_back(PCA_FilterBank(out_result->OutImg, PcaNet->PatchSize, PcaNet->NumFilters[s]));
		eb2 = cv::getTickCount();
		cout <<" stage"<<s<<" PCA_FilterBank time: "<<(eb2 - eb1)/ cv::getTickFrequency()<<endl;

		eo1 = cv::getTickCount();
		if(s != PcaNet->NumStages - 1){
			temp = PCA_output(out_result->OutImg, out_result->OutImgIdx, PcaNet->PatchSize, 
												PcaNet->NumFilters[s], train_result->Filters[s], omp_get_num_procs());
			delete out_result;
			out_result = temp;
		}
		eo2 = cv::getTickCount();
		cout <<" stage"<<s<<" output time: "<<(eo2 - eo1)/ cv::getTickFrequency()<<endl;
	}
	int64 e2 = cv::getTickCount();
	double time = (e2 - e1)/ cv::getTickFrequency();
	cout <<"\n totle FilterBank time: "<<time<<endl;

	InImg.clear();
	vector<cv::Mat>().swap(InImg);

	vector<cv::Mat> tempF;
	int end = PcaNet->NumStages - 1;
	int outIdx_length = out_result->OutImgIdx.size();

	if(is_extract_feature){

		vector<cv::Mat>::const_iterator first = out_result->OutImg.begin();
		vector<cv::Mat>::const_iterator last = out_result->OutImg.begin();
		
		vector<cv::Mat> features;
		Hashing_Result* hashing_r;

		int coreNum = omp_get_num_procs();//»ñµÃŽŠÀíÆ÷žöÊý
		e1 = cv::getTickCount();
# pragma omp parallel for default(none) num_threads(coreNum) private(temp, hashing_r) shared(features, out_result, PcaNet, first, last, outIdx_length, img_length, train_result, end)
		for(int i=0; i<img_length; i++){
			vector<cv::Mat> subInImg(first + i * PcaNet->NumFilters[end], last + (i + 1) * PcaNet->NumFilters[end]);
			vector<int> subIdx;
			/*for(int j=0; j<outIdx_length; j++){
				if(out_result->OutImgIdx[j] == i) subIdx.push_back(1);
				else subIdx.push_back(0);
			}*/
			for(int j=0; j< PcaNet->NumFilters[end]; j++)
				subIdx.push_back(j);
			
			temp = PCA_output(subInImg, subIdx, PcaNet->PatchSize, 
								PcaNet->NumFilters[end], train_result->Filters[end], 2);
			
			hashing_r = HashingHist(PcaNet, temp->OutImgIdx, temp->OutImg);	
			
#pragma omp critical 
{
			features.push_back(hashing_r->Features);
			train_result->feature_idx.push_back(out_result->OutImgIdx[i]);
}
			delete hashing_r;
			delete temp;
			subIdx.clear();
			vector<int>().swap(subIdx);
		}
		e2 = cv::getTickCount();
		time = (e2 - e1)/ cv::getTickFrequency();
		cout <<"\n hasing time: "<<time<<endl;
		
		//out_result->OutImg.clear();
		//vector<cv::Mat>().swap(out_result->OutImg);
		delete out_result;

		int size = features.size();
		if(size > 0){

			train_result->Features.create(0, features[0].cols, features[0].type());
			for(int i=0 ;i<size; i++){
				train_result->Features.push_back(features[i]);
			}

			/*
			train_result->Features = features[0];
			for(int i=1 ;i<size; i++){
				vconcat(train_result->Features, features[i], train_result->Features);
			}*/
		}

		features.clear();
		vector<cv::Mat>().swap(features);
	}
	
	//if(temp != NULL)
	//	delete temp;
	
	return train_result;
}

double round(double r){  
    return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);  
}  

Hashing_Result* HashingHist(PCANet* PcaNet, vector<int>& ImgIdx, vector<cv::Mat>& Imgs){
	Hashing_Result* ha_result = new Hashing_Result;

	int length = Imgs.size();
	int NumFilters =  PcaNet->NumFilters[PcaNet->NumStages - 1];
	int NumImgin0 = length / NumFilters;

	cv::Mat T;
	int row = Imgs[0].rows;
	int col = Imgs[0].cols;
	int depth = Imgs[0].depth();

	vector<double> map_weights;
	cv::Mat temp;
	for(int i=NumFilters - 1; i>=0; i--)
		map_weights.push_back(pow(2.0, (double)i));

	vector<int> Ro_BlockSize;
	double rate = 1 - PcaNet->BlkOverLapRatio;
	for(int i=0 ;i<PcaNet->HistBlockSize.size(); i++)
		Ro_BlockSize.push_back(round(PcaNet->HistBlockSize[i] * rate));
	
	
	cv::Mat BHist;

	int ImgIdx_length = ImgIdx.size();
	int* new_idx = new int[ImgIdx_length];
	for(int i=0; i<ImgIdx_length; i++)
		new_idx[ImgIdx[i]] = i;

	for(int i=0; i<NumImgin0; i++){
		T = cv::Mat::zeros(row, col, depth);	


		for(int j=0; j<NumFilters; j++){
			temp = Heaviside(Imgs[NumFilters * new_idx[i] + j]);
			temp = temp * map_weights[j];
			T = T + temp;
		}
		
		temp = im2col_general(T, PcaNet->HistBlockSize, Ro_BlockSize); 
		temp = Hist(temp, (int)(pow(2.0, NumFilters)) - 1);

		temp = bsxfun_times(temp, NumFilters);

		if(i == 0) BHist = temp;
		else hconcat(BHist, temp, BHist);
	}
	
	int rows = BHist.rows;
	int cols = BHist.cols;

	ha_result->Features.create(1, rows * cols, BHist.type()); 

	double *p_Fe = ha_result->Features.ptr<double>(0);
	double *p_Hi;
	for(int i=0; i<rows; i++){
		p_Hi = BHist.ptr<double>(i);
		for(int j=0; j<cols; j++){
			p_Fe[j * rows + i] = p_Hi[j];
		}
	}
	return ha_result;
}


cv::Mat Heaviside(cv::Mat& X){
	int row = X.rows;
	int col = X.cols;
	int depth = X.depth();

	cv::Mat H(row, col, depth);

	double *p_X, *p_H;

////# pragma omp parallel for default(none) num_threads(4) private(p_X, p_H) shared(X, H, row, col)
	for(int i=0; i<row; i++){
		p_X = X.ptr<double>(i);
		p_H = H.ptr<double>(i);
		
		for(int j=0; j<col; j++){
			if(p_X[j] > 0) p_H[j] = 1;
			else p_H[j] = 0;
		}
	}
	return H;
}

cv::Mat Hist(cv::Mat& mat, int Range){
	cv::Mat temp = mat.t();
	int row = temp.rows;
	int col = temp.cols;
	int depth = temp.depth();
	cv::Mat Hist = cv::Mat::zeros(row, Range + 1, depth);

	double *p_M, *p_H;

////# pragma omp parallel for default(none) num_threads(4) private(p_M, p_H) shared(temp, Hist, row, col)
	for(int i=0; i<row; i++){
		p_M = temp.ptr<double>(i);
		p_H = Hist.ptr<double>(i);
		
		for(int j=0; j<col; j++){
			p_H[(int)p_M[j]] += 1;
		}
	}
	
	temp = Hist.t();

	return temp;
}


cv::Mat bsxfun_times(cv::Mat& BHist, int NumFilters){
	
	double *p_BHist;
	int row = BHist.rows;
	int col = BHist.cols;

	double* sum = new double[col];
	for(int i=0; i<col; i++)
		sum[i] = 0;
	
	for(int i=0; i<row; i++){
		p_BHist = BHist.ptr<double>(i);
		for(int j=0; j<col; j++)
			sum[j] += p_BHist[j];
	}
	double p = pow(2.0, NumFilters);

////# pragma omp parallel for default(none) num_threads(4) shared(col, sum, p)
	for(int i=0; i<col; i++)
		sum[i] = p / sum[i];

////# pragma omp parallel for default(none) num_threads(4) private(p_BHist) shared(col, row, sum, BHist)
	for(int i=0; i<row; i++){
		p_BHist = BHist.ptr<double>(i);
		for(int j=0; j<col; j++)
			p_BHist[j] *= sum[j];
	}

	return BHist;
}
