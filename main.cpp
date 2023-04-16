#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;


class LDC
{
public:
	LDC(string modelpath);
	void detect(Mat srcimg, Mat& average_image, Mat& fuse_image);
private:
	vector<float> input_image_;
	int inpWidth;
	int inpHeight;
	int outWidth;
	int outHeight;
	int num_outs;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Lightweight Dense CNN for Edge Detection");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

LDC::LDC(string model_path)
{
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->outHeight = output_node_dims[0][2];
	this->outWidth = output_node_dims[0][3];
	this->num_outs = output_node_dims.size();
}

void LDC::detect(Mat srcimg, Mat& average_image, Mat& fuse_image)
{
	Mat dstimg;
	resize(srcimg, dstimg, Size(this->inpWidth, this->inpHeight));
	this->input_image_.resize(this->inpWidth * this->inpHeight * dstimg.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < this->inpHeight; i++)
		{
			for (int j = 0; j < this->inpWidth; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * this->inpHeight * this->inpWidth + i * this->inpWidth + j] = pix;
			}
		}
	}
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	
	for (int n = 0; n < num_outs; n++)
	{
		float* pred = ort_outputs[n].GetTensorMutableData<float>();
		/*this->outHeight = output_node_dims[n][2];
		this->outWidth = output_node_dims[n][3];*/
		Mat result(outHeight, outWidth, CV_32FC1, pred);
		/*for (int i = 0; i < this->outHeight; i++)
		{
			for (int j = 0; j < this->outWidth; j++)
			{
				float pix = result.ptr<float>(i)[j];
				result.at<float>(i, j) = 1 / (1 + expf(-pix));   ///sigmoid
			}
		}*/
		Mat TmpExp;
		cv::exp(-result, TmpExp);   ///不用for循环
		Mat mask = 1.0 / (1.0 + TmpExp);


		double min_value, max_value;
		minMaxLoc(mask, &min_value, &max_value, 0, 0);
		mask = (mask - min_value) * 255.0 / (max_value - min_value + 1e-12);
		mask.convertTo(mask, CV_8UC1);
		bitwise_not(mask, mask);
		resize(mask, mask, Size(srcimg.cols, srcimg.rows));
		
		accumulate(mask, average_image);  //将所有图像叠加
		fuse_image = mask;
	}
	average_image = average_image / (float)num_outs; //求出平均图像
	average_image.convertTo(average_image, CV_8UC1);
}

int main()
{
	LDC mynet("weights/LDC_640x360.onnx");  
	string imgpath = "images/IMG_2567.jpg";
	Mat srcimg = imread(imgpath);

	Mat average_image = Mat::zeros(srcimg.rows, srcimg.cols, CV_32FC1);
	Mat fuse_image(srcimg.rows, srcimg.cols, CV_8UC1);
	mynet.detect(srcimg, average_image, fuse_image);
	

	namedWindow("LDC-srcimg", WINDOW_NORMAL);
	imshow("LDC-srcimg", srcimg);
	namedWindow("LDC-average_image", WINDOW_NORMAL);
	imshow("LDC-average_image", average_image);
	namedWindow("LDC-fuse_image", WINDOW_NORMAL);
	imshow("LDC-fuse_image", fuse_image);
	waitKey(0);
	destroyAllWindows();
}