#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h> 
#include <time.h> //需包含该头文件，或者包含<time.h>

using namespace cv;
using namespace std;

int main1()
{
	//定义使用使用的device
	torch::DeviceType device;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Predicting on GPU." << std::endl;
		device = torch::kCUDA;
	}
	else
	{
		std::cout << "Predicting on CPU." << std::endl;
		device = torch::kCPU;
	}
	auto device_type = torch::Device(device);

	//加载模型
	std::string model_pb = "G:\\Coding\\BetelnutDetect\\Production\\model.pt";

	auto model = torch::jit::load(model_pb);
	model.to(device_type);
	model.eval();

	const clock_t begin_time = clock();
	//读取图片
	std::string img_path = "G:\\Coding\\BetelnutDetect\\cut_11.bmp";
	auto width_num = 28;
	auto height_num = 8;
	Mat image = cv::imread(img_path);
	cvtColor(image, image, COLOR_BGR2RGB);
	auto width = image.cols;
	auto height = image.rows;
	int width_step = width / width_num;
	int height_step = height / height_num;

	cv::Mat img_patch_transformed;
	for (int i = 0; i <= width_num; i++) {
		auto width_start = i * width_step;
		auto width_end = ((i + 1) * width_step < width ? ((i + 1) * width_step) : width);
		for (int j = 0; j <= height_num; j++) {
			// cout << j << endl;
			auto height_start = j * height_step;
			auto height_end = ((j + 1) * height_step < height ? ((j + 1) * height_step) : height);
			auto img_patch = image(cv::Range(height_start, height_end), cv::Range(width_start, width_end)).clone();
			/*cv::imshow("show", img_patch);
			cv::waitKey(0);*/

			//缩放至指定大小
			cv::resize(img_patch, img_patch_transformed, cv::Size(80, 160));
			
			//转成张量
			auto input_tensor = torch::from_blob(img_patch_transformed.data, { img_patch_transformed.rows, img_patch_transformed.cols, 3 }, 
				torch::kByte);
			input_tensor = input_tensor.permute({ 2,0,1 }).toType(torch::kFloat32);
			// std::cout << input_tensor.size(0) << input_tensor.size(1);
			input_tensor[0] = input_tensor[0].div_(255).sub_(0.2619703004633249).div_(0.021910381946245575);
			input_tensor[1] = input_tensor[1].div_(255).sub_(0.16063937884372356).div_(0.015363815458932483);
			input_tensor[2] = input_tensor[2].div_(255).sub_(0.11861206329461361).div_(0.01316140398841059);
			// std::cout << input_tensor[0][1];
			input_tensor = input_tensor.unsqueeze(0);
			//前向传播
			auto output = model.forward({ input_tensor.to(device_type) }).toTensor();
			//std::cout << output.size(0) << output.size(1) << output.size(2) << output.size(3);
			output = torch::argmax(output, 1);
			auto pred_label = output.mul_(255).squeeze_(0).toType(torch::kByte);
			pred_label = pred_label.permute({ 1,0 });
			
			cv::Mat pred_label_mat(cv::Size(40, 80), CV_8U, pred_label.data_ptr());
			/*cv::imshow("show", pred_label_mat);
			cv::waitKey(0);*/

			cv::resize(pred_label_mat, pred_label_mat, cv::Size(img_patch.cols, img_patch.rows));
			cv::morphologyEx(pred_label_mat, pred_label_mat, cv::MORPH_CLOSE, cv::Mat(3, 3, CV_8U, cv::Scalar(1)));
			std::vector< std::vector< cv::Point> > contours;
			cv::findContours(pred_label_mat, contours, cv::noArray(), cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			int area = 0;
			int id = 0;
			for (int k = 0; k < contours.size(); k++) {
				auto area_tmp = cv::contourArea(contours.at(k));
				if (area_tmp > area) {
					area = area_tmp;
					id = k;
				}
			}
			cv::drawContours(img_patch, contours, id, cv::Scalar(255, 0, 0), 1, 8);
			Mat tmp = image(cv::Range(height_start, height_end), cv::Range(width_start, width_end));
			img_patch.copyTo(tmp);
			cv::cvtColor(image, image, cv::COLOR_RGB2BGR); 
			
		}
	}
	float seconds = float(clock() - begin_time) / 1000; //此处1000指的是每秒为1000个时钟周期，所以要想得到以秒为单位的时间，需要除以1000.
	cout << "Time:" << seconds << endl;
	system("Pause");
	cv::imwrite("show.bmp", image);
	return 0;
}