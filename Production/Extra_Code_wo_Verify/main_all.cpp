#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h> 
#include <time.h> //需包含该头文件，或者包含<time.h>

using namespace cv;
namespace F = torch::nn::functional;
using namespace std;
using namespace torch::indexing;

int main2()
{
	std::string model_pb = "model.pt";
	std::string img_path = "21_cut.bmp";
	int nW = 28;
	int nH = 8;
	auto mean_info = torch::tensor({ {{0.47008159783297326, 0.3302275149738268, 0.26258365359780844}} });
	auto std_info = torch::tensor({ {{0.043578123562233576, 0.03262873283790422, 0.030290686596445255}} });

	auto model = torch::jit::load(model_pb);
	model.eval();

	// const clock_t begin_time = clock();
	//读取图片
	Mat image = cv::imread(img_path);
	auto src_w = image.cols;
	auto src_h = image.rows;
	int dst_h = src_h - (src_h % nH);
	int dst_w = src_w - (src_w % nW);
	int nPatchH = dst_h / nH;
	int nPatchW = dst_w / nW;
	Mat dst_rgb;
	resize(image, dst_rgb,Size(dst_w, dst_h));
	cvtColor(dst_rgb, dst_rgb, COLOR_BGR2RGB);
	auto input_tensor = torch::from_blob(dst_rgb.data, {dst_h, dst_w, 3 }, torch::kByte);
	input_tensor = input_tensor.toType(torch::kFloat32);
	input_tensor = input_tensor.div(255.).sub(mean_info).div(std_info);
	input_tensor = input_tensor.view({ nH, nPatchH, nW, nPatchW,  3 });
	input_tensor = input_tensor.permute({ 0, 2, 4, 1, 3 });
	input_tensor = input_tensor.contiguous().view({ -1, 3, nPatchH, nPatchW });

	input_tensor = F::interpolate(input_tensor, F::InterpolateFuncOptions().size(std::vector<int64_t>{ 160, 80}));
	const clock_t begin_time = clock();
	auto pred_label = model.forward({ input_tensor}).toTensor();
	float seconds = float(clock() - begin_time) / 1000;
	pred_label = torch::argmax(pred_label, 1);
	pred_label = pred_label.view({ nH, nW, 80, 40 });
	pred_label = pred_label.permute({ 0, 2, 1, 3 });
	pred_label = pred_label.contiguous().view({ (nH * 80), (nW * 40) });

	pred_label = (255 * pred_label).toType(torch::kByte);
	cv::Mat pred_label_mat(cv::Size(40 * nW, 80 * nH), CV_8U, pred_label.data_ptr());
	resize(pred_label_mat, pred_label_mat, Size(src_w, src_h));

	//float seconds = float(clock() - begin_time) / 1000; //此处1000指的是每秒为1000个时钟周期，所以要想得到以秒为单位的时间，需要除以1000.
	cout << "Time:" << seconds << endl;
	cv::imwrite("show.png", pred_label_mat);
	system("Pause");
	return 0;
}