#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h> 
#include <time.h>
#include <inference_engine.hpp>

using namespace cv;
namespace F = torch::nn::functional;
using namespace std;
using namespace torch::indexing;
using namespace InferenceEngine;

int main()
{

	std::string model_xml = "model.xml";
	std::string model_bin = "model.bin";
	std::string img_path = "21_cut.bmp";
	int nW = 28;
	int nH = 8;
	auto mean_info = torch::tensor({ {{0.47008159783297326, 0.3302275149738268, 0.26258365359780844}} });
	auto std_info = torch::tensor({ {{0.043578123562233576, 0.03262873283790422, 0.030290686596445255}} });
	const clock_t time1 = clock();
	InferenceEngine::Core core("plugins.xml");
	CNNNetwork network = core.ReadNetwork(model_xml, model_bin);
	network.setBatchSize(nH * nW);

	InferenceEngine::InputsDataMap input_info = network.getInputsInfo();
	for (auto& item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(InferenceEngine::Precision::FP32);
		input_data->setLayout(InferenceEngine::Layout::NCHW);
		/*input_data->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
		input_data->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);*/
	}
	InferenceEngine::OutputsDataMap output_info = network.getOutputsInfo();
	for (auto& item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(InferenceEngine::Precision::FP32);
		output_data->setLayout(InferenceEngine::Layout::NCHW);
	}

	ExecutableNetwork executable_network;
	executable_network = core.LoadNetwork(network, "CPU");

	const clock_t time2 = clock();

	Mat image = cv::imread(img_path);
	auto src_w = image.cols;
	auto src_h = image.rows;
	int dst_h = src_h - (src_h % nH);
	int dst_w = src_w - (src_w % nW);
	int nPatchH = dst_h / nH;
	int nPatchW = dst_w / nW;
	Mat dst_rgb;
	resize(image, dst_rgb, Size(dst_w, dst_h));
	cvtColor(dst_rgb, dst_rgb, COLOR_BGR2RGB);
	torch::Tensor input_tensor = torch::from_blob(dst_rgb.data, { dst_h, dst_w, 3 }, torch::kByte);
	input_tensor = input_tensor.toType(torch::kFloat32);
	input_tensor = input_tensor.div(255.).sub(mean_info).div(std_info);
	input_tensor = input_tensor.view({ nH, nPatchH, nW, nPatchW,  3 });
	input_tensor = input_tensor.permute({ 0, 2, 4, 1, 3 });
	input_tensor = input_tensor.contiguous().view({ -1, 3, nPatchH, nPatchW });
	input_tensor = F::interpolate(input_tensor, F::InterpolateFuncOptions().size(std::vector<int64_t>{ 160, 80}));

	auto infer_request = executable_network.CreateInferRequest();
	for (auto& item : input_info) {
		auto input_name = item.first;
		Blob::Ptr inputBlob = infer_request.GetBlob(input_name);
		memcpy(inputBlob->buffer(), input_tensor.data_ptr(), inputBlob->byteSize());
	}
	const clock_t time3 = clock();

	infer_request.StartAsync();
	infer_request.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
	// infer_request.Infer(); 
	const clock_t time4 = clock();
	torch::Tensor output_tensor;
	for (auto& item : output_info) {
		auto output_name = item.first;
		Blob::Ptr outputBlob = infer_request.GetBlob(output_name);
		output_tensor = torch::from_blob(outputBlob->buffer(), { nH*nW, 2, 80, 40 }, torch::kFloat32);
	}

	
	output_tensor = torch::argmax(output_tensor, 1);
	output_tensor = output_tensor.view({ nH, nW, 80, 40 });
	output_tensor = output_tensor.permute({ 0, 2, 1, 3 });
	output_tensor = output_tensor.contiguous().view({ (nH * 80), (nW * 40) });

	output_tensor = (255 * output_tensor).toType(torch::kByte);
	cv::Mat pred_label_mat(cv::Size(40 * nW, 80 * nH), CV_8U, output_tensor.data_ptr());
	resize(pred_label_mat, pred_label_mat, Size(src_w, src_h));
	const clock_t time5 = clock();

	cout << "Time Load Model:" << (time2 - time1) / 1000. << endl;
	cout << "Time Pre Processing:" << (time3 - time2) / 1000. << endl;
	cout << "Time Infer:" << (time4 - time3) / 1000. << endl;
	cout << "Time Post Processing:" << (time5 - time4) / 1000. << endl;
	cout << "Time All:" << (time5 - time2) / 1000. << endl;
	cv::imwrite("show.png", pred_label_mat);
	system("Pause");
	return 0;
}