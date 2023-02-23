#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
  cv::Mat img(1080, 1920, CV_8UC3);
  cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

  cv::dnn::Net net = cv::dnn::readNet("yolov5m_based.onnx");
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);


  cv::Mat blob;
  cv::dnn::blobFromImage(img, blob, 1. / 255., cv::Size(640, 640), cv::Scalar(), false, false);
  net.setInput(blob);
  std::vector<cv::Mat> outputs;

  // Don't measure this, GPU needs to warm up.
  for (int i = 0; i < 5; ++i)
    net.forward(outputs, "output0");

  auto c1 = cv::getTickCount();
  for (int i = 0; i < 200; ++i)
    net.forward(outputs, "output0");
  auto c2 = cv::getTickCount();

  std::cout << "TOTAL TIME: " << ((c2 - c1) / cv::getTickFrequency() * 1000) << std::endl;

  return 0;
}
