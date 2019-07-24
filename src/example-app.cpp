#include <torch/script.h> // One-stop header.
#include <torch/serialize.h> // One-stop header.
// #include <torch/Functions.h> // One-stop header.

#include <iostream>
#include <memory>
#include <algorithm>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <superpoint.hpp>
// #include <vo_features.h>

#define MAX_P_NUM  200
#define NMS_Threshold 4


int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
  module->to(at::kCUDA);
  assert(module != nullptr);
  int H = 480;
  int W = 640;
  float* tmpfloat = new float [1*1*H*W];
  
  std::ifstream inpfile("/home/tsui/yujc/testcpptorch/pytorchtest/inp.qwe", std::ios::binary);
  inpfile.read((char*)tmpfloat, 1*1*H*W*sizeof(float));
  inpfile.close();
  std::vector<cv::Point> keypoints;
  cv::Mat descriptors;
  for (int i = 0; i < 1;i++){
    run_superpoint(module, tmpfloat, keypoints, descriptors, H, W);
    std::cout << i * 100 << std::endl;
  }
  // debug info ==================================================================================================================================
  std::cout << std::endl;
  for (int j = 0; j < descriptors.cols; j++ ){
    std::cout <<  descriptors.at<float>(0 , j) << " ";
    if (j % 4 == 3){
      std::cout << std::endl;
    }
  }
  cv::Mat inputimg(H, W, CV_32FC1, tmpfloat);
  for(int i = 0; i < keypoints.size(); i++)
  { 
    cv::Point p(keypoints[i].y, keypoints[i].x);
    cv::circle(inputimg, p , 1, (0, 255, 0), -1);
  }
  cv::imshow("src", inputimg);
  cv::waitKey();
  return 0;
}
