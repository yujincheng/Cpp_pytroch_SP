#include <torch/script.h> // One-stop header.
#include <torch/serialize.h> // One-stop header.
// #include <torch/Functions.h> // One-stop header.

#include <iostream>
#include <memory>
#include <algorithm>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <superpoint.h>
// #include <vo_features.h>

#define MAX_P_NUM  200
#define NMS_Threshold 4


int main(int argc, const char* argv[]) {
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("../data/modelSP_fuse_less.pt");
  module->to(at::kCUDA);
  assert(module != nullptr);
  int H = 480;
  int W = 640;
  float* tmpfloat = new float [1*1*H*W];
  
  std::ifstream inpfile("../data/inp.qwe", std::ios::binary);
  inpfile.read((char*)tmpfloat, 1*1*H*W*sizeof(float));
  inpfile.close();
  std::vector<cv::Point> keypoints;
  cv::Mat descriptors;
  auto t1=std::chrono::steady_clock::now();
  for (int i = 0; i < 1;i++){
    run_superpoint(module, tmpfloat, keypoints, descriptors, H, W);
    std::cout << i * 100 << std::endl;
  }
  auto t2=std::chrono::steady_clock::now();
  double dr_ms=std::chrono::duration<double,std::milli>(t2-t1).count();
  std::cout << "SP all : " << dr_ms << "ms" << std::endl;
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
