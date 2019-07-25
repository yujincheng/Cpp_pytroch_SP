#ifndef __SUPERPOINT_H_
#define __SUPERPOINT_H_

#include <torch/script.h> // One-stop header.
#include <torch/serialize.h> // One-stop header.
// #include <torch/Functions.h> // One-stop header.

#include <iostream>
#include <memory>
#include <algorithm>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
// #include <vo_features.h>


int run_superpoint(std::shared_ptr<torch::jit::script::Module> module_cuda, float* grey, std::vector<cv::Point>& samp_pts, cv::Mat& descout, int H=480, int W=640);


#endif
