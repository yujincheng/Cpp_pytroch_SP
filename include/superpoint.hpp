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

#define MAX_P_NUM  200
#define NMS_Threshold 4



int run_superpoint(std::shared_ptr<torch::jit::script::Module> module_cuda, float* grey, std::vector<cv::Point>& samp_pts, cv::Mat& descout, int H=480, int W=640);
int run_superpoint(std::shared_ptr<torch::jit::script::Module> module_cuda, float* grey, std::vector<cv::Point>& samp_pts, cv::Mat& descout, int H, int W){
  auto inputcuda = torch::from_blob(grey, {1,1,H,W},at::ScalarType::Float).to(at::kCUDA);
  torch::Tensor output = module_cuda->forward({inputcuda}).toTensor().to(at::kCPU);
  torch::Tensor output_s = output.squeeze();
  torch::Tensor heatmap = output_s.slice(0,0,1);  // .index_select(0,lsize1);
  heatmap = heatmap*255;
  torch::Tensor desc = output_s.slice(0,1,257); // .index_select(0,lsize2);
  int Hc = int(H / 8);
  int Wc = int(W / 8);
  auto fixmap= heatmap.round().to(torch::kUInt8);
  uint8_t* fixptr = (uint8_t* ) fixmap.data_ptr();
  uint8_t tmpmax = 0;
  std::vector<float> pts;
  pts.clear();
  uint8_t tmp_semi = 0;
  for(int i=0; i<H; i++) {
    for(int j=0; j<W; j++) {
        if( fixptr[i*W+j] > 0) {
            tmp_semi = fixptr[i*W+j];
            for(int kh=std::max(0,i-NMS_Threshold); kh<std::min(H,i+NMS_Threshold+1); kh++)
                for(int kw=std::max(0,j-NMS_Threshold); kw<std::min(W,j+NMS_Threshold+1); kw++)
                    if(i!=kh||j!=kw) {
                        if(tmp_semi >= fixptr[kh*W+kw])
                            fixptr[kh*W+kw] = 0;
                        else
                            fixptr[i*W+j] = 0;
                    }
            if(fixptr[i*W+j] != 0){
                torch::Tensor tmptensor = torch::ones({3});
                // std::cout << "i,j: " << i << " " << j << std::endl;
                pts.push_back(i);
                pts.push_back(j);
                pts.push_back(fixptr[i*W+j]);
                // pts.push_back( tmptensor );
            }
        }
    }
  }
  
  int afternms = pts.size();
  torch::Tensor pts_tensor = torch::from_blob(pts.data(),{afternms/3,3});
  auto idx = 2 * torch::ones( pts_tensor.size(0), torch::kLong);
  auto rows = torch::arange(0, pts_tensor.size(0), torch::kLong);

  auto inds2 = torch::argsort(pts_tensor.index({rows,idx}),-1,true );
  int outpoint_num = std::min(int(pts_tensor.size(0) ),MAX_P_NUM);
  inds2 = inds2.index({torch::arange(0, outpoint_num, torch::kLong)});

  samp_pts.clear();
  samp_pts.resize(outpoint_num);
  // std::cout << " num :" << int (samp_pts.size() ) << std::endl;
  // std::cout << " D :" << desc.size(0) << std::endl;
  descout.create( int (samp_pts.size() ), desc.size(0), CV_32FC1);
  // std::cout << " col :" << descout.cols << std::endl;
  // std::cout << " rows :" << descout.rows << std::endl;
  // samp_pts[0] = new float[outpoint_num];
  // samp_pts[1] = new float[outpoint_num];
  for (int i = 0; i < outpoint_num; i++){
    samp_pts[i].x = pts_tensor[ inds2[i] ][0].item<float>() ;
    samp_pts[i].y = pts_tensor[ inds2[i] ][1].item<float>() ;
    float* pData = descout.ptr<float>(i);
    for(int j = 0; j < descout.cols; j++){
        pData[j] = desc[j][int(samp_pts[i].x)][int(samp_pts[i].y)].item<float>();
    }
  }


  return 0;

}
