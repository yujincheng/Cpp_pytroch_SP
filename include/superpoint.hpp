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
    std::chrono::steady_clock::time_point timepoint[10];
        timepoint[0] =std::chrono::steady_clock::now();
  auto inputcuda = torch::from_blob(grey, {1,1,H,W},at::ScalarType::Float).to(at::kCUDA);
        timepoint[1] =std::chrono::steady_clock::now();
  torch::Tensor output = module_cuda->forward({inputcuda}).toTensor();
        timepoint[2] =std::chrono::steady_clock::now();
//   torch::Tensor output =      output12.to(at::kCPU);
        timepoint[3] =std::chrono::steady_clock::now();

  int Hc = int(H / 8);
  int Wc = int(W / 8);
  torch::Tensor output_s = output.squeeze();
//   output_s = output_s.to(at::kCPU);
  torch::Tensor heatmap = output_s.slice(0,0,1);  // .index_select(0,lsize1);
//   heatmap = heatmap*255;
  torch::Tensor desc_reshape = output_s.slice(0,1,65); // .index_select(0,lsize2);
  torch::Tensor desc = torch::reshape(desc_reshape, {-1,Hc,Wc} );
  desc = torch::transpose(desc,0,1);
  desc = torch::transpose(desc,1,2);
  heatmap = heatmap.to(at::kCPU);
  desc = desc.to(at::kCPU);
//   auto fixmap= heatmap.round().to(torch::kUInt8);
//   uint8_t* fixptr = (uint8_t* ) fixmap.data_ptr();
  float* fixptr = (float* ) heatmap.data_ptr();
  float tmpmax = 0;
  std::vector<float> pts;
  pts.clear();
  float tmp_semi = 0;
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
        timepoint[4] =std::chrono::steady_clock::now();
  
  int afternms = pts.size();
  torch::Tensor pts_tensor = torch::from_blob(pts.data(),{afternms/3,3});
  auto idx = 2 * torch::ones( pts_tensor.size(0), torch::kLong);
  auto rows = torch::arange(0, pts_tensor.size(0), torch::kLong);

  auto inds2 = torch::argsort(pts_tensor.index({rows,idx}),-1,true );
  int outpoint_num = std::min(int(pts_tensor.size(0) ),MAX_P_NUM);
  inds2 = inds2.index({torch::arange(0, outpoint_num, torch::kLong)});
       timepoint[5] =std::chrono::steady_clock::now();

  samp_pts.clear();
  samp_pts.resize(outpoint_num);
  std::cout << " num :" << int (samp_pts.size() ) << std::endl;
  std::cout << " D :" << desc.size(2) << std::endl;
  descout.create( int (samp_pts.size() ), desc.size(2), CV_32FC1);
  std::cout << " col :" << descout.cols << std::endl;
  std::cout << " rows :" << descout.rows << std::endl;
  // samp_pts[0] = new float[outpoint_num];
  // samp_pts[1] = new float[outpoint_num];
  float* desc_ptr = (float*)desc.data_ptr();
  for (int i = 0; i < outpoint_num; i++){
    samp_pts[i].x = pts_tensor[ inds2[i] ][0].item<float>() ;
    samp_pts[i].y = pts_tensor[ inds2[i] ][1].item<float>() ;
    float* pData = descout.ptr<float>(i);
    // for(int j = 0; j < descout.cols; j++){
        // std::cout << " js  :" << j << std::endl;
        // pData[j] = desc[int(samp_pts[i].x / 8)][int(samp_pts[i].y / 8)][j].item<float>();
        int x1 = int(samp_pts[i].x / 8);
        int x2 = int(samp_pts[i].y / 8);
        memcpy(pData, desc_ptr+(x2*descout.cols + x1*Wc*descout.cols) , sizeof(float)*descout.cols );
        // pData[j] = *(desc_ptr+( j + x2*descout.cols + x1*Wc*descout.cols ) );
    // }
  }
        timepoint[6] =std::chrono::steady_clock::now();
        for (int i = 0; i < 7; i ++ ){
            std::cout << " time step: " << i << " : " << std::chrono::duration<double,std::milli>(timepoint[i+1] - timepoint[i]).count() << std::endl;
        }


  return 0;

}
