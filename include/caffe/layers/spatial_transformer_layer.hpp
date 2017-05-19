#ifndef CAFFE_SPATIAL_TRANSFORMER_LAYER_HPP_
#define CAFFE_SPATIAL_TRANSFORMER_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SpatialTransformerLayer : public Layer<Dtype> {
  
public:
  explicit SpatialTransformerLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SpatialTransformer"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
  inline Dtype abs(Dtype x) { return (x >= 0) ? x : -x; }
  inline Dtype max(Dtype x, Dtype y) { return (x >= y) ? x : y; }
  Dtype Transform_forward_cpu(const Dtype* bottom_data,
    const Dtype source_x_norm, const Dtype source_y_norm);
  void Transform_backward_cpu(const Dtype top_diff, const Dtype* bottom_data,
    const Dtype source_y_norm, const Dtype source_x_norm, Dtype* bottom_diff, 
    Dtype& source_y_norm_diff, Dtype& source_x_norm_diff);
  
  string transformer_type_;
  string sampler_type_;
  
  int bottom_height_;
  int bottom_width_;
  int top_height_;
  int top_width_;

  bool first_bottom_diff_;
  int theta_num_;

  int inner_num_;
  int outer_num_;

  // Store grid offset
  Blob<Dtype> offset_; 
  // Store data and diff for full six-dim theta, if we have default theta
  Blob<Dtype> full_theta_;
  // Store the intermediate gradient of theta
  Blob<Dtype> theta_tmp_;
  // Store all one
  Blob<Dtype> all_one_; 
  // standard output coordinate system, [0, 1) by [0, 1)
  Blob<Dtype> input_grid_;
  // corresponding coordinate on input image after projection for each output pixel
  Blob<Dtype> output_grid_;
};

} // namespace caffe

#endif // CAFFE_SPATIAL_TRANSFORMER_LAYER_HPP_
