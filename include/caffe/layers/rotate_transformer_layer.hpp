#ifndef CAFFE_ROTATE_TRANSFORMER_LAYER_HPP_
#define CAFFE_ROTATE_TRANSFORMER_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class RotateTransformerLayer : public Layer<Dtype> {
  
public:
  explicit RotateTransformerLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RotateTransformer"; }
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

  int bottom_height_;
  int bottom_width_;
  int top_height_;
  int top_width_;

  bool first_bottom_diff_;

  int inner_num_;
  int outer_num_;

  Dtype theta_threshold_;
  Dtype shift_threshold_;

  // Store transform coefficient
  Blob<Dtype> rotate_coef_;
  // Store grid offset
  Blob<Dtype> offset_; 
  // Store data and diff for full six-dim theta, if we have default theta
  Blob<Dtype> rotate_tmp_;
  // Store all one
  Blob<Dtype> all_one_;
  // standard output coordinate system, [0, 1) by [0, 1)
  Blob<Dtype> source_grid_;
  // corresponding coordinate on input image after projection for each output pixel
  Blob<Dtype> target_grid_;
};

} // namespace caffe

#endif // CAFFE_SPATIAL_TRANSFORMER_LAYER_HPP_
