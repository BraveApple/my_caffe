#include <vector>
#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/rotate_transformer_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RotateTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  CHECK_EQ(bottom[1]->count(1), 3) << "The number of rotate must be 3";

  const RotateTransformerParameter& rotate_transformer_param = \
    this->layer_param_.rotate_transformer_param();
  
  CHECK(rotate_transformer_param.has_first_bottom_diff())
    << "You miss first_bottom_diff parameter";
  this->first_bottom_diff_ = rotate_transformer_param.first_bottom_diff();

  this->theta_threshold_ = rotate_transformer_param.theta_threshold();
  this->shift_threshold_ = rotate_transformer_param.shift_threshold();

  this->bottom_height_ = bottom[0]->height();
  this->bottom_width_ = bottom[0]->width();

  this->top_height_ = rotate_transformer_param.top_height();
  if(this->top_height_ == -1) {
    this->top_height_ = bottom[0]->height();
  }
  this->top_width_ = rotate_transformer_param.top_width();
  if(this->top_width_ == -1) {
    this->top_width_ = bottom[0]->width();
  }

  this->outer_num_ = bottom[0]->count(0, 1);
  this->inner_num_ = this->top_height_ * this->top_width_;
 
  this->offset_.Reshape(1, 1, 1, 8);
  Dtype* offset_ptr = this->offset_.mutable_cpu_data();
  offset_ptr[0] = 0; offset_ptr[1] = 0; // (0, 0)
  offset_ptr[2] = 0; offset_ptr[3] = 1; // (0, 1)
  offset_ptr[4] = 1; offset_ptr[5] = 0; // (1, 0)
  offset_ptr[6] = 1; offset_ptr[7] = 1; // (1, 1)
}

template <typename Dtype>
void RotateTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
 
  // Reshape top
  vector<int> top_shape = bottom[0]->shape();
  top_shape[2] = this->top_height_;
  top_shape[3] = this->top_width_;
  top[0]->Reshape(top_shape);

  // Reshape rotate_coef_
  // (N, 4) (cos_theta, sin_theta, shift_y_norm, shift_x_norm)
  vector<int> rotate_coef_shape(2);
  rotate_coef_shape[0] = this->outer_num_;
  rotate_coef_shape[1] = 4;
  this->rotate_coef_.Reshape(rotate_coef_shape);

  // Reshape target_grid_
  // (top_H * top_W, 2)
  vector<int> target_shape(2);
  target_shape[0] = this->inner_num_;
  target_shape[1] = 2;
  this->target_grid_.Reshape(target_shape);

  // Initialize target_grid_
  Dtype* target_grid_data = this->target_grid_.mutable_cpu_data();
  for(int i = 0; i < this->inner_num_; i++) {
    // Normalize the height of target_grid
    target_grid_data[3 * i] = Dtype(i / this->top_width_) / this->top_height_ * 2 - 1;
    // Normalize the width of target_grid
    target_grid_data[3 * i + 1] = Dtype(i % this->top_width_) / this->top_width_ * 2 - 1;
  }

  // Reshape source_grid_
  // (N, top_H * top_W, 2)
  vector<int> source_shape(3);
  source_shape[0] = bottom[1]->num();
  source_shape[1] = this->inner_num_;
  source_shape[2] = 2;
  this->source_grid_.Reshape(source_shape);

  // Reshape all_one_
  // (C * top_H * top_W, )
  vector<int> all_one_shape(1);
  all_one_shape[0] = top[0]->count(1);
  this->all_one_.Reshape(all_one_shape);
  caffe_set(this->all_one_.count(), Dtype(1), this->all_one_.mutable_cpu_data()); 

  // Reshape rotate_tmp_
  // (N, 3, C * top_H * top_W)
  vector<int> rotate_tmp_shape(3);
  rotate_tmp_shape[0] = this->outer_num_;
  rotate_tmp_shape[1] = 3;
  rotate_tmp_shape[2] = top[0]->count(1);
  this->rotate_tmp_.Reshape(rotate_tmp_shape);
}

template <typename Dtype>
void RotateTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void RotateTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RotateTransformerLayer);
#endif

INSTANTIATE_CLASS(RotateTransformerLayer);
REGISTER_LAYER_CLASS(RotateTransformer);

} // namespace caffe
