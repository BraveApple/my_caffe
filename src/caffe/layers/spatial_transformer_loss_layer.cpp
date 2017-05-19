#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/spatial_transformer_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SpatialTransformerLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  const SpatialTransformerLossParameter& spatial_transformer_loss_param = \
    this->layer_param_.spatial_transformer_loss_param();
  
  CHECK(spatial_transformer_loss_param.has_top_height())
    << "You miss top_height parameter";
  this->top_height_ = spatial_transformer_loss_param.top_height();
  
  CHECK(spatial_transformer_loss_param.has_top_width())
    << "You miss top_width parameter";  
  this->top_width_ = spatial_transformer_loss_param.top_width();

  this->outer_num_ = bottom[0]->num();
  this->inner_num_ = bottom[0]->count(1);
  CHECK_EQ(bottom[0]->count(1), 6) << "Input theta must have dimension of 6.";
  this->theta_num_ = bottom[0]->count(1);
}

template <typename Dtype>
void SpatialTransformerLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
  
  // Reshape loss_
  // (N, top_H, top_W)
  vector<int> loss_shape(3);
  loss_shape[0] = this->outer_num_;
  loss_shape[1] = this->top_height_;
  loss_shape[2] = this->top_width_;
  this->loss_.Reshape(loss_shape);

  // Reshape theta_tmp_
  // (N, 6, top_H * top_W)
  vector<int> theta_tmp_shape(3);
  theta_tmp_shape[0] = this->outer_num_;
  theta_tmp_shape[1] = this->theta_num_;
  theta_tmp_shape[2] = this->inner_num_;
  this->theta_tmp_.Reshape(theta_tmp_shape);

  // Reshape all_one_
  // (top_H * top_W, )
  vector<int> all_one_shape(1);
  all_one_shape[0] = this->inner_num_;
  this->all_one_.Reshape(all_one_shape);
  caffe_set(this->all_one_.count(), Dtype(1), this->all_one_.mutable_cpu_data());
}

template <typename Dtype>
void SpatialTransformerLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SpatialTransformerLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SpatialTransformerLossLayer);
#endif

INSTANTIATE_CLASS(SpatialTransformerLossLayer);
REGISTER_LAYER_CLASS(SpatialTransformerLoss);

} 
