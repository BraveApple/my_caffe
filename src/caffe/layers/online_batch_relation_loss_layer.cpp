#include <algorithm>
#include <vector>

#include "caffe/layers/online_batch_relation_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OnlineBatchRelationLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  const OnlineBatchRelationLossParameter& online_batch_relation_loss_param = \
    this->layer_param_.online_batch_relation_loss_param();

  CHECK(online_batch_relation_loss_param.has_label_num())
    << "You miss parameter label_num";

  this->label_num_ = online_batch_relation_loss_param.label_num();
  CHECK_EQ(this->label_num_, bottom[0]->count(1)) 
    << "The number of labels dose not match";

  this->batch_num_ = bottom[0]->num();

  this->moving_average_fraction_ = online_batch_relation_loss_param.moving_average_fraction();
  CHECK_GE(this->moving_average_fraction_, 0) 
    << "moving_average_fraction must be greater or equal to 0";
  CHECK_LE(this->moving_average_fraction_, 1) 
    << "moving_average_fraction must be less or equal to 1";

  this->negative_threshold_ = online_batch_relation_loss_param.negative_threshold();
  CHECK_GE(this->negative_threshold_, 0) 
    << "negative_threshold must be greater or equal to 0";
  CHECK_LE(this->negative_threshold_, 1) 
    << "negative_threshold must be less or equal to 1";

  this->positive_threshold_ = online_batch_relation_loss_param.positive_threshold();
  CHECK_GE(this->positive_threshold_, 0) 
    << "positive_threshold must be greater or equal to 0";
  CHECK_LE(this->positive_threshold_, 1) 
    << "positive_threshold must be less or equal to 1";

  CHECK_LE(this->negative_threshold_, this->positive_threshold_)
    << "negative_threshold must be less or equal to positive_threshold";

  this->batch_relation_matrix_ptr_.reset(new Blob<Dtype>());
  this->global_relation_matrix_ptr_.reset(new Blob<Dtype>());

  this->blobs_.resize(2);
  CHECK_EQ(this->layer_param_.param_size(), 0)
    << "The number of ParamSpec parameters must be equal to 0";
  for (int i = 0; i < this->blobs_.size(); i++) {
    this->blobs_[i].reset(new Blob<Dtype>());
    ParamSpec* fixed_param_spec = this->layer_param_.add_param();
    fixed_param_spec->set_lr_mult(0.0);
  }

  this->sigmoid_layer_ptr_.reset(new SigmoidLayer<Dtype>(this->layer_param_));
  this->sigmoid_output_ptr_.reset(new Blob<Dtype>());
  this->sigmoid_bottom_vec_.clear();
  this->sigmoid_bottom_vec_.push_back(bottom[0]);
  this->sigmoid_top_vec_.clear();
  this->sigmoid_top_vec_.push_back(this->sigmoid_output_ptr_.get());
  this->sigmoid_layer_ptr_->SetUp(this->sigmoid_bottom_vec_, this->sigmoid_top_vec_);
}

template <typename Dtype>
void OnlineBatchRelationLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::Reshape(bottom, top);

  this->sigmoid_layer_ptr_->Reshape(this->sigmoid_bottom_vec_, this->sigmoid_top_vec_);

  // Reshape batch_relation_matrix_ptr_
  vector<int> batch_relation_matrix_shape(2);
  batch_relation_matrix_shape[0] = this->label_num_;
  batch_relation_matrix_shape[1] = this->label_num_;
  this->batch_relation_matrix_ptr_->Reshape(batch_relation_matrix_shape);

  // Reshape global_relation_matrix_ptr_
  vector<int> global_relation_matrix_shape(2);
  global_relation_matrix_shape[0] = this->label_num_;
  global_relation_matrix_shape[1] = this->label_num_;
  this->global_relation_matrix_ptr_->Reshape(global_relation_matrix_shape);

  // Reshape sum of batch relation matrix
  this->blobs_[0]->Reshape(batch_relation_matrix_shape);
  caffe_set<Dtype>(this->blobs_[0]->count(), Dtype(0), this->blobs_[0]->mutable_cpu_data());

  // Reshape sum of moving_average_fraction
  vector<int> moving_average_fraction_shape(0);
  this->blobs_[1]->Reshape(moving_average_fraction_shape);
  caffe_set<Dtype>(this->blobs_[1]->count(), Dtype(0), this->blobs_[1]->mutable_cpu_data());
}

template <typename Dtype>
void OnlineBatchRelationLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
   NOT_IMPLEMENTED;
}

template <typename Dtype>
void OnlineBatchRelationLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
   NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(OnlineBatchRelationLossLayer);
#endif

INSTANTIATE_CLASS(OnlineBatchRelationLossLayer);
REGISTER_LAYER_CLASS(OnlineBatchRelationLoss);

}  // namespace caffe
