#include <algorithm>
#include <vector>

#include "caffe/layers/multi_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  const MultiSigmoidCrossEntropyLossParameter& multi_sigmoid_loss_param = \
    this->layer_param_.multi_sigmoid_loss_param();
  this->task_num_ = multi_sigmoid_loss_param.task_num();
  this->axis_ = multi_sigmoid_loss_param.axis();
  CHECK_EQ(this->task_num_, bottom[0]->shape(this->axis_))
    << "the " << this->axis_ << "-th dimension of bottom[0] must be equal to " << this->task_num_;
  CHECK_EQ(this->task_num_, bottom[1]->shape(this->axis_))
    << "the " << this->axis_ << "-th dimension of bottom[1] must be equal to " << this->task_num_;

  this->outer_num_ = bottom[0]->count(0, this->axis_);
  this->inner_num_ = bottom[0]->count(this->axis_ + 1);

  this->sigmoid_layer_ptr_.reset(new SigmoidLayer<Dtype>(this->layer_param_));
  this->sigmoid_output_.reset(new Blob<Dtype>());
  this->sigmoid_bottom_vec_.clear();
  this->sigmoid_bottom_vec_.push_back(bottom[0]);
  this->sigmoid_top_vec_.clear();
  this->sigmoid_top_vec_.push_back(this->sigmoid_output_.get());
  this->sigmoid_layer_ptr_->SetUp(this->sigmoid_bottom_vec_, this->sigmoid_top_vec_); 
}

template <typename Dtype>
void MultiSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::Reshape(bottom, top);
  this->sigmoid_layer_ptr_->Reshape(this->sigmoid_bottom_vec_, this->sigmoid_top_vec_);
}

template <typename Dtype>
void MultiSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
  this->sigmoid_bottom_vec_[0] = bottom[0];
  this->sigmoid_layer_ptr_->Forward(this->sigmoid_bottom_vec_, this->sigmoid_top_vec_);
  
  const Dtype* pred_data = bottom[0]->cpu_data();
  const Dtype* truth_label = bottom[1]->cpu_data();
  Dtype loss = 0;
  for(int i = 0; i < bottom[0]->count(); i++) {
    loss -= pred_data[0] * (truth_label[i] - (pred_data[0] >= 0)) - \
      log(1 + exp(pred_data[0] - 2 * pred_data[0] * (pred_data[0] >= 0)));
  }
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->count();
}

template <typename Dtype>
void MultiSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* pred_prob = this->sigmoid_output_->cpu_data();
    const Dtype* truth_label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, pred_prob, truth_label, bottom_diff);
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / count;
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiSigmoidCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(MultiSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(MultiSigmoidCrossEntropyLoss);

}  // namespace caffe