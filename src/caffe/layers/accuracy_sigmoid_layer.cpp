#include <functional>
#include <utility>
#include <vector>
#include <string>

#include "caffe/layers/accuracy_sigmoid_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracySigmoidLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const AccuracySigmoidParameter& accuracy_sigmoid_param = this->layer_param_.accuracy_sigmoid_param();
  this->task_num_ = accuracy_sigmoid_param.task_num();
  this->label_axis_ = bottom[0]->CanonicalAxisIndex(accuracy_sigmoid_param.axis());
  CHECK_GE(bottom[0]->shape(this->label_axis_), this->task_num_) 
    << "The dimension of bottom[0] predicted labels must be greater or equal to " << this->task_num_;
  CHECK_GE(bottom[1]->shape(this->label_axis_), this->task_num_)
    << "The dimension of bottom[1] ground truth labels must be greater or equal to " << this->task_num_;

  this->compute_average_accuracy_ = accuracy_sigmoid_param.compute_average_accuracy();
  this->top_num_ = this->compute_average_accuracy_ ? this->task_num_ + 1 : this->task_num_; 
  this->outer_num_ = bottom[0]->count(0, this->label_axis_);
  this->inner_num_ = bottom[0]->count(this->label_axis_ + 1);
}

template <typename Dtype>
void AccuracySigmoidLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  // bottom[0]: predicted label (N, C, H, W)
  // bottom[1]: ground truth (N, 1, H, W)
  CHECK_EQ(top.size(), this->top_num_) 
    << "The number of top blobs  must be equal to " << this->top_num_;
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  for(int i = 0; i < this->top_num_; i++) {
    top[i]->Reshape(top_shape);
  }
}

template <typename Dtype>
void AccuracySigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 
  const Dtype* bottom_pred = bottom[0]->cpu_data();
  const Dtype* bottom_truth = bottom[1]->cpu_data();
  for(int i = 0; i < this->top_num_; i++) {
    caffe_set(top[i]->count(), Dtype(0), top[i]->mutable_cpu_data());
  }

  const int pred_dim = bottom[0]->count() / this->outer_num_;
  const int truth_dim = bottom[1]->count() / this->outer_num_;
  for (int i = 0; i < outer_num_; i++) {
    for (int j = 0; j < inner_num_; j++) {
      for (int k = 0; k < this->task_num_; k++) {
        const int pred_id = i * pred_dim + k * this->inner_num_ + j;
        const int pred_label = (bottom_pred[pred_id] >= 0) ? 1 : 0;

        const int truth_id = i * truth_dim + k * this->inner_num_ + j;
        const int truth_label = static_cast<int>(bottom_truth[truth_id]);
        if(pred_label == truth_label)
          top[k]->mutable_cpu_data()[0]++;
      }
    } 
  }

  for (int i = 0; i < this->task_num_; i++) {
    top[i]->mutable_cpu_data()[0] /= static_cast<Dtype>(this->outer_num_ * this->inner_num_);
  }

  if (this->compute_average_accuracy_) {
    for (int i = 0; i < this->task_num_; i++) {
      top[this->top_num_ - 1]->mutable_cpu_data()[0] += top[i]->cpu_data()[0];
    }
    top[this->top_num_ - 1]->mutable_cpu_data()[0] /= static_cast<Dtype>(this->task_num_);
  }
}

INSTANTIATE_CLASS(AccuracySigmoidLayer);
REGISTER_LAYER_CLASS(AccuracySigmoid);

}  // namespace caffe
