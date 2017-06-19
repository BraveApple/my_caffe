#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  const CenterLossParameter& center_loss_param = this->layer_param_.center_loss_param();
  CHECK(center_loss_param.has_num_output()) << "You miss the parameter num_output";
  this->class_num_ = center_loss_param.num_output();
  const int axis = bottom[0]->CanonicalAxisIndex(center_loss_param.axis());
  // Demensions starting from "axis" are "flattened" into a single
  // length inner_num_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension C*H*W are performed.
  this->batch_size_ = bottom[0]->count(0, axis);
  this->center_dim_ = bottom[0]->count(axis);
  CHECK_EQ(bottom[1]->count(axis), 1) 
    << "Bottom[1] must be fed with a one-dimensional vector (N, 1, 1, 1)";
  
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> center_shape(2);
    center_shape[0] = this->batch_size_;
    center_shape[1] = this->center_dim_;
    this->blobs_[0].reset(new Blob<Dtype>(center_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(center_loss_param.center_filler()));
    center_filler->Fill(this->blobs_[0].get());
  } // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  // The top shape will be the bottom shape with the flattended axes droppped,
  // and repalced by a single axis with dimension class_num_
  LossLayer<Dtype>::Reshape(bottom, top);
  this->distance_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* true_label = bottom[1]->cpu_data();
  const Dtype* center_data = this->blobs_[0]->cpu_data();
  Dtype* distance_data = this->distance_.mutable_cpu_data();

  // the i-th distance_data
  for (int i = 0; i < this->batch_size_; i++) {
    const int label_value = static_cast<int>(true_label[i]);
    // D[i, :] = X[i, :] - C[y(i), :]
    const Dtype* bottom_data_tmp = bottom_data + i * this->center_dim_;
    const Dtype* center_data_tmp = center_data + label_value * this->center_dim_;
    Dtype* distance_data_tmp = distance_data + i * this->center_dim_;

    caffe_sub<Dtype>(this->center_dim_, bottom_data_tmp, center_data_tmp, distance_data_tmp); 
  }
  Dtype dot = caffe_cpu_dot<Dtype>(this->batch_size_ * this->center_dim_, distance_data, distance_data);
  Dtype loss = dot / (this->batch_size_ * 2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* true_label = bottom[1]->cpu_data();
  const Dtype* distance_data = this->distance_.cpu_data();

  Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  // Gradient with respect to centers
  if (this->param_propagate_down_[0]) { 
    // \sum_{y_i == j}
    caffe_set<Dtype>(this->class_num_ * this->center_dim_, 0, center_diff);
    for (int class_id = 0; class_id < this->class_num_; class_id++) {
      Dtype* center_diff_tmp = center_diff + class_id * this->center_dim_;
      int count = 0;
      for (int batch_id = 0; batch_id < this->batch_size_; batch_id++) {
        const Dtype* distance_data_tmp = distance_data + batch_id * this->center_dim_;
        const int label_value = static_cast<int>(true_label[batch_id]);
        if (label_value == class_id) {
          count++;
          caffe_sub<Dtype>(this->center_dim_, center_diff_tmp, distance_data_tmp, center_diff_tmp);
        }
      }
      const Dtype scale = 1./ Dtype(1. + count);
      caffe_scal<Dtype>(this->center_dim_, scale, center_diff_tmp);
    }
  }

  // Gradient with respect to bottom data
  if (propagate_down[0]) {
    const int length = this->batch_size_ * this->center_dim_;
    caffe_copy<Dtype>(length, distance_data, bottom_diff);
    caffe_scal<Dtype>(length, top[0]->cpu_diff()[0] / this->batch_size_, bottom_diff);
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
      << "Layer cannot backpropagate to ground truth label inputs";
  }
} 

#ifdef CPU_ONLY
STUB_GPU(CenterLossLayer);
#endif

INSTANTIATE_CLASS(CenterLossLayer);
REGISTER_LAYER_CLASS(CenterLoss);

} // namesapce caffe
