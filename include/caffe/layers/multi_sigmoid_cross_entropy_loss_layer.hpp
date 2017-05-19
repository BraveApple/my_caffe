#ifndef CAFFE_MULTI_SIGMOID_CROSS_ENTROPY_LOSS_HPP_
#define CAFFE_MULTI_SIGMOID_CROSS_ENTROPY_LOSS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
class MultiSigmoidCrossEntropyLossLayer : public LossLayer<Dtype> {
public:
  explicit MultiSigmoidCrossEntropyLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "MultiSigmoidCrossEntropyLoss"; }
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

  int task_num_;
  int outer_num_;
  int inner_num_;
  int axis_;
  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_ptr_;
  shared_ptr<Blob<Dtype> > sigmoid_output_;
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  vector<Blob<Dtype>*> sigmoid_top_vec_;
};

}

#endif
