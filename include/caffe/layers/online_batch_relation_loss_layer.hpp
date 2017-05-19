#ifndef CAFFE_ONLINE_BATCH_RELATION_LAYER_HPP_
#define CAFFE_ONLINE_BATCH_RELATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
class OnlineBatchRelationLossLayer : public LossLayer<Dtype> {
public:
  explicit OnlineBatchRelationLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "OnlineBatchRelationLoss"; }
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

  int label_num_;
  int batch_num_;

  Dtype moving_average_fraction_;
  Dtype negative_threshold_;
  Dtype positive_threshold_;

  shared_ptr<Blob<Dtype> > batch_relation_matrix_ptr_;
  shared_ptr<Blob<Dtype> > global_relation_matrix_ptr_;

  shared_ptr<SigmoidLayer<Dtype> > sigmoid_layer_ptr_;
  vector<Blob<Dtype>*> sigmoid_bottom_vec_;
  vector<Blob<Dtype>*> sigmoid_top_vec_;
  shared_ptr<Blob<Dtype> > sigmoid_output_ptr_;
};

}

#endif
