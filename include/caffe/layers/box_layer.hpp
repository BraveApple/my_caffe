#ifndef CAFFE_BOX_LAYER_HPP_
#define CAFFE_BOX_LAYER_HPP_

#include <vector>
#include <map>
#include <string>

#include "caffe/layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"
#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
class BoxLayer : public Layer<Dtype> {
public:
  explicit BoxLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Box"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

protected:
  void Compute_Box_Grid();
  void Box_Gen_Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  void Box_Merge_Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  void Box_Gen_Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<Blob<Dtype>*>& bottom);
  void Box_Merge_Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<Blob<Dtype>*>& bottom);

  void Box_Gen_Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  void Box_Merge_Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  void Box_Gen_Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<Blob<Dtype>*>& bottom);
  void Box_Merge_Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int box_height_;
  int box_width_;
  int box_num_;
  int height_;
  int width_;

  vector<int> grid_height_;
  vector<int> grid_width_;
  vector<int> in_box_shape_;
  vector<int> out_box_shape_;

  vector<Blob<Dtype>*> bottom_boxes_;
  vector<Blob<Dtype>*> top_boxes_;

  vector<Blob<Dtype>*> conv_output_;
  vector<std::map<std::string, shared_ptr<Layer<Dtype> > > > pip_layers_;
  vector<Blob<Dtype>*> tmp_bottom_;
  vector<Blob<Dtype>*> tmp_conv_output_;
  vector<Blob<Dtype>*> tmp_top_;
};

}

#endif
