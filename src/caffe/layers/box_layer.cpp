#include <vector>
#include <iostream>

#include "caffe/layers/box_layer.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void BoxLayer<Dtype>::Compute_Box_Grid() {
  
  int n_height = this->height_ / this->box_height_;
  if(this->height_ % this->box_height_ != 0) {
    n_height += 1;
  }
  int n_width = this->width_ / this->box_width_;
  if(this->width_ % this->box_width_ != 0) {
    n_width += 1;
  }

  this->grid_height_.resize(n_height);
  this->grid_width_.resize(n_width);
  for(int i = 0; i < n_height; i++) {
    this->grid_height_[i] = i * this->box_height_;
  }
  for(int i = 0; i < n_width; i++) {
    this->grid_width_[i] = i * this->box_width_; 
  }

  this->box_num_ = n_height * n_width;
}

template <typename Dtype>
void BoxLayer<Dtype>::Box_Gen_Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom.size(), 1) << "The number of the bottom blobs must be one" \
    << " bottom_num VS. 1 --> " << bottom.size() << " VS. 1"; 
  CHECK_EQ(top.size(), this->box_num_) << "The number of the top blobs must be equal to boxes" \
    << "top_num VS. box_num --> " << top.size() << " VS. " << this->box_num_;

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int n_height = this->grid_height_.size();
  const int n_width = this->grid_width_.size();
  const int box_size = this->box_height_ * this->box_width_;

  for(int i = 0; i < n_height; i++) {
    const int height_start = this->grid_height_[i];
    for(int j = 0; j < n_width; j++) {
      const int width_start = this->grid_width_[j];
      Dtype* top_data = top[j + i * n_width]->mutable_cpu_data();
      for(int total_element_id = 0; total_element_id < top[0]->count(); total_element_id++) {
        const int box_num = total_element_id / box_size;
        const int element_id = total_element_id % box_size;
        const int height_offset = element_id / this->box_width_;
        const int width_offset = element_id % this->box_width_;
        const int element_height = height_start + height_offset;
        const int element_width = width_start + width_offset;
        const int bottom_data_id = box_num * this->height_ * this->width_ + \
          element_height * this->width_ + element_width;
        top_data[total_element_id] = (bottom_data_id < bottom[0]->count()) ? bottom_data[bottom_data_id] : 0;
      }
    }
  }
}

template <typename Dtype>
void BoxLayer<Dtype>::Box_Merge_Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom.size(), this->box_num_) << "The number of the bottom blobs must be equal to boxes" \
    << "bottom_num VS. box_num --> " << bottom.size() << " VS. " << this->box_num_;
  CHECK_EQ(top.size(), 1) << "The number of the top blobs must be one" \
    << "top_num VS. 1 --> " << top.size() << " VS. 1";

  Dtype* top_data = top[0]->mutable_cpu_data();
  const int n_height = this->grid_height_.size();
  const int n_width = this->grid_width_.size();
  const int box_size = this->box_height_ * this->box_width_;

  for(int i = 0; i < n_height; i++) {
    const int height_start = this->grid_height_[i];
    for(int j = 0; j < n_width; j++) {
      const int width_start = this->grid_width_[j];
      const Dtype* bottom_data = bottom[j + i * n_width]->cpu_data();
      for(int total_element_id = 0; total_element_id < bottom[0]->count(); total_element_id++) {
        const int box_num = total_element_id / box_size;
        const int element_id = total_element_id % box_size;
        const int height_offset = element_id / this->box_width_;
        const int width_offset = element_id % this->box_width_;
        const int element_height = height_start + height_offset;
        const int element_width = width_start + width_offset;
        const int top_data_id = box_num * this->height_ * this->width_ + \
          element_height * this->width_ + element_width;
        if(top_data_id < top[0]->count()) {
          top_data[top_data_id] = bottom_data[total_element_id];
        }
      }
    }
  }
}


template <typename Dtype>
void BoxLayer<Dtype>::Box_Gen_Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<Blob<Dtype>*>& bottom) {

  CHECK_EQ(bottom.size(), 1) << "The number of the bottom blobs must be one" \
    << " bottom_num VS. 1 --> " << bottom.size() << " VS. 1"; 
  CHECK_EQ(top.size(), this->box_num_) << "The number of the top blobs must be equal to boxes" \
    << "top_num VS. box_num --> " << top.size() << " VS. " << this->box_num_;

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int n_height = this->grid_height_.size();
  const int n_width = this->grid_width_.size();
  const int box_size = this->box_height_ * this->box_width_;

  for(int i = 0; i < n_height; i++) {
    const int height_start = this->grid_height_[i];
    for(int j = 0; j < n_width; j++) {
      const int width_start = this->grid_width_[j];
      const Dtype* top_diff = top[j + i * n_width]->cpu_diff();
      for(int total_element_id = 0; total_element_id < top[0]->count(); total_element_id++) {
        const int box_num = total_element_id / box_size;
        const int element_id = total_element_id % box_size;
        const int height_offset = element_id / this->box_width_;
        const int width_offset = element_id % this->box_width_;
        const int element_height = height_start + height_offset;
        const int element_width = width_start + width_offset;
        const int bottom_data_id = box_num * this->height_ * this->width_ + \
          element_height * this->width_ + element_width;
        if(bottom_data_id < bottom[0]->count()) {
          bottom_diff[bottom_data_id] = top_diff[total_element_id];
        }
      }
    }
  }
}

template <typename Dtype>
void BoxLayer<Dtype>::Box_Merge_Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<Blob<Dtype>*>& bottom) {

  CHECK_EQ(bottom.size(), this->box_num_) << "The number of the bottom blobs must be equal to boxes" \
    << "bottom_num VS. box_num --> " << bottom.size() << " VS. " << this->box_num_;
  CHECK_EQ(top.size(), 1) << "The number of the top blobs must be one" \
    << "top_num VS. 1 --> " << top.size() << " VS. 1";

  const Dtype* top_diff = top[0]->cpu_diff();
  const int n_height = this->grid_height_.size();
  const int n_width = this->grid_width_.size();
  const int box_size = this->box_height_ * this->box_width_;

  for(int i = 0; i < n_height; i++) {
    const int height_start = this->grid_height_[i];
    for(int j = 0; j < n_width; j++) {
      const int width_start = this->grid_width_[j];
      Dtype* bottom_diff = bottom[j + i * n_width]->mutable_cpu_diff();
      for(int total_element_id = 0; total_element_id < bottom[0]->count(); total_element_id++) {
        const int box_num = total_element_id / box_size;
        const int element_id = total_element_id % box_size;
        const int height_offset = element_id / this->box_width_;
        const int width_offset = element_id % this->box_width_;
        const int element_height = height_start + height_offset;
        const int element_width = width_start + width_offset;
        const int top_data_id = box_num * this->height_ * this->width_ + \
          element_height * this->width_ + element_width;
        bottom_diff[total_element_id] = (total_element_id < top[0]->count()) ? top_diff[top_data_id] : 0;
      }
    }
  }
}

template <typename Dtype>
void BoxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  const BoxParameter box_param = this->layer_param_.box_param();
  CHECK(box_param.has_box_height()) << "You miss the box_height parameter";
  this->box_height_ = box_param.box_height();
  
  CHECK(box_param.has_box_width()) << "You miss the box_width parameter";
  this->box_width_ = box_param.box_width();
  
  this->height_ = bottom[0]->height();
  CHECK_GE(this->height_, this->box_height_) << "The height of feature blob must be equal or greater than the box"
    << "(height, box_height) = " << "(" << this->height_ << "," << this->box_height_ << ")";
  
  this->width_ = bottom[0]->width();
  CHECK_GE(this->width_, this->box_width_) << "The width of feature blob must be equal or greater than the box"
    << "(width, box_width) = " << "(" << this->width_ << "," << this->box_width_ << ")";

  this->Compute_Box_Grid();

  CHECK(box_param.has_convolution_param()) << "You miss the convolution_param parameter";
  LayerParameter convolution_layer_param(this->layer_param_);
  convolution_layer_param.set_type("Convolution");
  convolution_layer_param.mutable_convolution_param()->CopyFrom(box_param.convolution_param());
  CHECK_EQ(box_param.param_size(), 2) << "The number of param for lr_mult and decay_mult must be two";
  convolution_layer_param.clear_param();
  convolution_layer_param.add_param()->CopyFrom(box_param.param(0));
  convolution_layer_param.add_param()->CopyFrom(box_param.param(1));

  CHECK(box_param.has_pooling_param()) << "You miss the pooling_param parameter";
  LayerParameter pooling_layer_param(this->layer_param_);
  pooling_layer_param.set_type("Pooling");
  pooling_layer_param.mutable_pooling_param()->CopyFrom(box_param.pooling_param());

  LayerParameter relu_layer_param(this->layer_param_);
  relu_layer_param.set_type("ReLU");
  if(box_param.has_relu_param()) {
    relu_layer_param.mutable_relu_param()->CopyFrom(box_param.relu_param());
  }

  this->in_box_shape_.resize(4);
  this->in_box_shape_[0] = bottom[0]->num();
  this->in_box_shape_[1] = bottom[0]->channels();
  this->in_box_shape_[2] = this->box_height_;
  this->in_box_shape_[3] = this->box_width_;

  this->out_box_shape_.resize(4);
  this->out_box_shape_[0] = bottom[0]->num();
  this->out_box_shape_[1] = box_param.convolution_param().num_output();
  this->out_box_shape_[2] = this->box_height_;
  this->out_box_shape_[3] = this->box_width_;
  
  // cout << "box_num = " << this->box_num_ << endl;
  this->pip_layers_.resize(this->box_num_);
  for(int i = 0; i < this->box_num_; i++) {
    this->pip_layers_[i]["Convolution"] = LayerRegistry<Dtype>::CreateLayer(convolution_layer_param);
    this->pip_layers_[i]["ReLU"] = LayerRegistry<Dtype>::CreateLayer(relu_layer_param);
    this->pip_layers_[i]["Pooling"] = LayerRegistry<Dtype>::CreateLayer(pooling_layer_param);
  }
  
  // cout << "1" << endl;

  this->bottom_boxes_.resize(this->box_num_);
  this->conv_output_.resize(this->box_num_);
  this->top_boxes_.resize(this->box_num_);
  for(int i = 0; i < this->box_num_; i++) {
      this->bottom_boxes_[i] = new Blob<Dtype>();
      this->bottom_boxes_[i]->Reshape(this->in_box_shape_);

      this->conv_output_[i] = new Blob<Dtype>();
      this->conv_output_[i]->Reshape(this->out_box_shape_);

      this->top_boxes_[i] = new Blob<Dtype>();
      this->top_boxes_[i]->Reshape(this->out_box_shape_);
  }
  
  // cout << "2" << endl;

  this->tmp_bottom_.resize(1);
  this->tmp_conv_output_.resize(1);
  this->tmp_top_.resize(1);
  for(int i = 0; i < this->box_num_; i++) {
      this->tmp_bottom_[0] = this->bottom_boxes_[i];
      this->tmp_conv_output_[0] = this->conv_output_[i];
      this->tmp_top_[0] = this->top_boxes_[i];

      // set up convolution layer
      // cout << "2.1" << endl;
      this->pip_layers_[i]["Convolution"]->SetUp(this->tmp_bottom_, this->tmp_conv_output_);
      // set up relu layer
      // cout << "2.2" << endl;
      this->pip_layers_[i]["ReLU"]->SetUp(this->tmp_conv_output_, this->tmp_conv_output_);
      // set up pooling layer
      // cout << "2.3" << endl;
      this->pip_layers_[i]["Pooling"]->SetUp(this->tmp_conv_output_, this->tmp_top_);
  }
  // cout << "3" << endl;
}

template <typename Dtype>
void BoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  // cout << "4" << endl;
  // reshape pipline layers
  for(int i = 0; i < this->box_num_; i++) {

    this->tmp_bottom_[0] = this->bottom_boxes_[i];
    this->tmp_conv_output_[0] = this->conv_output_[i];
    this->tmp_top_[0] = this->top_boxes_[i];

    // reshape convolution layer
    // cout << "4.1" << endl;
    this->pip_layers_[i]["Convolution"]->Reshape(this->tmp_bottom_, this->tmp_conv_output_);
    // reshape relu layer
    // cout << "4.2" << endl;
    this->pip_layers_[i]["ReLU"]->Reshape(this->tmp_conv_output_, this->tmp_conv_output_);
    // reshape pooling layer
    // cout << "4.3" << endl;
    this->pip_layers_[i]["Pooling"]->Reshape(this->tmp_conv_output_, this->tmp_top_);
  }

  // check the shape of top boxes
  for(int i = 0; i < this->box_num_; i++) {
    const vector<int>& top_shape = this->top_boxes_[i]->shape();
    for(int j = 0; j < this->out_box_shape_.size(); j++) {
      CHECK_EQ(this->out_box_shape_[j], top_shape[j]) << "The " << j << "-th shape of out_box blobs must be equal to top blobs " \
        << "(out_box_shape[j], top_shape[j]) = " << "(" << this->out_box_shape_[j] << "," << top_shape[j] << ")";
    }
  }

  vector<int> most_top_shape(4, 0);
  most_top_shape[0] = this->out_box_shape_[0];
  most_top_shape[1] = this->out_box_shape_[1];
  most_top_shape[2] = this->height_;
  most_top_shape[3] = this->width_;
  top[0]->Reshape(most_top_shape);
}

template <typename Dtype>
void BoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  this->Box_Gen_Forward_cpu(bottom, this->bottom_boxes_);
  for(int i = 0; i < this->box_num_; i++) {
    this->tmp_bottom_[0] = this->bottom_boxes_[i];
    this->tmp_conv_output_[0] = this->conv_output_[i];
    this->tmp_top_[0] = this->top_boxes_[i];
    
    // the convolution layer forward
    this->pip_layers_[i]["Convolution"]->Forward(this->tmp_bottom_, this->tmp_conv_output_);
    // the relu layer forward
    this->pip_layers_[i]["ReLU"]->Forward(this->tmp_conv_output_, this->tmp_conv_output_);
    // the pooling layer forward
    this->pip_layers_[i]["Pooling"]->Forward(this->tmp_conv_output_, this->tmp_top_);
  }
  this->Box_Merge_Forward_cpu(this->top_boxes_, top);
}

template <typename Dtype>
void BoxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  this->Box_Merge_Backward_cpu(top, this->top_boxes_);
  for(int i = 0; i < this->box_num_; i++) {
    this->tmp_bottom_[0] = this->bottom_boxes_[i];
    this->tmp_conv_output_[0] = this->conv_output_[i];
    this->tmp_top_[0] = this->top_boxes_[i];
    
    // the pooling layer backward
    this->pip_layers_[i]["Pooling"]->Backward(this->tmp_top_, propagate_down, this->tmp_conv_output_);
    // the relu layer forward
    this->pip_layers_[i]["ReLU"]->Backward(this->tmp_conv_output_, propagate_down, this->tmp_conv_output_);
    // the convolution layer forward
    this->pip_layers_[i]["Convolution"]->Backward(this->tmp_conv_output_, propagate_down, this->tmp_bottom_);
  }
  this->Box_Gen_Backward_cpu(this->bottom_boxes_, bottom);
}

#ifdef CPU_ONLY
STUB_GPU(BoxLayer);
#endif

INSTANTIATE_CLASS(BoxLayer);
REGISTER_LAYER_CLASS(Box);

}
