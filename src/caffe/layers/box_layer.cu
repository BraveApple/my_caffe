#include <vector>

#include "caffe/layers/box_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Box_Gen(const int nthreads, const bool forward, const Dtype* in_data, 
  const int bottom_count, const int box_height, const int box_width, const int height_start, 
  const int width_start, const int height, const int width, Dtype* out_data) {
  
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int box_count = box_height * box_width;
    const int num_box = index / box_count;
    const int element_id = index % box_count;
    const int height_offset = element_id / box_width;
    const int width_offset = element_id % box_width;
    const int element_height = height_start + height_offset;
    const int element_width = width_start + width_offset;
    const int bottom_data_id = num_box * height * width + element_height * width + element_width;
    if(forward) {
      out_data[index] = (bottom_data_id < bottom_count) ? in_data[bottom_data_id] : 0;
    } else {
      if(bottom_data_id < bottom_count)
        out_data[bottom_data_id] = in_data[index];
    }
  }
}

template <typename Dtype>
__global__ void Box_Merge(const int nthreads, const bool forward, const Dtype* in_data, 
  const int top_count, const int box_height, const int box_width, const int height_start, 
  const int width_start, const int height, const int width, Dtype* out_data) {
  
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int box_count = box_height * box_width;
    const int num_box = index / box_count;
    const int element_id = index % box_count;
    const int height_offset = element_id / box_width;
    const int width_offset = element_id % box_width;
    const int element_height = height_start + height_offset;
    const int element_width = width_start + width_offset;
    const int top_data_id = num_box * height * width + element_height * width + element_width;
    if(forward) {
      if(top_data_id < top_count)
        out_data[top_data_id] = in_data[index];
    } else {
      out_data[index] = (top_data_id < top_count) ? in_data[top_data_id] : 0;
    }
  }
}

template <typename Dtype>
void BoxLayer<Dtype>::Box_Gen_Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom.size(), 1) << "The number of the bottom blobs must be one" \
    << " bottom_num VS. 1 --> " << bottom.size() << " VS. 1"; 
  CHECK_EQ(top.size(), this->box_num_) << "The number of the top blobs must be equal to boxes" \
    << "top_num VS. box_num --> " << top.size() << " VS. " << this->box_num_;

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int num_per_box = bottom[0]->count(0, 2);
  const int bottom_count = bottom[0]->count();
  const int nthreads = num_per_box * this->box_height_ * this->box_width_;
  const int n_height = this->grid_height_.size();
  const int n_width = this->grid_width_.size();

  for(int i = 0; i < n_height; i++) {
    const int height_start = this->grid_height_[i];
    for(int j = 0; j < n_width; j++) {
      const int width_start = this->grid_width_[j];
      Dtype* top_data = top[j + i * n_width]->mutable_gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      Box_Gen<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, true, bottom_data, bottom_count, this->box_height_, this->box_width_,
        height_start, width_start, this->height_, this->width_, top_data);
    }
  }
}

template <typename Dtype>
void BoxLayer<Dtype>::Box_Merge_Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom.size(), this->box_num_) << "The number of the bottom blobs must be equal to boxes" \
    << "bottom_num VS. box_num --> " << bottom.size() << " VS. " << this->box_num_;
  CHECK_EQ(top.size(), 1) << "The number of the top blobs must be one" \
    << "top_num VS. 1 --> " << top.size() << " VS. 1";

  Dtype* top_data = top[0]->mutable_gpu_data();
  const int num_per_box = top[0]->count(0, 2);
  const int top_count = top[0]->count();
  const int nthreads = num_per_box * this->box_height_ * this->box_width_;
  const int n_height = this->grid_height_.size();
  const int n_width = this->grid_width_.size();

  for(int i = 0; i < n_height; i++) {
    const int height_start = this->grid_height_[i];
    for(int j = 0; j < n_width; j++) {
      const int width_start = this->grid_width_[j];
      const Dtype* bottom_data = bottom[j + i * n_width]->gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      Box_Merge<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, true, bottom_data, top_count, this->box_height_, this->box_width_,
        height_start, width_start, this->height_, this->width_, top_data);
    }
  }
}

template <typename Dtype>
void BoxLayer<Dtype>::Box_Gen_Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<Blob<Dtype>*>& bottom) {

  CHECK_EQ(bottom.size(), 1) << "The number of the bottom blobs must be one" \
    << " bottom_num VS. 1 --> " << bottom.size() << " VS. 1"; 
  CHECK_EQ(top.size(), this->box_num_) << "The number of the top blobs must be equal to boxes" \
    << "top_num VS. box_num --> " << top.size() << " VS. " << this->box_num_;

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int num_per_box = bottom[0]->count(0, 2);
  const int bottom_count = bottom[0]->count();
  const int nthreads = num_per_box * this->box_height_ * this->box_width_;
  const int n_height = this->grid_height_.size();
  const int n_width = this->grid_width_.size();

  for(int i = 0; i < n_height; i++) {
    const int height_start = this->grid_height_[i];
    for(int j = 0; j < n_width; j++) {
      const int width_start = this->grid_width_[j];
      const Dtype* top_diff = top[j + i * n_width]->gpu_diff();
      // NOLINT_NEXT_LINE(whitespace/operators)
      Box_Gen<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, false, top_diff, bottom_count, this->box_height_, this->box_width_,
        height_start, width_start, this->height_, this->width_, bottom_diff);
    }
  }
}

template <typename Dtype>
void BoxLayer<Dtype>::Box_Merge_Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<Blob<Dtype>*>& bottom) {

  CHECK_EQ(bottom.size(), this->box_num_) << "The number of the bottom blobs must be equal to boxes" \
    << "bottom_num VS. box_num --> " << bottom.size() << " VS. " << this->box_num_;
  CHECK_EQ(top.size(), 1) << "The number of the top blobs must be one" \
    << "top_num VS. 1 --> " << top.size() << " VS. 1";

  const Dtype* top_diff = top[0]->gpu_diff();
  const int num_per_box = top[0]->count(0, 2);
  const int top_count = top[0]->count();
  const int nthreads = num_per_box * this->box_height_ * this->box_width_;
  const int n_height = this->grid_height_.size();
  const int n_width = this->grid_width_.size();

  for(int i = 0; i < n_height; i++) {
    const int height_start = this->grid_height_[i];
    for(int j = 0; j < n_width; j++) {
      const int width_start = this->grid_width_[j];
      Dtype* bottom_diff = bottom[j + i * n_width]->mutable_gpu_diff();
      // NOLINT_NEXT_LINE(whitespace/operators)
      Box_Merge<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, false, top_diff, top_count, this->box_height_, this->box_width_,
        height_start, width_start, this->height_, this->width_, bottom_diff);
    }
  }
}

template <typename Dtype>
void BoxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

  this->Box_Gen_Forward_gpu(bottom, this->bottom_boxes_);
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
  this->Box_Merge_Forward_gpu(this->top_boxes_, top);
}


template <typename Dtype>
void BoxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  this->Box_Merge_Backward_gpu(top, this->top_boxes_);
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
  this->Box_Gen_Backward_gpu(this->bottom_boxes_, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(BoxLayer);

}

