#include <vector>

#include "caffe/layers/split_merge_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SplitMergeForward(const int count, const Dtype* bottom_data,
  const Dtype* id_blob, const int bottom_dim, const int top_dim, 
  const int inner_num, Dtype* top_data) {
  
  CUDA_KERNEL_LOOP(top_id, count) {
    const int i = top_id / top_dim;
    const int k = (top_id % top_dim) / inner_num;
    const int j = (top_id % top_dim) % inner_num;
    const int bottom_id = i * bottom_dim + static_cast<int>(id_blob[k]) * inner_num + j;
    top_data[top_id] = bottom_data[bottom_id]; 
  }
}

template <typename Dtype>
__global__ void SplitMergeBackward(const int count, const Dtype* top_diff,
  const Dtype* id_blob, const int bottom_dim, const int top_dim,
  const int inner_num, Dtype* bottom_diff) {

  CUDA_KERNEL_LOOP(top_id, count) {
    const int i = top_id / top_dim;
    const int k = (top_id % top_dim) / inner_num;
    const int j = (top_id % top_dim) % inner_num;
    const int bottom_id = i * bottom_dim + static_cast<int>(id_blob[k]) * inner_num + j;
    bottom_diff[bottom_id] = top_diff[top_id]; 
  }
}

template <typename Dtype>
void SplitMergeLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int bottom_dim = bottom[0]->count() / this->outer_num_;
  for(int w = 0; w < this->group_num_; w++) {
    const int top_dim = top[w]->count() / this->outer_num_;
    const Dtype* id_blob = this->group_vec_[w]->gpu_data();
    Dtype* top_data = top[w]->mutable_gpu_data();
    const int count = top[w]->count();

    SplitMergeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, id_blob, bottom_dim,
      top_dim, this->inner_num_, top_data);
  }
}

template <typename Dtype>
void SplitMergeLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  if(!propagate_down[0]) { return; }

  Dtype* bottom_diff = bottom[0]->mutable_gpu_data();
  const int bottom_dim = bottom[0]->count() / this->outer_num_;
  for(int w = 0; w < this->group_num_; w++) {
    const int top_dim = top[w]->count() / this->outer_num_;
    const Dtype* id_blob = this->group_vec_[w]->gpu_data();
    const Dtype* top_diff = top[w]->gpu_data();
    const int count = top[w]->count();

    SplitMergeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, id_blob, bottom_dim,
      top_dim, this->inner_num_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SplitMergeLayer);

}  // namespace caffe
