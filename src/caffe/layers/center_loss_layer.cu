#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ComputeDistance(const int nthreads, const int center_dim, const Dtype* bottom_data, 
  const Dtype* true_label, const Dtype* center_data, Dtype* distance_data) {
  
  // nthreads = this->batch_size_ * this->center_dim_
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int batch_id = index / center_dim;
    const int center_element_id = index % center_dim;
    const int label_value = static_cast<int>(true_label[batch_id]);
    // distance[i] = x[i] - c_{y[i]}
    distance_data[index] = bottom_data[index] - center_data[label_value * center_dim + center_element_id];
  }
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* true_label = bottom[1]->gpu_data();
  const Dtype* center_data = this->blobs_[0]->gpu_data();

  Dtype* distance_data = this->distance_.mutable_gpu_data();

  // Compute distance between data and center
  const int nthreads = this->batch_size_ * this->center_dim_;
  ComputeDistance<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads, this->center_dim_, bottom_data, true_label, center_data, distance_data);
  
  Dtype dot = 0;
  caffe_gpu_dot<Dtype>(nthreads, distance_data, distance_data, &dot);
  Dtype loss = dot / static_cast<Dtype>(this->batch_size_ * 2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void ComputeCenterDiff(const int nthreads, const int batch_size, const int center_dim,
  const Dtype* true_label, const Dtype* distance_data, Dtype* center_diff) {
  
  // nthreads = this->class_num_
  CUDA_KERNEL_LOOP(index, nthreads) {
    int count = 0;
    for (int batch_id = 0; batch_id < batch_size; batch_id++) {
      const int label_value = static_cast<int>(true_label[batch_id]);
      if (label_value == index) {
        count++;
        for (int element_id = 0; element_id < center_dim; element_id++) {
          center_diff[index * center_dim + element_id] -= distance_data[batch_id * center_dim + element_id];
        }
      }
    }

    for (int element_id = 0; element_id < center_dim; element_id++) {
      center_diff[index * center_dim + element_id] /= static_cast<Dtype>(1 + count);
    }
  }
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* true_label = bottom[1]->gpu_data();
  const Dtype* distance_data = this->distance_.gpu_data();
  
  Dtype* center_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  // Compute Center gradient
  if (this->param_propagate_down_[0]) {
     caffe_gpu_set<Dtype>(this->class_num_ * this->center_dim_, 0, center_diff);
     const int nthreads = this->class_num_;
     ComputeCenterDiff<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, this->batch_size_, this->center_dim_, true_label, distance_data, center_diff);
  }  
  if (propagate_down[0]) {    
    const Dtype scale = top[0]->cpu_diff()[0] / static_cast<Dtype>(this->batch_size_);
    caffe_gpu_scale(this->batch_size_ * this->center_dim_, scale, distance_data, bottom_diff); 
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
      << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CenterLossLayer);

} // namesapce caffe
