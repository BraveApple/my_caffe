#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/benchmark.hpp"
#include "caffe/layers/online_batch_relation_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype PositiveFunc(const Dtype x, const Dtype positive_threshold) {
  
  return (x >= positive_threshold) ? 1 : 0;
}

template <typename Dtype>
__device__ Dtype NegativeFunc(const Dtype x, const Dtype negative_threshold) {

  return (x <= negative_threshold) ? 1 : 0;
}

template <typename Dtype>
__global__ void ComputeBatchRelationMatrix(const int nthreads, const int label_num, 
  const int batch_num, const Dtype* truth_label, Dtype* batch_relation_matrix) {

  // nthreads = label_num * label_num 
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index / label_num;
    const int i = index % label_num;
    for (int n = 0; n < batch_num; n++) {
      const Dtype* truth_label_tmp = truth_label + n * label_num;
      if (truth_label_tmp[j] == truth_label_tmp[i])
        batch_relation_matrix[index] += 1;
    }
    batch_relation_matrix[index] /= batch_num;
  }
}

template <typename Dtype>
__global__ void ComputeRelationLoss(const int nthreads, const int label_num,
  const int batch_num, const Dtype negative_threshold, const Dtype positive_threshold,
  const Dtype* global_relation_matrix, const Dtype* prob_data, Dtype* relation_loss_data) {

  // nthreads = label_num * label_num
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index / label_num;
    const int i = index % label_num;
    Dtype loss_pos = 0;
    Dtype loss_neg = 0;
    for (int n = 0; n < batch_num; n++) {
      const Dtype* prob_data_tmp = prob_data + n * label_num;
      const Dtype loss_tmp = (prob_data_tmp[i] - prob_data_tmp[j]) * (prob_data_tmp[i] - prob_data_tmp[j]);
      loss_pos += loss_tmp;
      loss_neg += (1 - loss_tmp) * (1 - loss_tmp);
    }
    const Dtype relation = global_relation_matrix[index];
    loss_pos *= PositiveFunc<Dtype>(relation, positive_threshold);
    loss_neg *= NegativeFunc<Dtype>(relation, negative_threshold);
    relation_loss_data[index] = loss_pos + loss_neg;
  }
}

template <typename Dtype>
void OnlineBatchRelationLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
  const vector<Blob<Dtype>*>& top) {
  
  this->sigmoid_bottom_vec_[0] = bottom[0];
  this->sigmoid_layer_ptr_->Forward(this->sigmoid_bottom_vec_, this->sigmoid_top_vec_);

  const Dtype* pred_data = bottom[0]->gpu_data();
  const Dtype* truth_label = bottom[1]->gpu_data();

  int nthreads = 0;
  // Compute batch relation matrix
  caffe_gpu_set<Dtype>(this->batch_relation_matrix_ptr_->count(), Dtype(0), 
    this->batch_relation_matrix_ptr_->mutable_gpu_data());
  nthreads = this->batch_relation_matrix_ptr_->count();
  ComputeBatchRelationMatrix<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads, this->label_num_, this->batch_num_, truth_label, this->batch_relation_matrix_ptr_->mutable_gpu_data());

  // cout << "this->batch_relation_matrix_ptr = " << this->batch_relation_matrix_ptr_->cpu_data()[1] << endl;
  
  // Update sum of batch relation matrix
  caffe_gpu_axpby<Dtype>(this->blobs_[0]->count(), Dtype(1), this->batch_relation_matrix_ptr_->gpu_data(),
    this->moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
  
  // Update sum of moving_average_fraction
  this->blobs_[1]->mutable_cpu_data()[0] = this->blobs_[1]->cpu_data()[0] * this->moving_average_fraction_ + 1;

  // Compute normalized global relation matrix
  Dtype scale = 1.0 / this->blobs_[1]->cpu_data()[0];
  caffe_gpu_axpby<Dtype>(this->blobs_[0]->count(), scale, this->blobs_[0]->gpu_data(),
    Dtype(0), this->global_relation_matrix_ptr_->mutable_gpu_data());

  Dtype* relation_loss_data = this->blobs_[0]->mutable_gpu_diff();
  caffe_gpu_set<Dtype>(this->blobs_[0]->count(), Dtype(0), relation_loss_data);
  nthreads = this->blobs_[0]->count();
  const Dtype* prob_data = this->sigmoid_output_ptr_->gpu_data();
  ComputeRelationLoss<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads, this->label_num_ , this->batch_num_, this->negative_threshold_, this->positive_threshold_,
    this->global_relation_matrix_ptr_->gpu_data(), prob_data, relation_loss_data);
  Dtype relation_loss = 0;
  caffe_gpu_asum<Dtype>(this->blobs_[0]->count(), relation_loss_data, &relation_loss);
  top[0]->mutable_cpu_data()[0] = relation_loss * 0.5  / (this->batch_num_ * this->label_num_ * this->label_num_);
}

template <typename Dtype>
__global__ void ComputeBatchRelationDiff(const int nthreads, const int label_num,
  const Dtype negative_threshold, const Dtype positive_threshold, const Dtype* global_relation_matrix, 
  const Dtype* prob_data, Dtype* bottom_diff) {

  // nthreads = bottom[0]->count()
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / label_num;
    const int label_id = index % label_num;
    const Dtype* prob_data_tmp = prob_data + n * label_num;
    Dtype diff_pos = 0;
    Dtype diff_neg = 0;
    for (int j = 0; j < label_num; j++) {
      const Dtype relation = global_relation_matrix[j * label_num + label_id];
      const Dtype prob_tmp = prob_data_tmp[label_id] - prob_data_tmp[j];
      diff_pos += PositiveFunc<Dtype>(relation, positive_threshold) * 2 * prob_tmp * \
        prob_data_tmp[label_id] * (1 - prob_data_tmp[label_id]) * 2;
      
      diff_neg += NegativeFunc<Dtype>(relation, negative_threshold) * 2 * (prob_tmp * prob_tmp - 1) * \
        2 * prob_tmp * prob_data_tmp[label_id] * (1 - prob_data_tmp[label_id]) * 2; 
    }

    bottom_diff[index] = diff_pos + diff_neg;
  }
}

template <typename Dtype>
void OnlineBatchRelationLossLayer<Dtype>::Backward_gpu( const vector<Blob<Dtype>*>& top, 
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  
  if (propagate_down[0]) {
    const Dtype* prob_data = this->sigmoid_output_ptr_->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_data();
    caffe_gpu_set<Dtype>(bottom[0]->count(), Dtype(0), bottom_diff);
    const int nthreads = bottom[0]->count();
    // const Dtype* pred_data = bottom[0]->gpu_data();
    ComputeBatchRelationDiff<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, this->label_num_, this->negative_threshold_, this->positive_threshold_, 
      this->global_relation_matrix_ptr_->gpu_data(), prob_data, bottom_diff);

    const Dtype loss_weight = top[0]->cpu_diff()[0] * 0.5 / (this->batch_num_ * this->label_num_ * this->label_num_);
    caffe_gpu_scal<Dtype>(bottom[0]->count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OnlineBatchRelationLossLayer);

}  // namespace caffe
