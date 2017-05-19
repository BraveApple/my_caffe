#include <vector>

#include "caffe/layers/multi_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void MultiSigmoidCrossEntropyLossForward(const int count,
  const Dtype* pred_data, const Dtype* truth_label, Dtype* loss_data) {
  
  CUDA_KERNEL_LOOP(i, count) {
      loss_data[i] = pred_data[i] * (truth_label[i] - (pred_data[i] >= 0)) - \
        log(1 + exp(pred_data[i] - 2 * pred_data[i] * (pred_data[i] >= 0)));
  }
}

template <typename Dtype>
void MultiSigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  this->sigmoid_bottom_vec_[0] = bottom[0];
  this->sigmoid_layer_ptr_->Forward(this->sigmoid_bottom_vec_, this->sigmoid_top_vec_);
  const int count = bottom[0]->count();

  const Dtype* pred_data = bottom[0]->gpu_data();
  const Dtype* truth_label = bottom[1]->gpu_data();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  MultiSigmoidCrossEntropyLossForward<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, pred_data, truth_label, loss_data);
  Dtype loss;
  caffe_gpu_asum(count, loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = loss / count;
}

template <typename Dtype>
void MultiSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* pred_data = sigmoid_output_->gpu_data();
    const Dtype* truth_label = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, pred_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), truth_label, bottom_diff);

    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / count;
    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiSigmoidCrossEntropyLossLayer);

}  // namespace caffe
