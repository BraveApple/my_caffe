#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/spatial_transformer_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SpatialTransformerLossLayerForwardGPU(const int nthreads, const int top_height, 
  const int top_width, const Dtype* theta_data, Dtype* loss_data) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    
    const int b = index / (top_height * top_width);
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    
    const Dtype target_y_norm = h * 2.0 / top_height - 1;
    const Dtype target_x_norm = w * 2.0 / top_width - 1;

    const Dtype source_x_norm = theta_data[b * 6] * target_x_norm + \
        theta_data[b * 6 + 1] * target_y_norm + theta_data[b * 6 + 2];
    const Dtype source_y_norm = theta_data[b * 3] * target_x_norm + \
        theta_data[b * 6 + 4] * target_y_norm + theta_data[b * 6 + 5];

    loss_data[index] += (source_x_norm < -1 ? 1 : 0) * (source_x_norm + 1) * (source_x_norm + 1) / 2;
    loss_data[index] += (source_x_norm > 1 ? 1 : 0) * (source_x_norm - 1) * (source_x_norm - 1) / 2;

    loss_data[index] += (source_y_norm < -1 ? 1 : 0) * (source_y_norm + 1) * (source_y_norm + 1) / 2;
    loss_data[index] += (source_y_norm > 1 ? 1 : 0) * (source_y_norm - 1) * (source_y_norm - 1) / 2;
  }
}

template <typename Dtype>
void SpatialTransformerLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  const Dtype* theta_data = bottom[0]->gpu_data();
  Dtype* loss_data = this->loss_.mutable_gpu_data();

  caffe_gpu_set(this->loss_.count(), Dtype(0), loss_data);

  // const int nthreads = this->outer_num_ * this->inner_num_;
  const int nthreads = this->loss_.count();
  SpatialTransformerLossLayerForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
    CAFFE_CUDA_NUM_THREADS>>>(nthreads, this->top_height_, this->top_width_, 
    theta_data, loss_data);

  Dtype loss = 0;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = loss / nthreads;
}

template <typename Dtype>
__global__ void SpatialTransformerLossLayerBackwardGPU(const int nthreads, const int top_height, 
  const int top_width, const Dtype* theta_data, Dtype* theta_tmp_diff) {

  CUDA_KERNEL_LOOP(index, nthreads) {

    const int b = index / (top_height * top_width);
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;

    const Dtype target_y_norm = h * 2.0 / top_height - 1;
    const Dtype target_x_norm = w * 2 / top_width - 1;

    const Dtype source_x_norm = theta_data[b * 6] * target_x_norm + \
        theta_data[b * 6 + 1] * target_y_norm + theta_data[b * 6 + 2];
    const Dtype source_y_norm = theta_data[b * 3] * target_x_norm + \
        theta_data[b * 6 + 4] * target_y_norm + theta_data[b * 6 + 5];

    Dtype source_x_norm_diff = 0;
    Dtype source_y_norm_diff = 0;

    source_x_norm_diff += (source_x_norm < -1 ? 1 : 0) * (source_x_norm + 1);
    source_x_norm_diff += (source_x_norm > 1 ? 1 : 0) * (source_x_norm - 1);

    source_y_norm_diff += (source_y_norm < -1 ? 1 : 0) * (source_y_norm + 1);
    source_y_norm_diff += (source_y_norm > 1 ? 1 : 0) * (source_y_norm - 1);

    const int inner_num = top_height * top_width;
    theta_tmp_diff[(6 * b + 0) * inner_num + h * top_width + w] = source_x_norm_diff * target_x_norm;
    theta_tmp_diff[(6 * b + 1) * inner_num + h * top_width + w] = source_x_norm_diff * target_y_norm;
    theta_tmp_diff[(6 * b + 2) * inner_num + h * top_width + w] = source_x_norm_diff;

    theta_tmp_diff[(6 * b + 3) * inner_num + h * top_width + w] = source_y_norm_diff * target_x_norm;
    theta_tmp_diff[(6 * b + 4) * inner_num + h * top_width + w] = source_y_norm_diff * target_y_norm;
    theta_tmp_diff[(6 * b + 5) * inner_num + h * top_width + w] = source_y_norm_diff;
  }
}

template <typename Dtype>
void SpatialTransformerLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* theta_data = bottom[0]->gpu_data();
  Dtype* theta_diff = bottom[0]->mutable_gpu_data();
  Dtype* theta_tmp_diff = this->theta_tmp_.mutable_gpu_diff();

  const int nthreads = this->theta_tmp_.count();
  SpatialTransformerLossLayerBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), 
    CAFFE_CUDA_NUM_THREADS>>>(nthreads, this->top_height_, this->top_width_, 
    theta_data, theta_tmp_diff);

  caffe_gpu_gemv(CblasNoTrans, this->outer_num_ * 6, this->inner_num_,
    Dtype(1), theta_tmp_diff, this->all_one_.gpu_data(), Dtype(0), theta_diff);

  const Dtype loss = top[0]->cpu_data()[0];
  caffe_gpu_scal(bottom[0]->count(), loss / nthreads, theta_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerLossLayer);

} // namespace caffe
