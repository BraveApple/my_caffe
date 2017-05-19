#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/benchmark.hpp"
#include "caffe/layers/spatial_transformer_layer.hpp"

namespace caffe {

template <typename Dtype>
inline __device__ Dtype Transform_max(const Dtype x, const Dtype y) {
  return (x >= y) ? x : y;
}

template <typename Dtype>
inline __device__ Dtype Transform_abs(const Dtype x) {
  return (x >= 0) ? x : -x;
}


template <typename Dtype>
__global__ void Transform_set(const int nthreads, const Dtype value, const int size,
  const int i, Dtype* out) {
  
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index * size + i] = value;
  }
}

template <typename Dtype>
__global__ void Transform_copy(const int nthreads, const int size_src, const int id_src,
 const Dtype* src, const int size_dst, const int id_dst, Dtype* dst) {
  
  CUDA_KERNEL_LOOP(index, nthreads) {
    dst[index * size_dst + id_dst] = src[index * size_src + id_src];
  }
}

template <typename Dtype>
__global__ void Transform_forward_gpu(const int nthreads, const int C, const int bottom_H, 
  const int bottom_W, const int top_H, const int top_W, const Dtype* offset_data, 
  const Dtype* input_grid_data, const Dtype* bottom_data, Dtype* top_data) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] = 0.;
    const int b = index / (C * top_H * top_W);
    const int c = (index / (top_H * top_W)) % C;
    const int h = (index / top_W) % top_H;
    const int w = index % top_W;

    const Dtype* input_grid_data_tmp = input_grid_data + b * (top_H * top_W * 2);
    const int id = h * top_W + w;

    const Dtype source_y_norm = input_grid_data_tmp[id * 2];
    const Dtype source_x_norm = input_grid_data_tmp[id * 2 + 1];

    const Dtype source_y = (source_y_norm + 1) / 2.0 * bottom_H;
    const Dtype source_x = (source_x_norm + 1) / 2.0 * bottom_W;

    const Dtype* bottom_data_tmp = bottom_data + b * (C * bottom_H * bottom_W) + c * (bottom_H * bottom_W);
    
    const int center_n = floor(source_y);
    const int center_m = floor(source_x);
    for(int i = 0; i < 4; i++) {
      const int n = center_n + offset_data[i * 2];
      const int m = center_m + offset_data[i * 2 + 1];
      if(n >= 0 && n < bottom_H && m >= 0 && m < bottom_W) {
        const Dtype weight =  Transform_max(Dtype(0), 1 - Transform_abs(source_x - m)) \
          * Transform_max(Dtype(0), 1 - Transform_abs(source_y - n));
        top_data[index] += weight * bottom_data_tmp[n * bottom_W + m];
      }
    } 
  }
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Forward_gpu( const vector<Blob<Dtype>*>& bottom, 
  const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* theta_data = bottom[1]->gpu_data();
  const Dtype* output_grid_data = this->output_grid_.gpu_data();

  Dtype* input_grid_data = this->input_grid_.mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  caffe_gpu_set(this->input_grid_.count(), Dtype(0), input_grid_data);
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

  // Compute input_grid_data
  for(int n = 0; n < this->outer_num_; n++) {
    // Mathematical formula: input_grid_data + this->inner_num_ * 2 * n = output_grid_data * Trans(theta_data + 6 * n)
    // output_grid_data: (top_H * top_W, 3)
    // theta_data + 6 * n: (2, 3)
    // input_grid_data + this->inner_num_ * 2 * n: (top_H * top_W, 2)
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, this->inner_num_, 2, 3,
      Dtype(1), output_grid_data, theta_data + 6 * n, 
      Dtype(0), input_grid_data + this->inner_num_ * 2 * n);
  }

  const int nthreads_2 = top[0]->count();
  Transform_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads_2), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads_2, bottom[0]->channels(), this->bottom_height_, this->bottom_width_, this->top_height_, 
    this->top_width_, this->offset_.gpu_data(), input_grid_data, bottom_data, top_data);
}

template <typename Dtype>
__global__ void Transform_backward_gpu_theta(const int nthreads, const int C, const int bottom_H, 
  const int bottom_W, const int top_H, const int top_W, const Dtype* input_grid_data, const Dtype* offset_data, 
  const Dtype* bottom_data, const Dtype* top_diff, Dtype* theta_tmp_diff) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int b = index / (C * top_H * top_W);
    const int c = (index / (top_H * top_W)) % C;
    const int h = (index / top_W) % top_H;
    const int w = index % top_W;

    const Dtype* input_grid_data_tmp = input_grid_data +  b * (top_H * top_W * 2);
    const int id = h * top_W + w;

    const Dtype source_y_norm = input_grid_data_tmp[id * 2];
    const Dtype source_x_norm = input_grid_data_tmp[id * 2 + 1];

    const Dtype source_y = (source_y_norm + 1) / 2.0 * bottom_H;
    const Dtype source_x = (source_x_norm + 1) / 2.0 * bottom_W;

    const Dtype* bottom_data_tmp = bottom_data + b * (C * bottom_H * bottom_W) + c * (bottom_H * bottom_W);
    Dtype source_y_norm_diff = 0.;
    Dtype source_x_norm_diff = 0.;
    
    const int center_n = floor(source_y);
    const int center_m = floor(source_x);
    for(int i = 0; i < 4; i++) {
      const int n = center_n + offset_data[i * 2];
      const int m = center_m + offset_data[i * 2 + 1];
      
      if(n >= 0 && n < bottom_H && m >= 0 && m < bottom_W) {
        
        // Compute the gradient of source_y_norm
        if(Transform_abs(source_y - n) < 1) {
          source_y_norm_diff += (n >= source_y ? 1 : -1) * Transform_max(Dtype(0), 1 - Transform_abs(source_x - m)) \
            * bottom_data_tmp[n * bottom_W + m] * top_diff[index] * bottom_H / 2.0;
        }

        // Compute the gradient of source_x_norm
        if(Transform_abs(source_x - m) < 1) {
          source_x_norm_diff += (m >= source_x ? 1 : -1) * Transform_max(Dtype(0), 1 - Transform_abs(source_y - n)) \
            * bottom_data_tmp[n * bottom_W + m] * top_diff[index] * bottom_W / 2.0;
        }
      }
    }

    const int inner_id = c * (top_H * top_W) + h * top_W + w;
    const int inner_num = C * top_H * top_W; 
    theta_tmp_diff[(b * 6 + 0) * inner_num + inner_id] = source_x_norm_diff * (w * 1.0 / top_W * 2 - 1);
    theta_tmp_diff[(b * 6 + 1) * inner_num + inner_id] = source_x_norm_diff * (h * 1.0 / top_H * 2 - 1);
    theta_tmp_diff[(b * 6 + 2) * inner_num + inner_id] = source_x_norm_diff;

    theta_tmp_diff[(b * 6 + 3) * inner_num + inner_id] = source_y_norm_diff * (w * 1.0 / top_W * 2 - 1);
    theta_tmp_diff[(b * 6 + 4) * inner_num + inner_id] = source_y_norm_diff * (h * 1.0 / top_H * 2 - 1);
    theta_tmp_diff[(b * 6 + 5) * inner_num + inner_id] = source_y_norm_diff;
  }
}

template <typename Dtype>
__global__ void Transform_backward_gpu_bottom(const int nthreads, const int C, const int bottom_H, 
  const int bottom_W, const int top_H, const int top_W, const Dtype* input_grid_data, 
  const Dtype* offset_data, const Dtype* top_diff, Dtype* bottom_diff) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int b = index / (C * top_H * top_W);
    const int c = (index / (top_H * top_W)) % C;
    const int h = (index / top_W) % top_H;
    const int w = index % top_W;

    const Dtype* input_grid_data_tmp = input_grid_data + b * (top_H * top_W * 2);
    const int id = h * top_W + w;

    const Dtype source_y_norm = input_grid_data_tmp[id * 2];
    const Dtype source_x_norm = input_grid_data_tmp[id * 2 + 1];

    const Dtype source_y = (source_y_norm + 1) / 2.0 * bottom_H;
    const Dtype source_x = (source_x_norm + 1) / 2.0 * bottom_W;

    const int center_n = floor(source_y); 
    const int center_m = floor(source_x); 
    Dtype* bottom_diff_tmp = bottom_diff + b * (C * bottom_H * bottom_W) + c * (bottom_H * bottom_W);
    for(int i = 0; i < 4; i++) {
      const int n = center_n + offset_data[i * 2];
      const int m = center_m + offset_data[i * 2 + 1];
      if(n >= 0 && n < bottom_H && m >= 0 && m < bottom_W) {
        const Dtype weight = Transform_max(Dtype(0), 1 - Transform_abs(source_x - m)) * \
          Transform_max(Dtype(0), 1 - Transform_abs(source_y - n));
        // 在这里我们使用原子操作，保证一个线程对其进行相加更新的保护
        // 如果有其他线程进行操作，必须要等待该线程结束之后。才能获取操作权限，
        // 但是这样会牺牲掉性能。
        caffe_gpu_atomic_add(weight * top_diff[index], bottom_diff_tmp + (n * bottom_W + m));
      }
    }
  }
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* input_grid_data = this->input_grid_.gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();

  Dtype* theta_diff = bottom[1]->mutable_gpu_diff();
  Dtype* theta_tmp_diff = this->theta_tmp_.mutable_gpu_diff();

  const int nthreads_1 = top[0]->count();
  Transform_backward_gpu_theta<Dtype><<<CAFFE_GET_BLOCKS(nthreads_1), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads_1, bottom[0]->channels(), this->bottom_height_, this->bottom_width_, 
    this->top_height_, this->top_width_, input_grid_data, this->offset_.gpu_data(), 
    bottom_data, top_diff, theta_tmp_diff);

  caffe_gpu_gemv(CblasNoTrans, bottom[1]->count(), top[0]->count(1),
    Dtype(1), theta_tmp_diff, this->all_one_.mutable_gpu_data(), Dtype(0), theta_diff);

  if(this->first_bottom_diff_) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const int nthreads = top[0]->count();

    Transform_backward_gpu_bottom<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, bottom[0]->channels(), this->bottom_height_, this->bottom_width_, 
      this->top_height_, this->top_width_, input_grid_data, this->offset_.gpu_data(), 
      top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerLayer);

} // namesapce caffe
