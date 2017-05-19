#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/benchmark.hpp"
#include "caffe/layers/rotate_transformer_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ComputeRotateCoef(const int nthreads, const Dtype theta_threshold, 
  const Dtype shift_threshold, const Dtype* rotate_data, Dtype* rotate_coef_data) {

  // (N) (cos, sin, shift_y_norm, shift_x_norm)
  CUDA_KERNEL_LOOP(index, nthreads) {
    const Dtype theta = tanh(rotate_data[index * 3]) * theta_threshold;
    const Dtype shift_y_norm = tanh(rotate_data[index * 3 + 1]) * shift_threshold;
    const Dtype shift_x_norm = tanh(rotate_data[index * 3 + 2]) * shift_threshold;
    
    rotate_coef_data[index * 4] = cos(theta);
    rotate_coef_data[index * 4 + 1] = sin(theta);
    rotate_coef_data[index * 4 + 2] = shift_y_norm;
    rotate_coef_data[index * 4 + 3] = shift_x_norm;
  }
}

template <typename Dtype>
__global__ void ComputeSourceGrid(const int nthreads, const int top_H, const int top_W, 
  const Dtype* rotate_coef_data, const Dtype* target_grid_data, Dtype* source_grid_data) {

  // (N, top_H * top_W)
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / (top_H * top_W);
    const int h = (index / top_W) % top_H;
    const int w = index % top_W;

    const int id = h * top_W + w;
    const Dtype target_y_norm = target_grid_data[id * 2];
    const Dtype target_x_norm = target_grid_data[id * 2 + 1];

    const Dtype cos_theta = rotate_coef_data[4 * n];
    const Dtype sin_theta = rotate_coef_data[4 * n + 1];
    const Dtype shift_y_norm = rotate_coef_data[4 * n + 2];
    const Dtype shift_x_norm = rotate_coef_data[4 * n + 3];
    
    const Dtype source_y_norm = cos_theta * target_y_norm + \
      sin_theta * target_x_norm + shift_y_norm;
    const Dtype source_x_norm = -sin_theta * target_y_norm + \
      cos_theta * target_x_norm + shift_x_norm;

    source_grid_data[2 * index] = source_y_norm;
    source_grid_data[2 * index + 1] = source_x_norm;
  }
}

template <typename Dtype>
__global__ void Transform_forward_gpu(const int nthreads, const int C, const int bottom_H, 
  const int bottom_W, const int top_H, const int top_W, const Dtype* offset_data, 
  const Dtype* source_grid_data, const Dtype* bottom_data, Dtype* top_data) {

  // (N, C, top_H, top_W)
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int b = index / (C * top_H * top_W);
    const int c = (index / (top_H * top_W)) % C;
    const int h = (index / top_W) % top_H;
    const int w = index % top_W;

    const Dtype* source_grid_data_tmp = source_grid_data + b * (top_H * top_W * 2);
    const int id = h * top_W + w;

    const Dtype source_y_norm = source_grid_data_tmp[id * 2];
    const Dtype source_x_norm = source_grid_data_tmp[id * 2 + 1];

    const Dtype source_y = (source_y_norm + 1) / 2.0 * bottom_H;
    const Dtype source_x = (source_x_norm + 1) / 2.0 * bottom_W;

    const Dtype* bottom_data_tmp = bottom_data + b * (C * bottom_H * bottom_W) + c * (bottom_H * bottom_W);
    
    const int center_n = floor(source_y);
    const int center_m = floor(source_x);
    for(int i = 0; i < 4; i++) {
      const int n = center_n + offset_data[i * 2];
      const int m = center_m + offset_data[i * 2 + 1];
      if(n >= 0 && n < bottom_H && m >= 0 && m < bottom_W) {
        const Dtype weight =  max(Dtype(0), 1 - abs(source_x - m)) \
          * max(Dtype(0), 1 - abs(source_y - n));
        top_data[index] += weight * bottom_data_tmp[n * bottom_W + m];
      }
    } 
  }
}

template <typename Dtype>
void RotateTransformerLayer<Dtype>::Forward_gpu( const vector<Blob<Dtype>*>& bottom, 
  const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* rotate_data = bottom[1]->gpu_data();
  const Dtype* target_grid_data = this->target_grid_.gpu_data();

  Dtype* source_grid_data = this->source_grid_.mutable_gpu_data();
  Dtype* rotate_coef_data = this->rotate_coef_.mutable_gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

  // Compute rotate_coef_data
  const int nthreads_1 = this->outer_num_;
  ComputeRotateCoef<Dtype><<<CAFFE_GET_BLOCKS(nthreads_1), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads_1, this->theta_threshold_, this->shift_threshold_, rotate_data, rotate_coef_data);
  
  // Compute source_grid_data
  const int nthreads_2 = this->outer_num_ * this->inner_num_;
  ComputeSourceGrid<Dtype><<<CAFFE_GET_BLOCKS(nthreads_2), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads_2, this->top_height_, this->top_width_, rotate_coef_data, 
    target_grid_data, source_grid_data);

  const int nthreads_3 = top[0]->count();
  Transform_forward_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads_3), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads_3, bottom[0]->channels(), this->bottom_height_, this->bottom_width_, this->top_height_, 
    this->top_width_, this->offset_.gpu_data(), source_grid_data, bottom_data, top_data);
}

template <typename Dtype>
__global__ void Transform_backward_gpu_rotate(const int nthreads, const int C, const int bottom_H, 
  const int bottom_W, const int top_H, const int top_W, const Dtype* source_grid_data, const Dtype* offset_data, 
  const Dtype* bottom_data, const Dtype* top_diff, const Dtype* rotate_coef_data, Dtype* rotate_tmp_diff) {

  // (N, C, top_H, top_W)
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int b = index / (C * top_H * top_W);
    const int c = (index / (top_H * top_W)) % C;
    const int h = (index / top_W) % top_H;
    const int w = index % top_W;

    const Dtype* source_grid_data_tmp = source_grid_data +  b * (top_H * top_W * 2);
    const int id = h * top_W + w;

    const Dtype source_y_norm = source_grid_data_tmp[id * 2];
    const Dtype source_x_norm = source_grid_data_tmp[id * 2 + 1];

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
        if(abs(source_y - n) < 1) {
          source_y_norm_diff += (n >= source_y ? 1 : -1) * max(Dtype(0), 1 - abs(source_x - m)) \
            * bottom_data_tmp[n * bottom_W + m] * top_diff[index] * bottom_H / 2.0;
        }

        // Compute the gradient of source_x_norm
        if(abs(source_x - m) < 1) {
          source_x_norm_diff += (m >= source_x ? 1 : -1) * max(Dtype(0), 1 - abs(source_y - n)) \
            * bottom_data_tmp[n * bottom_W + m] * top_diff[index] * bottom_W / 2.0;
        }
      }
    }

    const Dtype target_y_norm = h * 1.0 / top_H * 2 - 1;
    const Dtype target_x_norm = w * 1.0 / top_W * 2 - 1;

    const Dtype cos_theta = rotate_coef_data[4 * b];
    const Dtype sin_theta = rotate_coef_data[4 * b + 1];
    // const Dtype shift_y_norm = rotate_coef_data[4 * b + 2];
    // const Dtype shift_x_norm = rotate_coef_data[4 * b + 3];


    const Dtype source_y_norm_theta_diff = -target_y_norm * sin_theta + target_x_norm * cos_theta;
    const Dtype source_x_norm_theta_diff = -target_y_norm * cos_theta - target_x_norm * sin_theta;

    const int inner_id = c * (top_H * top_W) + h * top_W + w;
    const int inner_num = C * top_H * top_W;
    
    rotate_tmp_diff[(b * 3 + 0) * inner_num + inner_id] = (source_y_norm_diff * source_y_norm_theta_diff + \
      source_x_norm_diff * source_x_norm_theta_diff); 
    
    rotate_tmp_diff[(b * 3 + 1) * inner_num + inner_id] = source_y_norm_diff;
    rotate_tmp_diff[(b * 3 + 2) * inner_num + inner_id] = source_x_norm_diff;
  }
}

template <typename Dtype>
__global__ void ScaleDiff(const int nthreads, const Dtype theta_threshold, const Dtype shift_threshold,
  const Dtype* rotate_data, Dtype* rotate_diff) {

  // (N)
  CUDA_KERNEL_LOOP(index, nthreads) {
    const Dtype theta = rotate_data[index * 3];
    const Dtype shift_y = rotate_data[index * 3 + 1];
    const Dtype shift_x = rotate_data[index * 3 + 2];

    rotate_diff[index * 3] = rotate_diff[index * 3] * \
      (1 - tanh(theta) * tanh(theta)) * theta_threshold;
    rotate_diff[index * 3 + 1] = rotate_diff[index * 3 + 1] * \
      (1 - tanh(shift_y) * tanh(shift_y)) * shift_threshold;
    rotate_diff[index * 3 + 2] = rotate_diff[index * 3 + 2] * \
      (1 - tanh(shift_x) * tanh(shift_x)) * shift_threshold; 
  }
}

template <typename Dtype>
__global__ void Transform_backward_gpu_bottom(const int nthreads, const int C, const int bottom_H, 
  const int bottom_W, const int top_H, const int top_W, const Dtype* source_grid_data, 
  const Dtype* offset_data, const Dtype* top_diff, Dtype* bottom_diff) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int b = index / (C * top_H * top_W);
    const int c = (index / (top_H * top_W)) % C;
    const int h = (index / top_W) % top_H;
    const int w = index % top_W;

    const Dtype* source_grid_data_tmp = source_grid_data + b * (top_H * top_W * 2);
    const int id = h * top_W + w;

    const Dtype source_y_norm = source_grid_data_tmp[id * 2];
    const Dtype source_x_norm = source_grid_data_tmp[id * 2 + 1];

    const Dtype source_y = (source_y_norm + 1) / 2.0 * bottom_H;
    const Dtype source_x = (source_x_norm + 1) / 2.0 * bottom_W;

    const int center_n = floor(source_y); 
    const int center_m = floor(source_x); 
    Dtype* bottom_diff_tmp = bottom_diff + b * (C * bottom_H * bottom_W) + c * (bottom_H * bottom_W);
    for(int i = 0; i < 4; i++) {
      const int n = center_n + offset_data[i * 2];
      const int m = center_m + offset_data[i * 2 + 1];
      if(n >= 0 && n < bottom_H && m >= 0 && m < bottom_W) {
        const Dtype weight = max(Dtype(0), 1 - abs(source_x - m)) * \
          max(Dtype(0), 1 - abs(source_y - n));
        // 在这里我们使用原子操作，保证一个线程对其进行相加更新的保护
        // 如果有其他线程进行操作，必须要等待该线程结束之后。才能获取操作权限，
        // 但是这样会牺牲掉性能。
        caffe_gpu_atomic_add(weight * top_diff[index], bottom_diff_tmp + (n * bottom_W + m));
      }
    }
  }
}

template <typename Dtype>
void RotateTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* source_grid_data = this->source_grid_.gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* rotate_coef_data = this->rotate_coef_.mutable_gpu_data();
  const Dtype* rotate_data = bottom[1]->gpu_data();

  Dtype* rotate_diff = bottom[1]->mutable_gpu_diff();
  Dtype* rotate_tmp_diff = this->rotate_tmp_.mutable_gpu_diff();

  const int nthreads_1 = top[0]->count();
  Transform_backward_gpu_rotate<Dtype><<<CAFFE_GET_BLOCKS(nthreads_1), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads_1, bottom[0]->channels(), this->bottom_height_, this->bottom_width_, 
    this->top_height_, this->top_width_, source_grid_data, this->offset_.gpu_data(), 
    bottom_data, top_diff, rotate_coef_data, rotate_tmp_diff);

  caffe_gpu_gemv(CblasNoTrans, bottom[1]->count(), top[0]->count(1),
    Dtype(1), rotate_tmp_diff, this->all_one_.gpu_data(), Dtype(0), rotate_diff);
  
  const int nthreads_2 = this->outer_num_;
  ScaleDiff<Dtype><<<CAFFE_GET_BLOCKS(nthreads_2), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads_2, this->theta_threshold_, this->shift_threshold_, rotate_data, rotate_diff);
  
  if(this->first_bottom_diff_) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const int nthreads = top[0]->count();

    Transform_backward_gpu_bottom<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, bottom[0]->channels(), this->bottom_height_, this->bottom_width_, 
      this->top_height_, this->top_width_, source_grid_data, this->offset_.gpu_data(), 
      top_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RotateTransformerLayer);

} // namesapce caffe
