#include <vector>
#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/spatial_transformer_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  const SpatialTransformerParameter& spatial_transformer_param = \
    this->layer_param_.spatial_transformer_param();
  this->transformer_type_ = spatial_transformer_param.transformer_type();
  CHECK(this->transformer_type_ == "affine") << "The transformer type only supports affine now!";
  this->sampler_type_ = spatial_transformer_param.sampler_type();
  CHECK(this->sampler_type_ == "bilinear") << "The sampler type only supports affine now!";
  
  CHECK(spatial_transformer_param.has_first_bottom_diff())
    << "You miss first_bottom_diff parameter";
  this->first_bottom_diff_ = spatial_transformer_param.first_bottom_diff();

  this->bottom_height_ = bottom[0]->height();
  this->bottom_width_ = bottom[0]->width();

  this->top_height_ = spatial_transformer_param.top_height();
  if(this->top_height_ == -1) {
    this->top_height_ = bottom[0]->height();
  }
  this->top_width_ = spatial_transformer_param.top_width();
  if(this->top_width_ == -1) {
    this->top_width_ = bottom[0]->width();
  }

  this->outer_num_ = bottom[0]->count(0, 1);
  this->inner_num_ = this->top_height_ * this->top_width_;
  
  this->theta_num_ = spatial_transformer_param.theta_num();
 
  this->offset_.Reshape(1, 1, 1, 8);
  Dtype* offset_ptr = this->offset_.mutable_cpu_data();
  offset_ptr[0] = 0; offset_ptr[1] = 0; // (0, 0)
  offset_ptr[2] = 0; offset_ptr[3] = 1; // (0, 1)
  offset_ptr[4] = 1; offset_ptr[5] = 0; // (1, 0)
  offset_ptr[6] = 1; offset_ptr[7] = 1; // (1, 1)
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
 
  // Reshape top
  vector<int> top_shape = bottom[0]->shape();
  top_shape[2] = this->top_height_;
  top_shape[3] = this->top_width_;
  top[0]->Reshape(top_shape);

  // Reshape output_grid_
  // (top_H * top_W, 3)
  vector<int> output_shape(2);
  output_shape[0] = this->inner_num_;
  output_shape[1] = 3;
  this->output_grid_.Reshape(output_shape);

  // Initialize output_grid_
  Dtype* output_grid_data = this->output_grid_.mutable_cpu_data();
  for(int i = 0; i < this->inner_num_; i++) {
    // Normalize the height of output_grid
    output_grid_data[3 * i] = Dtype(i / this->top_width_) / this->top_height_ * 2 - 1;
    // Normalize the width of output_grid
    output_grid_data[3 * i + 1] = Dtype(i % this->top_width_) / this->top_width_ * 2 - 1;
    // shift value
    output_grid_data[3 * i + 2] = 1; 
  }

  // Reshape input_grid_
  // (N, top_H * top_W, 2)
  vector<int> input_shape(3);
  input_shape[0] = bottom[1]->num();
  input_shape[1] = this->inner_num_;
  input_shape[2] = 2;
  this->input_grid_.Reshape(input_shape);

  // Reshape all_one_
  // (C * top_H * top_W, )
  vector<int> all_one_shape(1);
  all_one_shape[0] = top[0]->count(1);
  this->all_one_.Reshape(all_one_shape);
  caffe_set(this->all_one_.count(), Dtype(1), this->all_one_.mutable_cpu_data()); 

  // Reshape theta_tmp_
  // (N, 6, C * top_H * top_W)
  vector<int> theta_tmp_shape(3);
  theta_tmp_shape[0] = this->outer_num_;
  theta_tmp_shape[1] = this->theta_num_;
  theta_tmp_shape[2] = top[0]->count(1);
  this->theta_tmp_.Reshape(theta_tmp_shape);
}

template <typename Dtype>
Dtype SpatialTransformerLayer<Dtype>::Transform_forward_cpu(const Dtype* bottom_data,
  const Dtype source_y_norm, const Dtype source_x_norm) {
  
  Dtype res = 0.;

  // Compute unnormalized sourced x-axis and y-axis
  // -1 <= source_x_norm, source_y_norm <= 1
  const Dtype source_y = (source_y_norm + 1) / 2.0 * this->bottom_height_;
  const Dtype source_x = (source_x_norm + 1) / 2.0 * this->bottom_width_;

  const int center_n = floor(source_y); 
  const int center_m = floor(source_x); 
  for(int i = 0; i < 4; i++) {
    const int n = center_n + this->offset_.cpu_data()[i * 2];
    const int m = center_m + this->offset_.cpu_data()[i * 2 + 1];
    // Only compute pixels in the bottom feature map
    if(n >= 0 && n < this->bottom_height_ && m >= 0 && m < this->bottom_width_) {
      const Dtype weight = this->max(0, 1 - this->abs(source_x - m)) * this->max(0, 1 - this->abs(source_y - n));
      res += weight * bottom_data[n * this->bottom_width_ + m];
    }
  }
  return res;
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* theta_data = bottom[1]->cpu_data();
  const Dtype* output_grid_data = this->output_grid_.cpu_data();

  Dtype* input_grid_data = this->input_grid_.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  caffe_set(this->input_grid_.count(), Dtype(0), input_grid_data);
  caffe_set(top[0]->count(), Dtype(0), top_data);

  // for each input
  for(int n = 0; n < this->outer_num_; n++) {
    Dtype* input_grid_data_tmp = input_grid_data + (this->inner_num_ * 2) * n;
    // Compute input_grid
    // Mathematical formula: input_grid_data_tmp = output_grid_data * Trans(theta_data + 6 * n)
    // output_grid_data: (top_H * top_W, 3)
    // theta_data + 6 * n: (2, 3)
    // input_grid_data_tmp: (top_H * top_W, 2)
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, this->inner_num_, 2, 3,
      Dtype(1), output_grid_data, theta_data + 6 * n, Dtype(0), input_grid_data_tmp);

    for(int c = 0; c < bottom[0]->channels(); c++) {
      for(int h = 0; h < this->top_height_; h++) { 
        for(int w = 0; w < this->top_width_; w++) {
          
          const int id = h * this->top_width_ + w;
          // Get source normalized y-axis
          const Dtype source_y_norm = input_grid_data_tmp[id * 2];
          // Get source normalized x-axis
          const Dtype source_x_norm = input_grid_data_tmp[id * 2 + 1];
          
          top_data[top[0]->offset(n, c, h, w)] = this->Transform_forward_cpu(
            bottom_data + bottom[0]->offset(n, c, 0, 0), source_y_norm, source_x_norm);
        }
      }
    }
  }
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Transform_backward_cpu(const Dtype top_diff, const Dtype* bottom_data,
  const Dtype source_y_norm, const Dtype source_x_norm, Dtype* bottom_diff, 
  Dtype& source_y_norm_diff, Dtype& source_x_norm_diff) {
 
   // Compute unnormalized sourced x-axis and y-axis
  // -1 <= source_x_norm, source_y_norm <= 1
  const Dtype source_y = (source_y_norm + 1) / 2.0 * this->bottom_height_;
  const Dtype source_x = (source_x_norm + 1) / 2.0 * this->bottom_width_;

  const int center_n = floor(source_y); 
  const int center_m = floor(source_x); 
  for(int i = 0; i < 4; i++) {
    const int n = center_n + this->offset_.cpu_data()[i * 2];
    const int m = center_m + this->offset_.cpu_data()[i * 2 + 1];
    // Only compute pixels in the feature map
    if(n >= 0 && n < this->bottom_height_ && m >= 0 && m < this->bottom_width_) {
      // if bottom[0] is input images, so we skip backward propagation for bottom[0]
      if(this->first_bottom_diff_) {
        const Dtype weight = this->max(0, 1 - this->abs(source_x - m)) * this->max(0, 1 - this->abs(source_y - n));
        bottom_diff[m * this->bottom_width_ + n] += weight * top_diff;
      }
      
      // Compute the gradient of source_y_norm
      if(this->abs(source_y - n) < 1) {
        source_y_norm_diff += (n >= source_y ? 1 : -1) * this->max(0, 1 - this->abs(source_x - m)) \
          * bottom_data[n * this->bottom_width_ + m] * top_diff * this->bottom_height_ / 2.0;
      }

      // Compute the gradient of source_x_norm
      if(this->abs(source_x - m) < 1) {
        source_x_norm_diff += (m >= source_x ? 1 : -1) * this->max(0, 1 - this->abs(source_y - n)) \
          * bottom_data[n * this->bottom_width_ + m] * top_diff * this->bottom_width_ / 2.0;
      }
    }
  }
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* input_grid_data = this->input_grid_.cpu_data();
  const Dtype* output_grid_data = this->output_grid_.cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* theta_diff = bottom[1]->mutable_cpu_diff();

  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  caffe_set(bottom[1]->count(), Dtype(0), theta_diff);

  for(int n = 0; n < this->outer_num_; n++) {
    const Dtype* input_grid_data_tmp = input_grid_data + (this->inner_num_ * 2) * n;
    for(int c = 0; c < bottom[0]->channels(); c++) {
      for(int h = 0; h < this->top_height_; h++) {
        for(int w = 0; w < this->top_width_; w++) {
          
          const int id = h * this->top_width_ + w;
          // Get source normalized y-axis
          const Dtype source_y_norm = input_grid_data_tmp[id * 2]; 
          // Get source normalized x-axis
          const Dtype source_x_norm = input_grid_data_tmp[id * 2 + 1]; 
          
          Dtype source_y_norm_diff = 0.;
          Dtype source_x_norm_diff = 0.;
          
          this->Transform_backward_cpu(top_diff[top[0]->offset(n, c, h, w)], bottom_data + bottom[0]->offset(n, c, 0, 0),
            source_y_norm, source_x_norm, bottom_diff + bottom[0]->offset(n, c, 0, 0), source_y_norm_diff, source_x_norm_diff);
          
          theta_diff[6 * n + 0] += source_x_norm_diff * output_grid_data[3 * id + 1]; // theta_1_1
          theta_diff[6 * n + 1] += source_x_norm_diff * output_grid_data[3 * id]; // theta_1_2
          theta_diff[6 * n + 2] += source_x_norm_diff; // theta_1_3
          
          theta_diff[6 * n + 3] += source_y_norm_diff * output_grid_data[3 * id + 1]; // theta_2_1
          theta_diff[6 * n + 4] += source_y_norm_diff * output_grid_data[3 * id]; // theta_2_2
          theta_diff[6 * n + 5] += source_y_norm_diff; // theta_2_3
        }
      }  
    } 
  }
}

#ifdef CPU_ONLY
STUB_GPU(SpatialTransformerLayer);
#endif

INSTANTIATE_CLASS(SpatialTransformerLayer);
REGISTER_LAYER_CLASS(SpatialTransformer);

} // namespace caffe
