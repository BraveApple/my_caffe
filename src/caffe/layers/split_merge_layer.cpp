#include <vector>
#include <string>
#include <iterator>
#include <sstream>

#include "caffe/layers/split_merge_layer.hpp"


namespace caffe {

template <typename Dtype>
inline void VectorToBlob(const vector<int>& vec, Blob<Dtype>* blob_ptr) {
  blob_ptr->Reshape(1, 1, 1, vec.size());
  for(int i = 0; i < vec.size(); i++) {
    blob_ptr->mutable_cpu_data()[i] = static_cast<Dtype>(vec[i]);
  }
}

template <typename Dtype>
void SplitMergeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const SplitMergeParameter& split_merge_param = this->layer_param_.split_merge_param();
  this->group_num_ = split_merge_param.group_num();
  this->axis_ = split_merge_param.axis();
  this->outer_num_ = bottom[0]->count(0, this->axis_);
  this->inner_num_ = bottom[0]->count(this->axis_ + 1);
  CHECK_EQ(this->group_num_, split_merge_param.group_size())
    << "The number of groups must be equal to group_num";
  
  this->group_vec_.resize(this->group_num_);
  int id_count = 0;
  for (int i = 0; i < this->group_num_; i++) {
    this->group_vec_[i].reset(new Blob<Dtype>());
    const string& str_group = split_merge_param.group(i);
    std::istringstream iss(str_group);
    vector<int> id_vec{std::istream_iterator<int>(iss), std::istream_iterator<int>()};
    // std::copy(id_vec.begin(), id_vec.end(), std::ostream_iterator<int>(std::cout, " "));
    CHECK_GE(id_vec.size(), 1)
      << "The size of id_vec must be greater or equal to 1"; 
    VectorToBlob(id_vec, this->group_vec_[i].get());
    id_count += id_vec.size();
  }
  CHECK_EQ(bottom[0]->shape(this->axis_), id_count)
    << "You miss some ids of groups";
}

template <typename Dtype>
void SplitMergeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  CHECK_EQ(this->group_num_, top.size())
    << "The number of top blobs must be equal to group_num";
  for(int i = 0; i < this->group_num_; i++) {
    vector<int> top_shape(bottom[0]->shape());
    top_shape[this->axis_] = this->group_vec_[i]->count();
    top[i]->Reshape(top_shape);
  }
}

template <typename Dtype>
void SplitMergeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int bottom_dim = bottom[0]->count() / this->outer_num_;
  for(int w = 0; w < this->group_num_; w++) {
    const int top_dim = top[w]->count() / this->outer_num_;
    const Dtype* id_blob = this->group_vec_[w]->cpu_data();
    const int id_count = this->group_vec_[w]->count();
    Dtype* top_data = top[w]->mutable_cpu_data();
    for(int i = 0; i < this->outer_num_; i++) {
      for(int j = 0; j < this->inner_num_; j++) {
        for(int k = 0; k < id_count; k++) {
          const int bottom_id = i * bottom_dim + static_cast<int>(id_blob[k]) * this->inner_num_ + j;
          const int top_id = i * top_dim +  k * this->inner_num_ + j;
          top_data[top_id] = bottom_data[bottom_id];
        }
      }
    }
  }
}

template <typename Dtype>
void SplitMergeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  if(!propagate_down[0]) { return; }

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int bottom_dim = bottom[0]->count() / this->outer_num_;
  for(int w = 0; w < this->group_num_; w++) {
    const int top_dim = top[w]->count() / this->outer_num_;
    const Dtype* id_blob = this->group_vec_[w]->cpu_data();
    const int id_count = this->group_vec_[w]->count();
    const Dtype* top_diff = top[w]->cpu_diff();
    for(int i = 0; i < this->outer_num_; i++) {
      for(int j = 0; j < this->inner_num_; j++) {
        for(int k = 0; k < id_count; k++) {
          const int bottom_id = i * bottom_dim + static_cast<int>(id_blob[k]) * this->inner_num_ + j;
          const int top_id = i * top_dim +  k * this->inner_num_ + j;
          bottom_diff[bottom_id] = top_diff[top_id];
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SplitMergeLayer);
#endif

INSTANTIATE_CLASS(SplitMergeLayer);
REGISTER_LAYER_CLASS(SplitMerge);

}  // namespace caffe
