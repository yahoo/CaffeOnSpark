
#include <vector>

#include "extra_layers.hpp"

namespace caffe {

template <typename Dtype>
CoSDataLayer<Dtype>::~CoSDataLayer<Dtype>() { }

template <typename Dtype>
void CoSDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  const CoSDataParameter& cos_data_param = this->layer_param_.cos_data_param();
  batch_size_ = cos_data_param.batch_size();
  top_size_ = this->layer_param_.top_size();
  CHECK(top_size_ == cos_data_param.top_size()) <<
    "number of tops in this layer should be equal to number" <<
    " of tops in CoSDataParameter";
  channels_.resize(top_size_);
  height_.resize(top_size_);
  width_.resize(top_size_);
  data_.resize(top_size_);
  sample_num_axes_.resize(top_size_);
  for (int i = 0; i < top_size_; i++) {
    sample_num_axes_[i] = cos_data_param.top(i).sample_num_axes();
    CHECK_LE(sample_num_axes_[i], 3) << "sample number of axes "
                                   << "should be less or equal to 3";
    channels_[i] = cos_data_param.top(i).out_channels() == 0 ?
                   cos_data_param.top(i).channels() : cos_data_param.top(i).out_channels();
    height_[i] = cos_data_param.top(i).out_height() == 0 ?
                   cos_data_param.top(i).height() : cos_data_param.top(i).out_height();
    width_[i] = cos_data_param.top(i).out_width() == 0 ?
                   cos_data_param.top(i).width() : cos_data_param.top(i).out_width();
    bool transpose = cos_data_param.top(i).transpose();
    if (transpose) {
      CHECK_EQ(sample_num_axes_[i], 1)
            << "each sample must be a 1d-vector if transpose is used";
      int shape[] = {channels_[i], batch_size_};
      top[i]->Reshape(vector<int>(shape, shape + 2));
    } else {
      int shape[] = {batch_size_, channels_[i], height_[i], width_[i]};
      // + 1 due to batch_size axis
      top[i]->Reshape(vector<int> (shape, shape + sample_num_axes_[i] + 1));
    }
    LOG(INFO) << "CoSDataLayer Top #" << i << " " << top[i]->shape_string();
  }
}

template <typename Dtype>
void CoSDataLayer<Dtype>::Reset(const vector<Dtype*>& data) {
  CHECK_EQ(data.size(), top_size_);
  for (int i = 0; i < data_.size(); i++) {
    CHECK(data[i]);
    data_[i] = data[i];
  }
}

template <typename Dtype>
void CoSDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top_size_; i++) {
    CHECK(data_[i]) <<
      "CoSDataLayer needs to be initialized by calling Reset";
    top[i]->set_cpu_data(data_[i]);
  }
}

template <typename Dtype>
void CoSDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top_size_; i++) {
    CHECK(data_[i]) <<
      "CoSDataLayer needs to be initialized by calling Reset";
    top[i]->set_gpu_data(data_[i]);
  }
}

INSTANTIATE_CLASS(CoSDataLayer);
REGISTER_LAYER_CLASS(CoSData);

}  // namespace caffe
