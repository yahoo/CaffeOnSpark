#ifndef CAFFE_EXTRA_LAYERS_HPP_
#define CAFFE_EXTRA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"

namespace caffe {

/**
 * interface between Caffe and Spark
 */
template <typename Dtype>
class CoSDataLayer : public Layer<Dtype> {
 public:
  explicit CoSDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~CoSDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual inline const char* type() const { return "CoSData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool ShareInParallel() const { return false; }
  void Reset(const vector<Dtype*>& data);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  int batch_size_;
  int top_size_;
  vector<int> channels_;
  vector<int> height_;
  vector<int> width_;
  vector<int> sample_num_axes_;
  vector<Dtype*> data_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
