#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class LstmLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  LstmLayerTest()
      : blob_bottom_(new Blob<Dtype>(12, 3, 2, 1)),
        blob_bottom2_(new Blob<Dtype>(12, 1, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-0.1);
    filler_param.set_max(0.1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    caffe_set<Dtype>(blob_bottom2_->count(), Dtype(0), blob_bottom2_->mutable_cpu_data());
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~LstmLayerTest() { delete blob_bottom_; delete blob_bottom2_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LstmLayerTest, TestDtypesAndDevices);

TYPED_TEST(LstmLayerTest, TestGradientDefault) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    LSTMParameter* lstm_param =
        layer_param.mutable_lstm_param();
    lstm_param->set_num_output(5);
    lstm_param->mutable_weight_filler()->set_type("uniform");
    lstm_param->mutable_weight_filler()->set_min(-0.01);
    lstm_param->mutable_weight_filler()->set_max(0.01);
    lstm_param->mutable_bias_filler()->set_type("constant");
    lstm_param->mutable_bias_filler()->set_value(0);
    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);

    LstmLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(LstmLayerTest, TestGradientBatchDefault) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    LSTMParameter* lstm_param =
        layer_param.mutable_lstm_param();
    lstm_param->set_num_output(5);
    lstm_param->set_batch_size(3);
    lstm_param->mutable_weight_filler()->set_type("uniform");
    lstm_param->mutable_weight_filler()->set_min(-0.01);
    lstm_param->mutable_weight_filler()->set_max(0.01);
    lstm_param->mutable_bias_filler()->set_type("constant");
    lstm_param->mutable_bias_filler()->set_value(0);
    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_);

    LstmLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_, 0);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}
}  // namespace caffe
