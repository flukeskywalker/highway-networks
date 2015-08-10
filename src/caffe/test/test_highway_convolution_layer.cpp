#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define NUM_IMAGES 2
#define NUM_CHANNELS 3
#define WIDTH 6
#define HEIGHT 4

namespace caffe {

template <typename TypeParam>
class HighwayConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HighwayConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(NUM_IMAGES, NUM_CHANNELS, HEIGHT, WIDTH)),
        blob_bottom_2_(new Blob<Dtype>(NUM_IMAGES, NUM_CHANNELS, HEIGHT, WIDTH)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~HighwayConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HighwayConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(HighwayConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HighwayConvolutionParameter* convolution_param =
      layer_param.mutable_highway_conv_param();
  convolution_param->set_kernel_size(1);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(NUM_CHANNELS);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new CuDNNHighwayLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), NUM_IMAGES);
  EXPECT_EQ(this->blob_top_->channels(), NUM_CHANNELS);
  EXPECT_EQ(this->blob_top_->height(), HEIGHT);
  EXPECT_EQ(this->blob_top_->width(), WIDTH);
  EXPECT_EQ(this->blob_top_2_->num(), NUM_IMAGES);
  EXPECT_EQ(this->blob_top_2_->channels(), NUM_CHANNELS);
  EXPECT_EQ(this->blob_top_2_->height(), HEIGHT);
  EXPECT_EQ(this->blob_top_2_->width(), WIDTH);
  // setting group should not change the shape
  convolution_param->set_num_output(NUM_CHANNELS);
  convolution_param->set_group(NUM_CHANNELS);
  layer.reset(new CuDNNHighwayLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), NUM_IMAGES);
  EXPECT_EQ(this->blob_top_->channels(), NUM_CHANNELS);
  EXPECT_EQ(this->blob_top_->height(), HEIGHT);
  EXPECT_EQ(this->blob_top_->width(), WIDTH);
  EXPECT_EQ(this->blob_top_2_->num(), NUM_IMAGES);
  EXPECT_EQ(this->blob_top_2_->channels(), NUM_CHANNELS);
  EXPECT_EQ(this->blob_top_2_->height(), HEIGHT);
  EXPECT_EQ(this->blob_top_2_->width(), WIDTH);
}

TYPED_TEST(HighwayConvolutionLayerTest, TestSimpleHighwayConvolution) {
  // We will simply see if the convolution succeeds at all.
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  HighwayConvolutionParameter* convolution_param =
      layer_param.mutable_highway_conv_param();
  convolution_param->set_kernel_size(1);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(NUM_CHANNELS);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new CuDNNHighwayLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(HighwayConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HighwayConvolutionParameter* convolution_param =
      layer_param.mutable_highway_conv_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_pad(1);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(NUM_CHANNELS);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  CuDNNHighwayLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
