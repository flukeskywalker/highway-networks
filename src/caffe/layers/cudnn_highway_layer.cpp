#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNHighwayLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
  CHECK_EQ(this->height_out_, this->height_)
      << "Input and output height must be same for Highway Convolution layer.";
  CHECK_EQ(this->width_out_, this->width_)
      << "Input and output width must be same for Highway Convolution layer.";
}

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNHighwayLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  HighwayConvolutionParameter conv_param = this->layer_param_.highway_conv_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  if (conv_param.has_transform_kernel_size()) {
    trans_kernel_h_ = trans_kernel_w_ = conv_param.transform_kernel_size();
  } else {
    trans_kernel_h_ = kernel_h_;
    trans_kernel_w_ = kernel_w_;
  }

  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(trans_kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(trans_kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (conv_param.has_transform_pad()) {
    trans_pad_h_ = trans_pad_w_ = conv_param.transform_pad();
  } else {
    trans_pad_h_ = pad_h_;
    trans_pad_w_ = pad_w_;
  }

  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  CHECK_EQ(stride_h_, 1) << "Highway layer only supports h stride of 1.";
  CHECK_EQ(stride_w_, 1) << "Highway layer only supports w stride of 1.";
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1 && trans_kernel_w_ == 1
      && trans_kernel_h_ == 1 && stride_h_ == 1 && stride_w_ == 1
      && pad_h_ == 0 && pad_w_ == 0 && trans_pad_h_ == 0 && trans_pad_w_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.highway_conv_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.highway_conv_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  conv_out_channels_ = num_output_;
  conv_in_channels_ = channels_;
  CHECK_EQ(conv_out_channels_, conv_in_channels_)
      << "Input and output number of channels must be same for Highway Convolution layer.";
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights for cell inputs
  // - blobs_[1] holds the filter weights for transform gates
  // - blobs_[2] holds the biases for cell inputs
  // - blobs_[3] holds the biases for transform gates

  bias_term_ = this->layer_param_.highway_conv_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(4);
    } else {
      this->blobs_.resize(2);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        conv_out_channels_, conv_in_channels_ / group_, kernel_h_, kernel_w_));
    this->blobs_[1].reset(new Blob<Dtype>(
        conv_out_channels_, conv_in_channels_ / group_,
        trans_kernel_h_, trans_kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.highway_conv_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    if (conv_param.has_transform_weight_filler()) {
      shared_ptr<Filler<Dtype> > transform_weight_filler(GetFiller<Dtype>(
        this->layer_param_.highway_conv_param().transform_weight_filler()));
      transform_weight_filler->Fill(this->blobs_[1].get());
    } else {
      weight_filler->Fill(this->blobs_[1].get());
    }
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
      this->blobs_[3].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.highway_conv_param().bias_filler()));
      bias_filler->Fill(this->blobs_[2].get());
      if (conv_param.has_transform_bias_filler()) {
        shared_ptr<Filler<Dtype> > transform_bias_filler(GetFiller<Dtype>(
          this->layer_param_.highway_conv_param().transform_bias_filler()));
        transform_bias_filler->Fill(this->blobs_[3].get());
      } else {
        bias_filler->Fill(this->blobs_[3].get());
      }
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // Initialize CUDA streams and cuDNN. Twice the number of streams for Highway.
  stream_       = new cudaStream_t[2 * this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_       = new cudnnHandle_t[2 * this->group_ * CUDNN_STREAMS_PER_GROUP];
  workspaceSizeInBytes = 0;
  workspace = NULL;

  for (int g = 0; g < 2 * this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
  }

  // Set the indexing parameters.
  weight_offset_ = (this->num_output_ / this->group_)
      * (this->channels_ / this->group_) * this->kernel_h_ * this->kernel_w_;
  trans_weight_offset_ = (this->num_output_ / this->group_)
      * (this->channels_ / this->group_)
      * this->trans_kernel_h_ * this->trans_kernel_w_;

  bias_offset_ = (this->num_output_ / this->group_);

  // Create filter descriptor.
  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      this->kernel_h_, this->kernel_w_);
  cudnn::createFilterDesc<Dtype>(&trans_filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_,
      this->trans_kernel_h_, this->trans_kernel_w_);

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  for (int i = 0; i < bottom.size(); i++) {
    cudnnTensorDescriptor_t bottom_desc;
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
    bottom_descs_.push_back(bottom_desc);
    cudnnTensorDescriptor_t top_desc;
    cudnn::createTensor4dDesc<Dtype>(&top_desc);
    top_descs_.push_back(top_desc);
    cudnnConvolutionDescriptor_t conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&conv_desc);
    conv_descs_.push_back(conv_desc);
    cudnnConvolutionDescriptor_t trans_conv_desc;
    cudnn::createConvolutionDesc<Dtype>(&trans_conv_desc);
    trans_conv_descs_.push_back(trans_conv_desc);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
  }

  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNHighwayLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }

  // Internal states.
  cell_states.resize(bottom.size());
  transform_gate_states.resize(bottom.size());

  // Shape the tops.
  compute_output_shape();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
    cell_states[top_id].reset(new Blob<Dtype>(num_, num_output_,
        height_out_, width_out_));
    transform_gate_states[top_id].reset(new Blob<Dtype>(num_, num_output_,
        height_out_, width_out_));
  }

  conv_in_height_ = height_;
  conv_in_width_ = width_;
  conv_out_spatial_dim_ = height_out_ * width_out_;

  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  trans_kernel_dim_ = conv_in_channels_ * trans_kernel_h_ * trans_kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  trans_weight_offset_ = conv_out_channels_ * trans_kernel_dim_ / group_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;

  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }

  bottom_offset_ = (this->channels_ / this->group_)
      * this->height_ * this->width_;
  top_offset_ = (this->num_output_ / this->group_)
      * this->height_out_ * this->width_out_;

  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_ / this->group_,
        this->height_, this->width_,
        this->channels_ * this->height_ * this->width_,
        this->height_ * this->width_,
        this->width_, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_ / this->group_,
        this->height_out_, this->width_out_,
        this->num_output_ * this->height_out_ * this->width_out_,
        this->height_out_ * this->width_out_,
        this->width_out_, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, this->pad_h_, this->pad_w_,
        this->stride_h_, this->stride_w_);
    cudnn::setConvolutionDesc<Dtype>(&trans_conv_descs_[i], bottom_descs_[i],
        trans_filter_desc_, this->trans_pad_h_, this->trans_pad_w_,
        this->stride_h_, this->stride_w_);
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
}

template <typename Dtype>
void CuDNNHighwayLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(INFO) << "Not Implemented Yet.";
}

template <typename Dtype>
void CuDNNHighwayLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(INFO) << "Not Implemented Yet.";
}

template <typename Dtype>
CuDNNHighwayLayer<Dtype>::~CuDNNHighwayLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
    cudnnDestroyConvolutionDescriptor(trans_conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);
  cudnnDestroyFilterDescriptor(trans_filter_desc_);

  for (int g = 0; g < 2 * this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  delete [] stream_;
  delete [] handle_;
}

INSTANTIATE_CLASS(CuDNNHighwayLayer);
REGISTER_LAYER_CLASS(CuDNNHighway);

}   // namespace caffe
#endif
