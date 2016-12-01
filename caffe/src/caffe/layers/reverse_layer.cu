#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void reverse_gpu(const int nthreads, const Dtype* from_data, Dtype* to_data, 
	const int* counts, const int axis_count, const int axis) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  	int ind=(index/counts[axis])%axis_count;
  	int to_index=counts[axis]*(axis_count-2*ind-1)+index;
  	*(to_data+to_index)=*(from_data+index);
  }
}

template <typename Dtype>
void ReverseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
		const vector<Blob<Dtype>*>& top) {
	const int nthreads=bottom[0]->count();
	reverse_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom[0]->gpu_data(), top[0]->mutable_gpu_data(), 
        bottom_counts_.gpu_data(), bottom[0]->shape(axis_), axis_);
}

template <typename Dtype>
void ReverseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
	const int nthreads=bottom[0]->count();
	reverse_gpu<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff(), 
        bottom_counts_.gpu_data(), bottom[0]->shape(axis_), axis_);
}

INSTANTIATE_LAYER_GPU_FUNCS(ReverseLayer);

}  // namespace caffe
