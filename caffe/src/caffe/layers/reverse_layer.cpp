#include "caffe/common_layers.hpp"

namespace caffe {
template <typename Dtype>
void reverse_cpu(const int count, const Dtype* from_data, Dtype* to_data, 
	const int* counts, const int axis_count, const int axis) {
	for(int index=0; index<count; index++) {
		int ind=(index/counts[axis])%axis_count;
		int to_index=counts[axis]*(axis_count-2*ind-1)+index;
		*(to_data+to_index)=*(from_data+index);
	}
}
template <typename Dtype>
void ReverseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
	CHECK_NE(bottom[0], top[0])<<this->type()<<" does not support in-place computation.";
	reverse_param_=this->layer_param_.reverse_param();
}

template <typename Dtype>
void ReverseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	vector<int> shape=bottom[0]->shape();
	axis_=reverse_param_.axis();
	CHECK_GT(shape.size(), 0)<<this->type()<<" does not support 0 axes blob.";
	CHECK_GE(axis_, 0)<<"axis must be greater than or equal to 0.";
	CHECK_LT(axis_, shape.size())<<"axis must be less than bottom's dimension.";
	top[0]->ReshapeLike(*bottom[0]);
	const int dim=shape.size();
	shape.clear();
	shape.push_back(dim);
	bottom_counts_.Reshape(shape);
	int* p=bottom_counts_.mutable_cpu_data();
	for (int i=1; i<dim; i++) {
		*p=bottom[0]->count(i);
		p++;
	}
	*p=1;
}

template <typename Dtype>
void ReverseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
		const vector<Blob<Dtype>*>& top) {
	reverse_cpu<Dtype>(bottom[0]->count(), bottom[0]->cpu_data(), 
		top[0]->mutable_cpu_data(), bottom_counts_.cpu_data(), 
		bottom[0]->shape(axis_), axis_);
}

template <typename Dtype>
void ReverseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) {
		return;
	}
	reverse_cpu<Dtype>(bottom[0]->count(), top[0]->cpu_diff(), 
		bottom[0]->mutable_cpu_diff(), bottom_counts_.cpu_data(), 
		bottom[0]->shape(axis_), axis_);
}

#ifdef CPU_ONLY
STUB_GPU(ReverseLayer);
#endif

INSTANTIATE_CLASS(ReverseLayer);
REGISTER_LAYER_CLASS(Reverse);

}  // namespace caffe
