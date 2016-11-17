#include "caffe/common_layers.hpp"

namespace caffe {

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
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void ReverseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(ReverseLayer);
REGISTER_LAYER_CLASS(Reverse);

}  // namespace caffe
