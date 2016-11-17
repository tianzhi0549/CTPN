import numpy as np
import yaml, caffe
from other import clip_boxes
from anchor import AnchorText


class ProposalLayer(caffe.Layer):
    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._feat_stride = layer_params['feat_stride']
        self.anchor_generator=AnchorText()
        self._num_anchors = self.anchor_generator.anchor_num

        top[0].reshape(1, 4)
        top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        assert bottom[0].data.shape[0]==1, \
            'Only single item batches are supported'

        scores = bottom[0].data[:, self._num_anchors:, :, :]

        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]
        height, width = scores.shape[-2:]

        anchors=self.anchor_generator.locate_anchors((height, width), self._feat_stride)

        scores=scores.transpose((0, 2, 3, 1)).reshape(-1, 1)
        bbox_deltas=bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 2))

        proposals=self.anchor_generator.apply_deltas_to_anchors(bbox_deltas, anchors)

        # clip the proposals in excess of the boundaries of the image
        proposals=clip_boxes(proposals, im_info[:2])

        blob=proposals.astype(np.float32, copy=False)
        top[0].reshape(*(blob.shape))
        top[0].data[...]=blob

        top[1].reshape(*(scores.shape))
        top[1].data[...]=scores

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass
