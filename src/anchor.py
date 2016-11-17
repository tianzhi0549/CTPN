import numpy as np


class AnchorText:
    def __init__(self):
        self.anchor_num=10

    def generate_basic_anchors(self, sizes, base_size=16):
        """
        :param sizes: [(h1, w1), (h2, w2)...]
        :param base_size
        :return:
        """
        assert(self.anchor_num==len(sizes))
        base_anchor=np.array([0, 0, base_size-1, base_size-1], np.int32)
        anchors=np.zeros((len(sizes), 4), np.int32)
        index=0
        for h, w in sizes:
            anchors[index]=self.scale_anchor(base_anchor, h, w)
            index+=1
        return anchors

    def scale_anchor(self, anchor, h, w):
        x_ctr=(anchor[0]+anchor[2])*0.5
        y_ctr=(anchor[1]+anchor[3])*0.5
        scaled_anchor=anchor.copy()
        scaled_anchor[0]=x_ctr-w/2
        scaled_anchor[2]=x_ctr+w/2
        scaled_anchor[1]=y_ctr-h/2
        scaled_anchor[3]=y_ctr+h/2
        return scaled_anchor

    def apply_deltas_to_anchors(self, boxes_delta, anchors):
        """
            :return [l t r b]
        """
        anchor_y_ctr=(anchors[:, 1]+anchors[:, 3])/2.
        anchor_h=anchors[:, 3]-anchors[:, 1]+1.
        global_coords=np.zeros_like(boxes_delta, np.float32)
        global_coords[:, 1]=np.exp(boxes_delta[:, 1])*anchor_h
        global_coords[:, 0]=boxes_delta[:, 0]*anchor_h+anchor_y_ctr-global_coords[:, 1]/2.
        return np.hstack((anchors[:, [0]], global_coords[:, [0]], anchors[:, [2]],
                          global_coords[:, [0]]+global_coords[:, [1]])).astype(np.float32)

    def basic_anchors(self):
        """
            anchor [l t r b]
        """
        heights=[11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
        widths=[16]
        sizes=[]
        for h in heights:
            for w in widths:
                sizes.append((h, w))
        return self.generate_basic_anchors(sizes)

    def locate_anchors(self, feat_map_size, feat_stride):
        """
            return all anchors on the feature map
        """
        basic_anchors_=self.basic_anchors()
        anchors=np.zeros((basic_anchors_.shape[0]*feat_map_size[0]*feat_map_size[1], 4), np.int32)
        index=0
        for y_ in range(feat_map_size[0]):
            for x_ in range(feat_map_size[1]):
                shift=np.array([x_, y_, x_, y_])*feat_stride
                anchors[index:index+basic_anchors_.shape[0], :]=basic_anchors_+shift
                index+=basic_anchors_.shape[0]
        return anchors
