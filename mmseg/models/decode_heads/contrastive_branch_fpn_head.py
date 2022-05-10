import pdb

import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .contrastive_semask_decode_head import ContrastiveSeMaskBaseDecodeHead


@HEADS.register_module()
class ContrastiveBranchFPNHead(ContrastiveSeMaskBaseDecodeHead):
    """
    ReCo style head with simple contrastive loss.
    """

    def __init__(self, feature_strides, **kwargs):
        super(ContrastiveBranchFPNHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
        
        self.cls_scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            cls_scale_head = []
            for k in range(head_length):
                if feature_strides[i] != feature_strides[0]:
                    cls_scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.cls_scale_heads.append(nn.Sequential(*cls_scale_head))

        self.feature_head = nn.Conv2d(self.channels, self.channels, kernel_size=1)

        self.projection_head_cluster = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.feature_channels, kernel_size=1),
        )
        self.projection_head_class = nn.Sequential(
            nn.Conv2d(self.channels, self.num_classes, kernel_size=1),
        )

    def forward(self, inputs):
        x = self._transform_inputs(inputs[0])

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        features = self.feature_head(output)
        output = self.projection_head_class(features)
        cluster_output = self.projection_head_cluster(features)

        # The upper head
        cls_x = self._transform_inputs(inputs[1])
        cls_output = cls_x[0]
        for i in range(1, len(self.feature_strides)):
            # non inplace
            cls_output = cls_output + resize(
                self.cls_scale_heads[i](cls_x[i]),
                size=cls_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        return output, cluster_output, cls_output
