from abc import ABCMeta, abstractmethod
from time import time
import pdb

import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.core.seg.sampler import OHEMPixelSampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy


class ContrastiveSeMaskBaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 feature_channels,
                 dropout_ratio=0.1,
                 cate_w=0.4,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 **kwargs):
        super(ContrastiveSeMaskBaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.feature_channels = feature_channels
        self.cate_w = cate_w
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.banks = {
            "internal": [[], []],
            "external": [[], []]
        }

        self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        pass

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits, cluster_output, cluster_aux = self.forward(inputs)
        losses = self.losses(seg_logits, cluster_output, cluster_aux, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)[0]

    def contrastive_loss(self, features, labels, temp=0.07):
        """
        Adopted from https://github.com/tfzhou/ContrastiveSeg/blob/c3e8eaee606a4c7a8cf2b88d2ccfbd443bc6756f/lib/loss/loss_contrast.py#L91
        """
        n, d = labels.shape[0], labels.device
        cls_weights = torch.tensor([0.0713, 0.2460, 0.2252, 0.7276, 0.2937, 0.2234, 1.0000, 0.3800], device=d)

        # Pairwise matrix of labels
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        sim_with_temp = torch.div(torch.matmul(features, features.T.detach()), temp)
        sim_max, _ = torch.max(sim_with_temp, dim=1, keepdim=True)
        logits = sim_with_temp - sim_max.detach()

        neg_mask = 1 - mask
        logits_mask = (torch.ones(n, n) - torch.eye(n)).to(d)
        mask = mask * logits_mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits + neg_logits)

        denom = mask.sum(1)
        select = denom >= 1
        mean_log_prob_pos = (mask * log_prob).sum(1)[select] / denom[select]
        loss = -mean_log_prob_pos

        loss = loss * cls_weights[labels[select]]
        loss = loss.mean()
        if torch.isnan(loss):
            pdb.set_trace()
        return loss

    def update_bank_and_setup_loss(self, feat, seg_label, chosen, bank_choice, num_sample=None):
        feat = feat.permute((0, 2, 3, 1)).reshape(-1, feat.shape[1])
        seg_label = seg_label.reshape(-1)
        valid_inds = torch.where(seg_label != self.ignore_index)[0]

        rand_idx = valid_inds[torch.randperm(valid_inds.shape[0])[:num_sample]]
        if chosen is None:
            chosen = valid_inds[torch.randperm(valid_inds.shape[0])[:num_sample]]
        else:
            chosen = chosen.view(-1).bool()

        hard_mined = feat[chosen]
        hard_mined_label = seg_label[chosen]

        random_mined = feat[rand_idx].detach()
        random_mined_label = seg_label[rand_idx].detach()

        self.banks[bank_choice][0].append(random_mined)
        self.banks[bank_choice][1].append(random_mined_label)
        if len(self.banks[bank_choice][0]) > 3:
            self.banks[bank_choice][0].pop(0)
            self.banks[bank_choice][1].pop(0)

        optim_feats = torch.cat(self.banks[bank_choice][0] + [hard_mined], dim=0)
        optim_labels = torch.cat(self.banks[bank_choice][1] + [hard_mined_label], dim=0)
        return optim_feats, optim_labels

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, no_norm_cluster_output, seg_aux, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        original_seg_logit = seg_logit

        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        seg_aux = resize(
            input=seg_aux,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        cluster_seg_label = resize(
            input=seg_label.float(),
            size=no_norm_cluster_output.shape[2:],
            mode='nearest',
        ).long()
        # sampler = OHEMPixelSampler(self, min_kept=512)
        # chosen = sampler.sample(original_seg_logit, cluster_seg_label)
        chosen = None
        cluster_output = nn.functional.normalize(no_norm_cluster_output, dim=1)
        e_feats, e_labels = self.update_bank_and_setup_loss(
            cluster_output,
            cluster_seg_label,
            chosen,
            "external",
            num_sample=4096)
        loss["loss_cluster"] = 0.3 * self.contrastive_loss(e_feats, e_labels)

        seg_label = seg_label.squeeze(1)
        loss['loss_ce'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=None,
            ignore_index=self.ignore_index)

        loss['loss_cate'] = self.cate_w * self.loss_decode(
            seg_aux,
            seg_label,
            weight=None,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)

        for k, v in loss.items():
            print(f"{k}: {v.item():.4f}", end="\t")
        print()

        return loss
