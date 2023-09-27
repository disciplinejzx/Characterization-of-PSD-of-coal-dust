from typing import Dict
from torch import nn
from torch.nn import functional as F

from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec

from adet.layers import conv_with_kaiming_uniform


BASIS_MODULE_REGISTRY = Registry("BASIS_MODULE")
BASIS_MODULE_REGISTRY.__doc__ = """
Registry for basis module, which produces global bases from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""

import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn.functional import upsample
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, \
    PairwiseDistance
from torch.nn import functional as F
from torch.autograd import Variable

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
__all__ = ['CPAMEnc', 'CPAMDec', 'CCAMDec']


class CPAMEnc(Module):
    """
    CPAM encoding module
    """

    def __init__(self, in_channels, norm_layer):
        super(CPAMEnc, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        self.conv1 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, in_channels, 1, bias=False),
                                norm_layer(in_channels),
                                ReLU(True))

    def forward(self, x):
        b, c, h, w = x.size()

        feat1 = self.conv1(self.pool1(x)).view(b, c, -1)
        feat2 = self.conv2(self.pool2(x)).view(b, c, -1)
        feat3 = self.conv3(self.pool3(x)).view(b, c, -1)
        feat4 = self.conv4(self.pool4(x)).view(b, c, -1)

        return torch.cat((feat1, feat2, feat3, feat4), 2)


class CPAMDec(Module):
    """
    CPAM decoding module
    """

    def __init__(self, in_channels):
        super(CPAMDec, self).__init__()
        self.softmax = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))

        self.conv_query = Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)  # query_conv2
        self.conv_key = Linear(in_channels, in_channels // 4)  # key_conv2
        self.conv_value = Linear(in_channels, in_channels)  # value2

    def forward(self, x, y):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,M)
            returns :
                out : compact position attention feature
                attention map: (H*W)*M
        """
        m_batchsize, C, width, height = x.size()
        m_batchsize, K, M = y.size()

        proj_query = self.conv_query(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # BxNxd
        proj_key = self.conv_key(y).view(m_batchsize, K, -1).permute(0, 2, 1)  # BxdxK
        energy = torch.bmm(proj_query, proj_key)  # BxNxK
        attention = self.softmax(energy)  # BxNxk

        proj_value = self.conv_value(y).permute(0, 2, 1)  # BxCxK
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # BxCxN
        out = out.view(m_batchsize, C, width, height)
        out = self.scale * out + x
        return out


class CCAMDec(Module):
    """
    CCAM decoding module
    """

    def __init__(self):
        super(CCAMDec, self).__init__()
        self.softmax = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))

    def forward(self, x, y):
        """
            inputs :
                x : input feature(N,C,H,W) y:gathering centers(N,K,H,W)
            returns :
                out : compact channel attention feature
                attention map: K*C
        """
        m_batchsize, C, width, height = x.size()
        x_reshape = x.view(m_batchsize, C, -1)

        B, K, W, H = y.size()
        y_reshape = y.view(B, K, -1)
        proj_query = x_reshape  # BXC1XN
        proj_key = y_reshape.permute(0, 2, 1)  # BX(N)XC
        energy = torch.bmm(proj_query, proj_key)  # BXC1XC
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B, K, -1)  # BCN

        out = torch.bmm(attention, proj_value)  # BC1N
        out = out.view(m_batchsize, C, width, height)

        out = x + self.scale * out
        return out


class DranHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(DranHead, self).__init__()
        inter_channels = in_channels // 4

        ## Convs or modules for CPAM
        self.conv_cpam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU())  # conv5_s
        self.cpam_enc = CPAMEnc(inter_channels, norm_layer)  # en_s
        self.cpam_dec = CPAMDec(inter_channels)  # de_s
        self.conv_cpam_e = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU())  # conv52

        ## Convs or modules for CCAM
        self.conv_ccam_b = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU())  # conv5_c
        self.ccam_enc = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 8, 1, bias=False),    # inter_channels // 16
                                      norm_layer(inter_channels // 8),
                                      nn.ReLU())  # conv51_c
        self.ccam_dec = CCAMDec()  # de_c
        self.conv_ccam_e = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU())  # conv51

        ## Fusion conv
        self.conv_cat = nn.Sequential(nn.Conv2d(inter_channels * 2, out_channels, 3, padding=1, bias=False),
                                      norm_layer(out_channels),
                                      nn.ReLU())  # conv_f

    def forward(self, multix):
        ## Compact Channel Attention Module(CCAM)
        ccam_b = self.conv_ccam_b(multix)
        ccam_f = self.ccam_enc(ccam_b)
        ccam_feat = self.ccam_dec(ccam_b, ccam_f)

        ## Compact Spatial Attention Module(CPAM)
        cpam_b = self.conv_cpam_b(multix)
        cpam_f = self.cpam_enc(cpam_b).permute(0, 2, 1)  # BKD
        cpam_feat = self.cpam_dec(cpam_b, cpam_f)

        ## Fuse two modules
        ccam_feat = self.conv_ccam_e(ccam_feat)
        cpam_feat = self.conv_cpam_e(cpam_feat)
        feat_sum = self.conv_cat(torch.cat([cpam_feat, ccam_feat], 1))

        return feat_sum


def build_basis_module(cfg, input_shape):
    name = cfg.MODEL.BASIS_MODULE.NAME
    return BASIS_MODULE_REGISTRY.get(name)(cfg, input_shape)


@BASIS_MODULE_REGISTRY.register()
class ProtoNet(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        TODO: support deconv and variable channel width
        """
        # official protonet has a relu after each conv
        super().__init__()
        # fmt: off
        mask_dim          = cfg.MODEL.BASIS_MODULE.NUM_BASES  # 4
        planes            = cfg.MODEL.BASIS_MODULE.CONVS_DIM  # 128
        self.in_features  = cfg.MODEL.BASIS_MODULE.IN_FEATURES  # ["p3", "p4", "p5"]
        self.loss_on      = cfg.MODEL.BASIS_MODULE.LOSS_ON  # True
        norm              = cfg.MODEL.BASIS_MODULE.NORM
        num_convs         = cfg.MODEL.BASIS_MODULE.NUM_CONVS
        self.visualize    = cfg.MODEL.BLENDMASK.VISUALIZE
        # fmt: on

        feature_channels = {k: v.channels for k, v in input_shape.items()}  # {'p3': 256, 'p4': 256, 'p5': 256, 'p6': 256, 'p7': 256}
        conv_block = conv_with_kaiming_uniform(norm, True)  # conv relu bn
        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature], planes, 3, 1))
        tower = []
        tower.append(  # DRANET
            DranHead(planes, planes))
        for i in range(num_convs):
            tower.append(
                conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        tower.append(
            conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Conv2d(planes, mask_dim, 1))
        self.add_module('tower', nn.Sequential(*tower))

        if self.loss_on:
            # fmt: off
            self.common_stride   = cfg.MODEL.BASIS_MODULE.COMMON_STRIDE
            num_classes          = cfg.MODEL.BASIS_MODULE.NUM_CLASSES + 1
            self.sem_loss_weight = cfg.MODEL.BASIS_MODULE.LOSS_WEIGHT
            # fmt: on

            inplanes = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, num_classes, kernel_size=1,
                                                    stride=1))

    def forward(self, features, targets=None):
        for i, f in enumerate(self.in_features):  # self.in_features: ['p3', 'p4', 'p5']
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])
                x_p = F.interpolate(x_p, x.size()[2:], mode="bilinear", align_corners=False)
                # x_p = aligned_bilinear(x_p, x.size(3) // x_p.size(3))
                x = x + x_p
        outputs = {"bases": [self.tower(x)]}  
        a = outputs['bases']
        losses = {}
        # auxiliary thing semantic loss 
        if self.training and self.loss_on:  
            sem_out = self.seg_head(features[self.in_features[0]])  
            # resize target to reduce memory
            gt_sem = targets.unsqueeze(1).float()  
            gt_sem = F.interpolate(
                gt_sem, scale_factor=1 / self.common_stride)  
            seg_loss = F.cross_entropy(
                sem_out, gt_sem.squeeze(1).long())
            losses['loss_basis_sem'] = seg_loss * self.sem_loss_weight  
        elif self.visualize and hasattr(self, "seg_head"):
            outputs["seg_thing_out"] = self.seg_head(features[self.in_features[0]])
        return outputs, losses
