#%%
import yaml, math, os
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch
import torch.nn.functional as F
import torch.nn as nn

from core.bricks import NormLayer
from core.backbone import MSCANet
from core.decoder import DecoderHead, HamDecoder, OCRDecoder
from core.bifpn import BiFPN, OP_BiFPN, Conv2dStaticSamePadding
from core.losses import (BinaryFocalLoss, lovasz_hinge, FocalLoss,
                         LovaszSoftmax, DiceLoss, FocalTverskyLoss, ComboLoss)
import timm

class MaxViT_B_512(nn.Module):
    def __init__(self):
        super().__init__()
        print('[INFO] Using MAxViT Small Base 384 model.')
        self.model = timm.create_model(
            # 'maxvit_base_tf_512.in1k',
            'maxvit_small_tf_384.in1k',
            pretrained=True,
            features_only=True,
        )
        
    def forward(self, x):
        # input -> B*C*H*W
        out = self.model(x)
        # out = out[1:]
        return out
    
class DROP2D(nn.Module):
    def __init__(self, prob=0.3):  
        super(DROP2D, self).__init__()
        self.dropout = nn.Dropout2d(p=prob)

    def forward(self, inputs):  
        inputs = [self.dropout(input) for input in inputs]
        return inputs

class FurrowDet(nn.Module):
    
    def __init__(self, num_classes, embed_dims=[64,96,192,384,768],
                 fpn_channels=256, config=config):
        super().__init__()

        # Get encoders
        self.rgb_encoder = MaxViT_B_512()
        self.depth_encoder = MaxViT_B_512()

        # limiting channels before BiFPN
        self.p_ch = nn.ModuleList([
            nn.Sequential(
                Conv2dStaticSamePadding(in_channels=embed_dim, out_channels=fpn_channels, kernel_size=1),
                NormLayer(fpn_channels, norm_type=config['norm_typ']),
                nn.SiLU(inplace=True),
                nn.Dropout2d(p=config['drop_prob'])
            )
            for embed_dim in embed_dims
        ])


        # Get BiFPN
        self.bifpn1 = BiFPN(num_channels=fpn_channels, conv_channels=embed_dims)
        self.bifpn2 = BiFPN(num_channels=fpn_channels, conv_channels=embed_dims)

        # nDropoutd
        self.dropout1 = DROP2D(prob=config['drop_prob'])
        self.dropout2 = DROP2D(prob=config['drop_prob'])

        # Get OP_BiFPN
        self.op_bifpn = OP_BiFPN(in_channels=fpn_channels)
        
        self.seg_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(fpn_channels, num_classes, kernel_size=1))
        self.line_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(fpn_channels, num_classes, kernel_size=1))
               
        # define loss here for balance load accross GPUs
        self.criterion_combo = ComboLoss()
        self.criterion_ft =  FocalTverskyLoss()
        self.criterion_focal = FocalLoss()
        self.criterion_cce = nn.CrossEntropyLoss()
        self.criterion_lov = LovaszSoftmax()

    def forward(self, rgb_depth, target=None):

        rgb, depth = rgb_depth
        rgb_feats = self.rgb_encoder(rgb)
        depth_feats = self.depth_encoder(depth)

        feats = [t1 + t2 for t1, t2 in zip(rgb_feats, depth_feats)]

        # control channels
        p_feats = [p_ch(feat) for p_ch, feat in zip(self.p_ch, feats)]

        # pass thorough bifpn
        bifpn1_feats = self.bifpn1(p_feats)
        bifpn1_feats = self.dropout1(bifpn1_feats)
        bifpn2_feats = self.bifpn2(bifpn1_feats)
        bifpn2_feats = self.dropout2(bifpn2_feats)

        feat_seg, feat_line = self.op_bifpn(bifpn2_feats)

        # output
        out_seg = self.seg_conv(feat_seg)
        out_line = self.line_conv(feat_line)

        if self.training and target is not None:
            # print(out_line.shape, out_seg.shape, target[0].shape, target[1].shape)
            # print(target[1].unsqueeze(1).shape)
            loss_seg = self.criterion_cce(out_seg, target[0])
            loss_line = self.criterion_combo(out_line, target[1]) + \
                        self.criterion_lov(out_line, target[1]) + \
                        self.criterion_ft(out_line, target[1].unsqueeze(1))

            loss = (0.5 * loss_seg) + (2 * loss_line)
 
            return {'loss' : loss}, \
                   {'out_seg' : out_seg, 'out_line' : out_line} 
        else:
            return {}, {'out_seg' : out_seg, 'out_line' : out_line} 
        


