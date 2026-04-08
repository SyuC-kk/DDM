from functools import reduce
from operator import add

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shutil
from torchvision.models import resnet
from torchvision.models import vgg

from torchvision import transforms
from .base.conv4d import CenterPivotConv4d as Conv4d
from .base.conv4d import DWConv4d, PWConv4d


class HPNLearner(nn.Module):
    def __init__(self, inch):
        super(HPNLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        def make_building_block_dwconv(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(DWConv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(PWConv4d(inch, outch))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128
        
        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block_dwconv(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block_dwconv(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        # Decoder layers
        self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid):
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid)
        hypercorr_sqz4 = self.encoder_layer4to3(hypercorr_sqz4) + hypercorr_sqz4
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_sqz4) + hypercorr_sqz4

        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        hypercorr_decoded = self.decoder1(hypercorr_encoded)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.decoder2(hypercorr_decoded)

        return logit_mask


class Correlation:
    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz1, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz1, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz2, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz2, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            if bsz1 != bsz2:
                bsz = min(bsz1, bsz2)
                support_feat = support_feat[:bsz]
                query_feat = query_feat[:bsz]
            else:
                bsz= bsz2
            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0) #[b,30,30,30,30]
            corrs.append(corr)
        corrs = torch.stack(corrs).transpose(0, 1).contiguous()
        return corrs

class SimpleEncoder(nn.Module):
    def __init__(self):
        super(SimpleEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # 卷积层2：输入通道16，输出通道32，卷积核大小3x3，步长1，填充1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 卷积层3：输入通道32，输出通道64，卷积核大小3x3，步长1，填充1
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 卷积层4：输入通道64，输出通道3，卷积核大小3x3，步长1，填充1
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        # 最大池化层：池化窗口大小2x2，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 输出尺寸：[1, 16, 128, 128]
        x = self.pool(F.relu(self.conv2(x)))  # 输出尺寸：[1, 32, 64, 64]
        x = self.pool(F.relu(self.conv3(x)))  # 输出尺寸：[1, 64, 32, 32]
        x = F.relu(self.conv4(x))  # 输出尺寸：[1, 3, 32, 32]

        # 使用自适应池化层将尺寸调整为 [1, 3, 30, 30]
        x = F.adaptive_avg_pool2d(x, output_size=(30, 30))

        return x

class HCCNet(nn.Module):
    def __init__(self, backbone, use_original_imgsize):
        super(HCCNet, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'dino':
            # self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.backbone = torch.hub.load('./dinov2', 'dinov2_vitb14', source='local')
            nbottlenecks = [12, 12, 12, 12]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]#[12,24,36]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        #self.encoder = SimpleEncoder()


    def multiview(self, img_name, query_feas, args, training):
        n = len(img_name)
        #load query multiview images
        corrs = []
        for i in range(n):
            if args.benchmark == 'coco':
                if training:
                    base_path = '/home/guofei/csy/One-2-3-45-master/exp/coco/trn'
                    fold_path = os.path.join(base_path, img_name[i], 'stage1_8')
                else:
                    base_path = '/home/guofei/csy/One-2-3-45-master/exp/coco/val'
                    fold_path = os.path.join(base_path, img_name[i], 'stage1_8')
            elif args.benchmark == 'pascal':
                if training:
                    base_path = '/home/guofei/csy/One-2-3-45-master/exp/pascal/trn'
                    fold_path = os.path.join(base_path, img_name[i], 'stage1_8')
                else:
                    base_path = '/home/guofei/csy/One-2-3-45-master/exp/pascal/val'
                    fold_path = os.path.join(base_path, img_name[i], 'stage1_8')

            # print(fold_path)
            file_names = os.listdir(fold_path)

            imgs = []
            for image_file in file_names:
                image_path = os.path.join(fold_path, image_file)
                image = cv2.imread(image_path) #[256,256,3]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() #[1,3,256,256]
                image = F.interpolate(image, size=(420, 420), mode='bilinear', align_corners=False).to('cuda') #[1,3,420,420]
                imgs.append(image)
            imgs = torch.cat(imgs, dim=0) #[n,3,420,420] n represents number of multiview images
            feas = self.backbone.get_intermediate_layers(x=imgs, n=range(0, 12), reshape=True) #12*[8,768,30,30]
            corr_s = Correlation.multilayer_correlation(feas, query_feas, stack_ids=self.stack_ids) #[n,12,30,30,30,30]
        #     '''modification'''
        #     oneimg_corr = []
        #     for i in range(len(corr_s)):
        #         corr = corr_s[i]
        #         min_value = corr.min()
        #         max_value = corr.max()
        #         corr_normalized = (corr - min_value) / (max_value - min_value)
        #         #计算归一化后的相关性张量 corr_normalized 的全局平均值
        #         # 这个全局平均值被用作一个阈值判断的依据，以决定是否将当前的相关性张量 corr 添加到列表 oneimg_corr 中
        #         global_mean = corr_normalized.mean().item()
        #         # print(global_mean)
        #         if global_mean > 0.2:
        #             oneimg_corr.append(corr)
        #
        #     if len(oneimg_corr) > 0:
        #         oneimg_corr = torch.mean(torch.stack(oneimg_corr), dim=0).to('cuda')  # [12,30,30,30,30]
        #     else:
        #         oneimg_corr = torch.zeros(12, 30, 30, 30, 30).to('cuda')
        #     corrs.append(oneimg_corr)
        # corrs = torch.stack(corrs)
        # '''end modification'''
            corr_s = torch.mean(corr_s, dim=0) #[12,30,30,30,30]
            # print(corr_s.shape)
            corrs.append(corr_s)

        corrs = torch.stack(corrs)#[Batch,12,30,30,30,30]
        # print(corrs.shape)

        return corrs

    def forward(self, query_img, support_img, support_mask, query_name, args, training):
        #query_img [B,3,420,420]
        with torch.no_grad():
            support_img2 = torch.zeros_like(support_img)
            if args.shot ==1:
                support_img2[:, 0, :, :] = support_img[:, 0, :, :] * support_mask
                support_img2[:, 1, :, :] = support_img[:, 1, :, :] * support_mask
                support_img2[:, 2, :, :] = support_img[:, 2, :, :] * support_mask
            elif args.shot == 5:
                for i in range(args.shot):
                    support_img2[:, i, 0, :, :] = support_img[:, i, 0, :, :] * support_mask[:, i, :, :]
                    support_img2[:, i, 1, :, :] = support_img[:, i, 1, :, :] * support_mask[:, i, :, :]
                    support_img2[:, i, 2, :, :] = support_img[:, i, 2, :, :] * support_mask[:, i, :, :]
                support_img2 = torch.mean(support_img2, dim=1, keepdim=True).squeeze(1)
            support_img = support_img2
            query_feats = self.backbone.get_intermediate_layers(x=query_img, n=range(0, 12), reshape=True)
            support_feats = self.backbone.get_intermediate_layers(x=support_img, n=range(0, 12), reshape=True) #12*[b,768,30,30] tuple 12
            mulv_corrs = self.multiview(query_name, query_feats, args, training) # [batch,12,30,30,30,30]
            # print(mulv_corrs.shape)
            qs_corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids) # [batch, 12，30，30，30，30]
            corr = torch.add(mulv_corrs, qs_corr)
            # print(corr.shape)

        logit_mask = self.hpn_learner(corr) #[20,2,60,60]
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True) #[20,2,420,420]

        return logit_mask

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])

            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
