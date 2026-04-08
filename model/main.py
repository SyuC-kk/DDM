from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
import open_clip
import math
from einops import rearrange

from .base.conv4d import CenterPivotConv4d as Conv4d
from .base.conv4d import DWConv4d, PWConv4d
from .restormer_arch import Restormer
from PIL import Image
from thop import profile
from thop import clever_format

def extract_feat_chossed(img, backbone):
    r""" Extract intermediate features from vit"""
    feats = []
    feat = backbone.conv1(img)
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
    feat = feat.permute(0, 2, 1)

    # B = feat.shape[0]
    # cls_tokens = backbone.class_embedding.expand(B, 1, -1)
    # feat = torch.cat((cls_tokens, feat), dim=1)
    feat = feat + backbone.positional_embedding.to(feat.dtype)

    feat = backbone.patch_dropout(feat)
    feat = backbone.ln_pre(feat)

    feat = feat.permute(1, 0, 2)  # NLD -> LND
    for i in range(backbone.transformer.layers-1):
        feat = backbone.transformer.resblocks[i](feat)
        # feats.append(feat.permute(1, 0, 2).clone())
    temp = backbone.transformer.resblocks[-1].ln_1(feat.clone())
    temp = F.linear(temp, backbone.transformer.resblocks[-1].attn.in_proj_weight, backbone.transformer.resblocks[-1].attn.in_proj_bias)
    N, L, C = temp.shape
    temp = temp.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
    temp = F.linear(temp, backbone.transformer.resblocks[-1].attn.out_proj.weight, backbone.transformer.resblocks[-1].attn.out_proj.bias)
    _, _, temp = temp.tensor_split(3, dim=0)
    temp += feat
    temp = temp + backbone.transformer.resblocks[-1].ls_2(backbone.transformer.resblocks[-1].mlp(backbone.transformer.resblocks[-1].ln_2(temp)))

    # feat = backbone.transformer.resblocks[-1](feat)
    # feats.append(feat.permute(1, 0, 2).clone())
    feats.append(temp.permute(1, 0, 2).clone())

    # feat = backbone.transformer(feat)
    # feat = feat.permute(1, 0, 2)  # LND -> NLD
    # feat = backbone.ln_post(feat)

    return feats


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
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)
            
            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)
        corrs = torch.stack(corrs).transpose(0, 1).contiguous()
        return corrs

class ImageClean(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False):
        super(ImageClean, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def clean(self, x, y):
        x = rearrange(x, 'B C (H W)-> B C H W', H=30)
        y = rearrange(y, 'B C (H W)-> B C H W', H=30)
        b, c, h, w = x.shape

        # q = self.qkv_dwconv(self.qkv(q))
        # kv = self.qkv_dwconv(self.qkv(s))
        # q, _, _ = q.chunk(3, dim=1)
        # _, k, v = kv.chunk(3, dim=1)
        qkv_q = self.qkv_dwconv(self.qkv(x))
        qkv_s = self.qkv_dwconv(self.qkv(y))
        q, _, _ = qkv_q.chunk(3, dim=1)
        _, k, v = qkv_s.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature #[c,c]
        # attn = (q.transpose(-2, -1) @ k) * self.temperature  # [hw, hw]
        # attn = attn.softmax(dim=-1)

        # out = (attn @ v.transpose(-2, -1)) #[hw,hw]
        out = (attn @ v) #[c,c]

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) #[c,c]
        # out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w) #[hw, hw]

        out = self.project_out(out)

        return out

    def forward(self, x, y):
        fea = self.clean(x, y) #[bs,512,30,30]
        fea = rearrange(fea, 'B C H W -> B C (H W)')

        return fea


class DDMNet(nn.Module):
    def __init__(self, backbone, use_original_imgsize):
        super(DDMNet, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'dino':
            self.backbone = torch.hub.load('./dinov2', 'dinov2_vitb14', source='local')
            nbottlenecks = [12, 13, 13, 13]
            self.clip = open_clip.create_model('ViT-B-16', pretrained='dfn2b')
            self.clip.eval()
            self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
            channel = 768
            positional_embedding = self.clip.visual.positional_embedding
            spatial_pos_embed = positional_embedding[1:, None, :]
            orig_size = int(math.sqrt(spatial_pos_embed.shape[0]))
            spatial_pos_embed = spatial_pos_embed.permute(1, 2, 0).reshape(1, channel, orig_size, orig_size)
            spatial_pos_embed = F.interpolate(spatial_pos_embed, size=(30, 30), mode='bilinear')
            spatial_pos_embed = spatial_pos_embed.permute(2, 3, 0, 1).reshape(30 * 30, 1, channel)
            self.clip.visual.positional_embedding.data = spatial_pos_embed.squeeze(1).contiguous()
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        self.dino_img_mean = [0.485, 0.456, 0.406]
        self.dino_img_std = [0.229, 0.224, 0.225]
        self.dino_img_mean = torch.tensor(self.dino_img_mean).view(1, 3, 1, 1).cuda()
        self.dino_img_std = torch.tensor(self.dino_img_std).view(1, 3, 1, 1).cuda()
        self.clip_img_mean = [0.48145466, 0.4578275, 0.40821073]
        self.clip_img_std = [0.26862954, 0.26130258, 0.27577711]
        self.clip_img_mean = torch.tensor(self.clip_img_mean).view(1, 3, 1, 1).cuda()
        self.clip_img_std = torch.tensor(self.clip_img_std).view(1, 3, 1, 1).cuda()
        self.ft = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.imgclean = ImageClean(dim=512, num_heads=1, bias=False)

    def text_preprocess(self, text):
        text = self.tokenizer(text).cuda()
        text_feat = self.clip.encode_text(text)
        text_feat = text_feat.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 30, 30)
        return text_feat
    def img_preprcess(self, img):
        clip_img = (img * self.dino_img_std) + self.dino_img_mean
        clip_img = F.interpolate(clip_img, size=(480, 480), mode='bilinear')
        clip_img = (clip_img - self.clip_img_mean) / self.clip_img_std
        clip_feat = extract_feat_chossed(clip_img, self.clip.visual)[0]  # [bs,900,768]
        clip_feat = self.clip.visual.ln_post(clip_feat)  # [bs,900,768]
        clip_feat = clip_feat @ self.clip.visual.proj  # [bs,900,512]
        clip_feat = rearrange(clip_feat, 'B (H W) C -> B C H W', H=30)  # [bs,512,30,30]
        return clip_feat
    def info_nce_loss(self, image_features, text_features, temperature=0.07):
        batch_size = image_features.shape[0]
        image_features = image_features.reshape(batch_size, -1)
        text_features = text_features.reshape(batch_size, -1)
        similarity_matrix = torch.matmul(image_features, text_features.T) / temperature
        logits_per_image = similarity_matrix
        logits_per_text = similarity_matrix.T
        gen_labels = torch.arange(batch_size).long().to(image_features.device)
        total_loss = (F.cross_entropy(logits_per_image, gen_labels) + F.cross_entropy(logits_per_text, gen_labels)) / 2

        return total_loss, logits_per_image, logits_per_text


    def forward(self, query_img, support_img, support_mask, class_name, args, q_deg, s_deg, training):
        with torch.no_grad():
            support_img2 = torch.zeros_like(support_img)
            support_img2[:, 0, :, :] = support_img[:, 0, :, :] * support_mask
            support_img2[:, 1, :, :] = support_img[:, 1, :, :] * support_mask
            support_img2[:, 2, :, :] = support_img[:, 2, :, :] * support_mask
            support_img = support_img2
            
            query_clip_feat = self.img_preprcess(query_img)
            support_clip_feats = self.img_preprcess(support_img)
            query_deg_feat = self.ft(query_clip_feat)
            supp_deg_feats = self.ft(support_clip_feats)
            text_feat = self.text_preprocess(class_name)
            q_degtype_feat = self.text_preprocess(q_deg)
            s_degtype_feat = self.text_preprocess(s_deg[0])
            
            bsz, ch, hb, wb = text_feat.size()
            text_feat = text_feat.view(bsz, ch, -1)
            text_feat = text_feat / (text_feat.norm(dim=1, p=2, keepdim=True) + 1e-5)
            q_degtype_feat = q_degtype_feat.view(bsz, ch, -1)
            q_degtype_feat = q_degtype_feat / (q_degtype_feat.norm(dim=1, p=2, keepdim=True) + 1e-5) #[20,512,900]
            s_degtype_feat = s_degtype_feat.view(bsz, ch, -1)
            s_degtype_feat = s_degtype_feat / (s_degtype_feat.norm(dim=1, p=2, keepdim=True) + 1e-5)


            bsz, ch, ha, wa = query_clip_feat.size()
            query_clip_feat = query_clip_feat.view(bsz, ch, -1)
            query_clip_feat = query_clip_feat / (query_clip_feat.norm(dim=1, p=2, keepdim=True) + 1e-5)
            support_clip_feats = support_clip_feats.view(bsz, ch, -1)
            support_clip_feats = support_clip_feats / (support_clip_feats.norm(dim=1, p=2, keepdim=True) + 1e-5)
            query_deg_feat = query_deg_feat.view(bsz, ch, -1)
            query_deg_feat = query_deg_feat / (query_deg_feat.norm(dim=1, p=2, keepdim=True) + 1e-5) #[20,512,900]
            supp_deg_feat = supp_deg_feats.view(bsz, ch, -1)
            supp_deg_feat = supp_deg_feat / (supp_deg_feat.norm(dim=1, p=2, keepdim=True) + 1e-5) #[20,512,900]

            loss1, _, _ = self.info_nce_loss(query_deg_feat, q_degtype_feat)
            loss2, _, _ = self.info_nce_loss(supp_deg_feat, s_degtype_feat)

            corrt = torch.bmm(query_clip_feat.transpose(1, 2), text_feat).view(bsz, ha, wa, hb, wb) #single cuda
            corrt = corrt.clamp(min=0).unsqueeze(1)


            s_sub_feat = self.imgclean(supp_deg_feat, support_clip_feats) #[bs,512,900]
            q_sub_feat = self.imgclean(query_deg_feat, query_clip_feat)
            q_sub_feat = self.imgclean(q_sub_feat, s_sub_feat) #[bs,512,900]
            corr_d = torch.bmm(q_sub_feat.transpose(1, 2), text_feat).view(bsz, ha, wa, hb, wb)
            corr_d = corr_d.clamp(min=0).unsqueeze(1)
            query_feats = self.backbone.get_intermediate_layers(x=query_img, n=range(0, 12), reshape=True)
            support_feats = self.backbone.get_intermediate_layers(x=support_img, n=range(0, 12), reshape=True) #12*[b,768,30,30] tuple 12
            fea = rearrange(q_sub_feat, 'B C (H W)-> B C H W', H=30)
            corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids) # [batch, 12，30，30，30，30]
            corr = torch.cat((corr, corr_d), dim=1) #需要修改hpn_learner的nbottlenecks值

        logit_mask = self.hpn_learner(corr)
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)

        return logit_mask, loss1+loss2

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot, args):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            training = False
            # logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx],  batch['class_name'])
            logit_mask, _ = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx],  batch['class_name'], args, batch['query_deg_type'],
                           batch['supp_deg_types'], training=False)
            # macs, params = profile(self, inputs=(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx],  batch['class_name'], args, batch['query_deg_type'],
            #                batch['supp_deg_types'], training), verbose=False)
            # flops, params = clever_format([macs * 2, params], "%.3f")  # FLOPs = MACs * 2
            # print(f"Model FLOPs: {flops}, Parameters: {params}")

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
