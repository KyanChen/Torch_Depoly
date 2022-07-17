import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.builder import build_backbone

class ZYNet(nn.Module):
    def __init__(self, backbone_cfg: dict, neck_cfg:dict, head_cfg: dict):
        super(ZYNet, self).__init__()
        self.backbone = build_backbone(**backbone_cfg)
        self.build_arch(head_cfg)
        self.build_neck(neck_cfg)
        self.img_size = head_cfg['img_size']
        self.backbone_cfg = backbone_cfg
        
    def build_neck(self,neck_cfg:dict):
        in_channels = neck_cfg['in_channels']
        out_channels = neck_cfg['out_channels']
        self.conv1=nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        ])

        self.conv2 = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        
    def build_arch(self, head_cfg):
        in_channels = head_cfg['in_channels']
        feat_channels = head_cfg['feat_channels']
        num_classes = head_cfg['num_classes']

        self.trans_conv1 = nn.Sequential(*[
            nn.Conv2d(in_channels, feat_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[0]),
            nn.ReLU(inplace=True)])
        self.trans_conv2 = nn.Sequential(*[
            nn.Conv2d(feat_channels[0], feat_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[1]),
            nn.ReLU(inplace=True)])
        self.trans_conv3 = nn.Sequential(*[
            nn.Conv2d(feat_channels[1], feat_channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[2]),
            nn.ReLU(inplace=True)])
        self.trans_conv4 = nn.Sequential(*[
            nn.Conv2d(feat_channels[2], feat_channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[3]),
            nn.ReLU(inplace=True)])
        self.trans_conv5 = nn.Sequential(*[
            nn.Conv2d(feat_channels[3], feat_channels[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[4]),
            nn.ReLU(inplace=True)])

        self.trans_conv6 = nn.Sequential(*[
            nn.Conv2d(feat_channels[4], feat_channels[5], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feat_channels[5]),
            nn.ReLU(inplace=True)])

        self.pred = nn.Conv2d(feat_channels[5], num_classes, kernel_size=3, stride=1, padding=1)
        
    def resize_img(self, x):
        x = torch.nn.functional.interpolate(x, (self.img_size, self.img_size))
        return x
    
    def forward(self, x):
        input_data = x
        x, endpoints = self.backbone(x)
        x=self.conv1(x)
        endpoints['block6'] = x
        x=self.pool1(x)
        x=self.conv2(x)
        
        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2., mode='bilinear')
        x = self.trans_conv1(x)
        if 'block6' in endpoints.keys():
            x = x + endpoints['block6']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2., mode='bilinear')
        x = self.trans_conv2(x)
        if 'block5' in endpoints.keys():
            x = x + endpoints['block5']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2., mode='bilinear')
        x = self.trans_conv3(x)
        if 'block4' in endpoints.keys():
            x = x + endpoints['block4']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2., mode='bilinear')
        x = self.trans_conv4(x)
        if 'block3' in endpoints.keys():
            x = x + endpoints['block3']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2., mode='bilinear')
        x = self.trans_conv5(x)
        if 'block2' in endpoints.keys():
            x = x + endpoints['block2']

        x = torch.nn.functional.interpolate(x, align_corners=True, scale_factor=2., mode='bilinear')
        x = self.trans_conv6(x)
        if 'block1' in endpoints.keys():
            x = x + endpoints['block1']
        # import pdb
        # pdb.set_trace()
        pred_feature = self.pred(x)
        pred_prob =  F.softmax(pred_feature, dim=1)
        
        return pred_prob


