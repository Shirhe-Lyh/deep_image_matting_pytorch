# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:40:13 2019

@author: shirhe-lyh
"""

import torch
import torchvision as tv


VGG16_BN_URL = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'

CFGS = {'all': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
                512, 512, 512, 'M', 512, 512, 512, 'M'],
        '13conv': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
                   512, 512, 512, 'M', 512, 512, 512],
        '10conv': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
                   512, 512, 512]}


class VGGFeatureExtractor(torch.nn.Module):
    """Extract features by VGG networks."""
    
    def __init__(self, cfg=None, init_weights=True):
        """Constructor.
        
        Args:
            cfg: Neural architecture (exclude FC layers) of VGG net.
            init_weights: A boolean indicating whether to initialize the
                weights or not.
        """
        super(VGGFeatureExtractor, self).__init__()
        self._cfg = cfg
        self.features = self._vgg16_bn_features()
        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        """Forward computation.
        
        Args:
            x: A float32 tensor with shape [batch_size, channels, height, width]
        """
        x = self._features(x)
        return x
        
    def _vgg16_bn_features(self):
        if self._cfg is None:
            self._cfg = CFGS['all']
        return tv.models.vgg.make_layers(self._cfg, batch_norm=True)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                              nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
    
    
def vgg16_bn_feature_extractor(pretrained=False, cfg=None, progress=True, 
                               **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGGFeatureExtractor(cfg, **kwargs)
    if pretrained:
        state_dict = tv.models.utils.load_state_dict_from_url(
                VGG16_BN_URL, progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


class DIMDecoder(torch.nn.Module):
    """Decoder of Deep Image Matting."""
    
    def __init__(self, cfg=None, init_weights=True):
        """Constructor.
        Args:
            cfg: Neural architecture (exclude FC layers) of VGG net.
                init_weights: A boolean indicating whether to initialize the
                    weights or not.
        """
        super(DIMDecoder, self).__init__()
        self._cfg = cfg
        self._decoder = self._dim_decoder()
        if init_weights:
            self._init_weights()
            
    def forward(self, x):
        return self._decoder(x)
    
    def _dim_decoder(self):
        if self._cfg is None:
            self._cfg = CFGS['all']
        deconv_strides = [1]
        deconv_channels = [512]
        cfg_reversed = self._cfg[::-1]
        for i, e in enumerate(cfg_reversed):
            if e == 'M':
                deconv_strides.append(2)
                deconv_channels.append(cfg_reversed[i+1])
        layers = []
        in_channels = deconv_channels[0]
        for stride, num_channels in zip(deconv_strides, deconv_channels):
            output_padding = 0 if stride == 1 else 1
            layers += [torch.nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=num_channels,
                kernel_size=5, padding=2, output_padding=output_padding,
                stride=stride)]
            layers += [torch.nn.BatchNorm2d(num_features=num_channels),
                       torch.nn.ReLU(inplace=True)]
            in_channels = num_channels
        return torch.nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                              nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)


class DIM(torch.nn.Module):
    """Variant of Deep Image Matting."""
    
    def __init__(self, cfg=None):
        """Constructor.
        
        Args:
            num_classes: Number of classes.
        """
        super(DIM, self).__init__()
        if cfg is None:
            cfg = CFGS.get('10conv')
        self._head_conv = torch.nn.Conv2d(in_channels=4,
                                          out_channels=3,
                                          kernel_size=5,
                                          padding=2)
        self._head_batchnorm = torch.nn.BatchNorm2d(num_features=3)
        self._head_relu = torch.nn.ReLU(inplace=True)
        self._feature_extractor = vgg16_bn_feature_extractor(
            pretrained=True, cfg=cfg)
        self._decoder = DIMDecoder(cfg)
        self._alpha_conv = torch.nn.Conv2d(in_channels=64, 
                                           out_channels=1,
                                           kernel_size=5, 
                                           padding=2)
        self._sigmoid = torch.nn.Sigmoid()
        # Random initialization
        torch.nn.init.kaiming_normal_(self._head_conv.weight, mode='fan_out',
                                      nonlinearity='relu')
        torch.nn.init.constant_(self._head_batchnorm.weight, 1)
        torch.nn.init.kaiming_normal_(self._alpha_conv.weight, mode='fan_out',
                                      nonlinearity='sigmoid')
        
    def forward(self, x):
        x = self._head_conv(x)
        x = self._head_batchnorm(x)
        x = self._head_relu(x)
        x = self._feature_extractor(x)
        x = self._decoder(x)
        x = self._alpha_conv(x)
        x = self._sigmoid(x)
        return x
    
    def loss(self, pred_alphas, gt_alphas, epsilon=1e-12):
        losses = torch.sqrt(
            torch.mul(pred_alphas - gt_alphas, pred_alphas - gt_alphas) + epsilon)
        loss_value = torch.mean(losses)
        return loss_value
    