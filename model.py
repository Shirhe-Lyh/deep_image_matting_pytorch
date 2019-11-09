# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 07:08:58 2019

@author: shirhe-lyh

Implementation of paper:
    Deep Image Matting, Ning Xu, eta., arxiv:1703.03872
"""

import torch
import torchvision as tv

VGG16_BN_MODEL_URL = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'

VGG16_BN_CONFIGS = {
    '13conv':
        [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 
         'M', 512, 512, 512],
    '10conv':
        [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
    }


def make_layers(cfg, batch_norm=False):
    """Copy from: torchvision/models/vgg.
    
    Changs retrue_indices in MaxPool2d from False to True.
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2, 
                                          return_indices=True)]
        else:
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(v), 
                           torch.nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
            in_channels = v
    return torch.nn.Sequential(*layers)


class VGGFeatureExtractor(torch.nn.Module):
    """Feature extractor by VGG network."""
    
    def __init__(self, config=None, batch_norm=True):
        """Constructor.
        
        Args:
            config: The convolutional architecture of VGG network.
            batch_norm: A boolean indicating whether the architecture 
                include Batch Normalization layers or not.
        """
        super(VGGFeatureExtractor, self).__init__()
        self._config = config
        if self._config is None:
            self._config = VGG16_BN_CONFIGS.get('10conv')
        self.features = make_layers(self._config, batch_norm=batch_norm)
        self._indices = None
        
    def forward(self, x):
        self._indices = []
        for layer in self.features:
            if isinstance(layer, torch.nn.modules.pooling.MaxPool2d):
                x, indices = layer(x)
                self._indices.append(indices)
            else:
                x = layer(x)
        return x
    
    
def vgg16_bn_feature_extractor(config=None, pretrained=True, progress=True):
    model = VGGFeatureExtractor(config, batch_norm=True)
    if pretrained:
        state_dict = tv.models.utils.load_state_dict_from_url(
            VGG16_BN_MODEL_URL, progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


class DIM(torch.nn.Module):
    """Deep Image Matting."""
    
    def __init__(self, feature_extractor):
        """Constructor.
        
        Args:
            feature_extractor: Feature extractor, such as VGGFeatureExtractor.
        """
        super(DIM, self).__init__()
        # Head convolution layer, number of channels: 4 -> 3
        self._head_conv = torch.nn.Conv2d(in_channels=4, out_channels=3,
                                          kernel_size=5, padding=2)
        # Encoder
        self._feature_extractor = feature_extractor
        self._feature_extract_config = self._feature_extractor._config
        # Decoder
        self._decode_layers = self.decode_layers()
        # Prediction
        self._final_conv = torch.nn.Conv2d(self._feature_extract_config[0], 1,
                                           kernel_size=5, padding=2)
        self._sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self._head_conv(x)
        x = self._feature_extractor(x)
        indices = self._feature_extractor._indices[::-1]
        index = 0
        for layer in self._decode_layers:
            if isinstance(layer, torch.nn.modules.pooling.MaxUnpool2d):
                x = layer(x, indices[index])
                index += 1
            else:
                x = layer(x)
        x = self._final_conv(x)
        x = self._sigmoid(x)
        return x
    
    def decode_layers(self):
        layers = []
        strides = [1]
        channels = []
        config_reversed = self._feature_extract_config[::-1]
        for i, v in enumerate(config_reversed):
            if v == 'M':
                strides.append(2)
                channels.append(config_reversed[i+1])
        channels.append(channels[-1])
        in_channels = self._feature_extract_config[-1]
        for stride, out_channels in zip(strides, channels):
            if stride == 2:
                layers += [torch.nn.MaxUnpool2d(kernel_size=2, stride=2)]
            layers += [torch.nn.Conv2d(in_channels, out_channels,
                                       kernel_size=5, padding=2),
                       torch.nn.BatchNorm2d(num_features=out_channels),
                       torch.nn.ReLU(inplace=True)]
            in_channels = out_channels
        return torch.nn.Sequential(*layers)
    
    
def loss(alphas_pred, alphas_gt, images=None, epsilon=1e-12):
    losses = torch.sqrt(
        torch.mul(alphas_pred - alphas_gt, alphas_pred - alphas_gt) + 
        epsilon)
    loss = torch.mean(losses)
    if images is not None:
        images_fg_gt = torch.mul(images, alphas_gt)
        images_fg_pred = torch.mul(images, alphas_pred)
        images_fg_error = images_fg_pred - images_fg_gt
        losses_image = torch.sqrt(
            torch.mul(images_fg_error, images_fg_error) + epsilon)
        loss += torch.mean(losses_image)
    return loss