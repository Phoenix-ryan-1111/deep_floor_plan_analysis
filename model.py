import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FloorPlanNet(nn.Module):

    def __init__(self):
        super(FloorPlanNet, self).__init__()

        # Encoder (VGG16 backbone)
        vgg = models.vgg16(pretrained=True)
        self.encoder = nn.Sequential(*list(vgg.features.children())[:30])

        # Freeze encoder layers
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder for boundary detection
        self.boundary_decoder = self._build_decoder(
            512, 256, 3)  # 3 classes for boundary

        # Decoder for room type detection
        self.room_decoder = self._build_decoder(512, 256, 9)  # 9 room types

        # Context modules with skip connections
        self.context_modules = nn.ModuleList([
            self._build_context_module(256),
            self._build_context_module(128),
            self._build_context_module(64),
            self._build_context_module(32)
        ])

    def _build_decoder(self, in_channels, mid_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               mid_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1), nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels,
                               mid_channels // 2,
                               kernel_size=4,
                               stride=2,
                               padding=1), nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, kernel_size=1))

    def _build_context_module(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels))

    def forward(self, x):
        # Encoder
        features = []
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                features.append(x)

        # Boundary decoder
        boundary = self.boundary_decoder(features[-1])

        # Room decoder with context
        room = self.room_decoder(features[-1])
        for i, (feat, context_module) in enumerate(
                zip(features[::-1][1:], self.context_modules)):
            room = F.interpolate(room,
                                 scale_factor=2,
                                 mode='bilinear',
                                 align_corners=False)
            room = context_module(torch.cat([room, feat], dim=1))

        return room, boundary
