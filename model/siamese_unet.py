import torch
import torch.nn as nn
import torch.nn.functional as F
import logging 

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)

class SiameseUNet(nn.Module):
    def __init__(self):
        super(SiameseUNet, self).__init__()
        
        # Encoder
        self.enc_conv1 = self.conv_block(1, 64)
        self.enc_conv2 = self.conv_block(64, 128)
        self.enc_conv3 = self.conv_block(128, 256)
        self.enc_conv4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder + Spatial Attention
        self.dec_conv4 = self.conv_block(1024 + 512, 512)
        self.att4 = SpatialAttention()
        
        self.dec_conv3 = self.conv_block(512 + 256, 256)
        self.att3 = SpatialAttention()
        
        self.dec_conv2 = self.conv_block(256 + 128, 128)
        self.att2 = SpatialAttention()
        
        self.dec_conv1 = self.conv_block(128 + 64, 64)
        self.att1 = SpatialAttention()
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )
    
    def forward_once(self, x):
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(F.max_pool2d(x1, kernel_size=2))
        x3 = self.enc_conv3(F.max_pool2d(x2, kernel_size=2))
        x4 = self.enc_conv4(F.max_pool2d(x3, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool2d(x4, kernel_size=2))
        return [x1, x2, x3, x4, bottleneck]
    
    def forward(self, t1, t2):
        features_t1 = self.forward_once(t1)
        features_t2 = self.forward_once(t2)
        
        # Compute absolute difference
        diff = [torch.abs(ft1 - ft2) for ft1, ft2 in zip(features_t1, features_t2)]
        
        # Decoder path + Attention
        x = self.dec_conv4(torch.cat([
            F.interpolate(diff[-1], scale_factor=2, mode='bilinear', align_corners=True),
            diff[-2]
        ], dim=1))
        x = self.att4(x)  # Spatial Attention
        
        x = self.dec_conv3(torch.cat([
            F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True),
            diff[-3]
        ], dim=1))
        x = self.att3(x)
        
        x = self.dec_conv2(torch.cat([
            F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True),
            diff[-4]
        ], dim=1))
        x = self.att2(x)
        
        x = self.dec_conv1(torch.cat([
            F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True),
            diff[-5]
        ], dim=1))
        x = self.att1(x)
        
        # Final output layer
        x = self.final_conv(x)
        return torch.sigmoid(x)

def get_model(device_index):
    model = SiameseUNet()
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_index}')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(device_index)}")
        model.to(device)
    else:
        logging.info("No GPU available, using CPU.")
        model.to('cpu')
    
    return model
