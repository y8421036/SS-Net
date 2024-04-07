from collections import OrderedDict
import torch
import torch.nn as nn
from ffc import FFC_BN_ACT, ConcatTupleLayer


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SSNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32, ratio_in=0.5, ffc=True, skip_ffc=False,
                 cat_merge=True):
        super(SSNet, self).__init__()

        self.ffc = ffc
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge

        features = init_features
        self.encoder1 = SSNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = SSNet._block(features, features * 2, name="enc2") 
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = SSNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = SSNet._block(features * 4, features * 4, name="enc4") 
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        if ffc:
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool3d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool2_f = nn.MaxPool3d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool3d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in) 
            self.pool4_f = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = SSNet._block(features * 8, features * 8, name="bottleneck") 

        if skip_ffc:
            self.upconv4 = nn.ConvTranspose3d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder4 = SSNet._block((features * 8) * 2, features * 8, name="dec4") 
            self.upconv3 = nn.ConvTranspose3d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = SSNet._block((features * 6) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose3d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = SSNet._block((features * 3) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose3d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = SSNet._block(features * 3, features, name="dec1") 

        else:
            self.upconv4 = nn.ConvTranspose3d(
                features * 16, features * 8, kernel_size=2, stride=2
            )
            self.decoder4 = SSNet._block((features * 6) * 2, features * 8, name="dec4")
            self.upconv3 = nn.ConvTranspose3d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = SSNet._block((features * 4) * 2, features * 4, name="dec3")
            self.upconv2 = nn.ConvTranspose3d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = SSNet._block((features * 2) * 2, features * 2, name="dec2")
            self.upconv1 = nn.ConvTranspose3d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = SSNet._block(features * 2, features, name="dec1") 

        self.final1 = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.final2 = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()
        self.catLayer = ConcatTupleLayer()

        initialize_weights(self)

    def apply_fft(self, inp, batch):
        ffted = torch.fft.fftn(inp)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        return ffted

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        enc3 = self.encoder3(self.pool2(enc2))
        enc3_2 = self.pool3(enc3) 

        if self.ffc:
            enc1_f = self.encoder1_f(x)
            enc1_l, enc1_g = enc1_f
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))
            
            enc3_l, enc3_g = enc3_f
            if self.ratio_in == 0:
                enc3_f2 = self.pool1_f(enc3_l)
            elif self.ratio_in == 1:
                enc3_f2 = self.pool1_f(enc3_g)
            else:
                enc3_f2 = self.catLayer((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))


        if self.cat_merge:
            a = torch.zeros_like(enc3_2)
            b = torch.zeros_like(enc3_f2)

            enc3_2 = enc3_2.view(torch.numel(enc3_2), 1)
            enc3_f2 = enc3_f2.view(torch.numel(enc3_f2), 1)

            bottleneck = torch.cat((enc3_2, enc3_f2), 1)
            bottleneck = bottleneck.view_as(torch.cat((a, b), 1))

        else:
            bottleneck = torch.cat((enc3_2, enc3_f2), 1) 

        bottleneck = self.bottleneck(bottleneck)

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)
        edge = self.sigmoid(self.final1(dec1))
        vessel = self.sigmoid(self.final2(dec1))

        return vessel, edge

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
