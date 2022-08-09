# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# whethre to consider all import modules and perform transformation based on all codes plus all import modules


from .unet_parts import *
import torch.nn as nn


class Two_UNet_Fusion(nn.Module):
    def __init__(self, n_channels_rgb, n_channels_gvf, n_classes, bilinear=False):
        super(Two_UNet_Fusion, self).__init__()
        self.n_channels_rgb = n_channels_rgb
        self.n_channels_gvf = n_channels_gvf
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc_rgb = DoubleConv(n_channels_rgb, 64)
        self.inc_gvf = DoubleConv(n_channels_gvf,64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #self.linear1 = nn.Linear(256*256*2,256*256)
        #self.linear2 = nn.Linear(64,64)
        # self.dropout = nn.Dropout(p=0.2)
        self.outc = OutConv(64, n_classes) #mlp linear fusion
        # self.flatten = nn.Flatten()
        #self.outc = OutConv(64*2, n_classes) #two stream

    def forward(self, x, y):
        #first net
        #x1 = self.inc(x)
        x1 = self.inc_rgb(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_out= self.outc(x)  

        #second net
        y1 = self.inc_gvf(y)
        y2 = self.down1(y1)
        y3 = self.down2(y2)
        y4 = self.down3(y3)
        y5 = self.down4(y4)
        y = self.up1(y5, y4)
        y = self.up2(y, y3)
        y = self.up3(y, y2)
        y = self.up4(y, y1)
        y_out = self.outc(y)  #mlp for fusion
        logits = 0.5* x_out + 0.5 *y_out

        # out = torch.cat([x_out,y_out], dim=1)
        # # out_flatten = self.flatten(out)
        # out_flatten = torch.reshape(out, (-1,))
        # out = self.linear1(out_flatten)
        # logits = out.reshape(1,1,256,256)
        # out_flatten = self.dropout(out_flatten)
        # out_fusion = out_flatten.reshape(64,256,256)
        # logits = self.outc(out)
        return logits
