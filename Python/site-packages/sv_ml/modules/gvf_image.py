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

import os
import numpy as np
import random

import torch

import cv2
import os
from os import path
import sys
# sys.path.append("..")
from modules.GVF2D import *

RESIZE=240
RE_VALUE=1e-10



def sv_gvf(image):
    blur_kernel_size = 5
    img = cv2.GaussianBlur(image, (blur_kernel_size,blur_kernel_size),0)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_img = 1 - img/255
    # img = img.transpose(2,0,1)

    img /= 255.0 #convert to [0.0 1.0]
    # fmin  = np.min(img[:, :])
    # fmax  = np.max(img[:, :])
    # img = (img-fmin)/(fmax-fmin+RE_VALUE);  #% Normalize f to the range [0,1]
    img = np.expand_dims(img, axis=0)

    img_torch = torch.from_numpy(img)

    u,v = GVF(edge_img,0.2,80)
    mag = np.sqrt(u*u + v*v)
    px = u / (mag + 1e-10)
    py = v / (mag + 1e-10)
    gvf_img = np.zeros((2,RESIZE,RESIZE), dtype=np.float32) #gt_image 3 channels
    gvf_img[0,:,:] = px[0:RESIZE,0:RESIZE]
    gvf_img[1,:,:] = py[0:RESIZE,0:RESIZE]
    gvf_tensor = torch.from_numpy(gvf_img)

    return img_torch, gvf_tensor
        





        