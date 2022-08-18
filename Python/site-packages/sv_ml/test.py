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
#This is a file to test how to call two-stream 

import os
import sys


from sv_wrapper import *
import cv2
import numpy as np
NET_FN = "TWO_STREAM"
IMAGE_FN = "9097.X.png"
RESIZE = 240
import timeit


sw = SVWrapper(NET_FN)
img = cv2.imread(IMAGE_FN,0).astype(np.float32)
# dim = (RESIZE,RESIZE)
# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
sw.set_image(img)
d = {
    "p":[0.0,0.0,0.0], #path point
    "n":[1.0,0.0,0.0], #normal to the cross-sectional plane
    "tx":[0.0,1.0,0.0] #tangent to the cross-sectional plane
}
d_s = json.dumps(d)
start = timeit.default_timer()

sw.segment(d_s)
stop = timeit.default_timer()

print("running time = {} sec".format(stop -start))