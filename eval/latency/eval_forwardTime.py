from __future__ import division
import os
import sys
import time

import torch
import numpy as np

from thop import profile


# TensorRT
from utils.latency_measure import compute_latency_ms_tensorrt as compute_latency

# Pytorch
# from utils.latency_measure import compute_latency_ms_pytorch as compute_latency
from FSFNet import FSFNet


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model = FSFNet(19)
    flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 2048),), verbose=False)


    print("params = ", params / 1e6,"M")
    print("FLOPs = ", flops / 1e9, "GB")


    model = model.cuda()

    latency = compute_latency(model, (1, 3, 1024, 2048))
    print("FPS:" ,str(1000./latency))

if __name__ == '__main__':
    main()















