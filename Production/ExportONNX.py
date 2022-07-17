import torch
from NetModules.Nets import *
from matplotlib import pyplot as plt
from Utils.Augmentations import *
import glob
import time
from skimage import io
import cv2
import seaborn as sns
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
from NetModules.Networks import *


if __name__ == '__main__':
    model_info_dict = {
        # vgg16_bn, resnet50, resnet18
        'backbone': 'resnet18',
        # FCNNetBaseline BarFormer
        'model_name': 'SimpleFCNBetelnutTF',
        'out_keys': ['block0', 'block1', 'block2', 'block3', 'block4'],
        'in_channels': 3,
        'n_classes': 2,
        # 'top_k_s': 64,
        # 'top_k_c': 16,
        # 'encoder_pos': True,
        # 'decoder_pos': True,
        # 'model_pattern': ['S', 'X', 'C']
    }

    weights_path = '../Checkpoints/Betelnut_10/best_ckpt.pt'
    # An instance of your model.
    model = SimpleFCNBetelnutTF(in_channels=3, n_classes=2)
    checkpoint = torch.load(weights_path, map_location='cpu')
    from collections import OrderedDict
    new_dict = OrderedDict()
    for key, value in checkpoint['model_state_dict'].items():
        key = key.replace('module.', '')
        new_dict[key] = value
    model.load_state_dict(new_dict)

    model = model.cpu()
    model.eval()

    example = torch.rand(1, 3, 160, 80)
    traced_script_module = torch.jit.trace(model, example)
    # traced_script_module.save("model.pt")
    # Export the model
    torch.onnx.export(model,  # model being run
                      example,  # model input (or a tuple for multiple inputs)
                      "model.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                     )

    # python mo.py --input_model G:\Coding\BetelnutDetect\Production\model.onnx --output_dir G:\Coding\BetelnutDetect\Production --data_type FP32

