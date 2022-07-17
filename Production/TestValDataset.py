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
import pandas as pd
import os


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

    model = model
    model.eval()

    # An example input you would normally provide to your model's forward() method.
    example = torch.rand(1, 3, 160, 80)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    from Utils.Augmentations import *

    PRIOR_MEAN = torch.tensor([[[0.47008159783297326, 0.3302275149738268, 0.26258365359780844]]])
    PRIOR_STD = torch.tensor([[[0.043578123562233576, 0.03262873283790422, 0.030290686596445255]]])

    data_info = pd.read_csv("../Tools/generate_dep_info/betelnut_val.csv", index_col=0)
    for idx, data in data_info.iterrows():
        img_name = data['rgb']
        src1 = io.imread(img_name, as_gray=False)
        src_h, src_w, c = src1.shape
        dst_tensor = torch.from_numpy(src1)
        dst_tensor = (dst_tensor / 255. - PRIOR_MEAN) / PRIOR_STD
        dst_tensor = dst_tensor.permute(2, 0, 1).unsqueeze(0)
        dst_tensor = F.interpolate(dst_tensor, size=(160, 80))
        output = traced_script_module(dst_tensor)
        pred_label = torch.argmax(output, dim=1)
        pred_label = (255 * pred_label[0]).cpu().numpy().astype(np.uint8)
        result = cv2.resize(pred_label, (src_w, src_h))
        tmp = src1.copy()
        src1[result == 255] = (255, 255, 255)
        tmp = cv2.addWeighted(src1, 0.3, tmp, 0.7, 0)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'../Dataset/Results/ValData/{os.path.basename(img_name)}', tmp)
