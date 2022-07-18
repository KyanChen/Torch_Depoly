import os.path

import torch
from Nets.ZYNet import ZYNet
import cv2
import numpy as np

if __name__ == '__main__':
    n_class = 2
    img_size = 1024
    weights_path = 'Weights/best.resnet.2022-07-10-4110.pth.tar'
    img_file = 'Examples/00000022796.jpg'
    
    device = torch.device('cpu')
    
    model_cfg = dict(
        backbone_cfg=dict(
            name='resnet18',
            num_classes=None,
            in_channels=3,
            pretrained=False,
            out_keys=('block2', 'block3', 'block4', 'block5')),
        neck_cfg=dict(
            in_channels=512,
            out_channels=512
            ),
        head_cfg=dict(
            name='MattingNeck',
            in_channels=512,
            img_size=img_size,
            feat_channels=(512, 256, 128, 64, 64, 32),
            num_classes=n_class
        )
    )

    # create model
    model = ZYNet(**model_cfg)

    checkpoint = torch.load(weights_path, map_location='cpu')
    
    from collections import OrderedDict
    new_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        key = key.replace('module.', '')
        new_dict[key] = value
    model.load_state_dict(new_dict)
    
    model = model.to(device)
    model.eval()

    example = torch.rand(1, 3, img_size, img_size)

    # torch.jit.load("model.pt")
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    save_deploy_w = 'Weights_Deploy/torch_scritp_model.pt'
    if not os.path.exists(save_deploy_w):
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save(save_deploy_w)
    else:
        traced_script_module = torch.jit.load(save_deploy_w)
    
    # test
    img = cv2.imread(img_file)
    src_h, src_w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    dst_tensor = torch.from_numpy(img)
    dst_tensor = dst_tensor.view((1, img_size, img_size, 3))
    dst_tensor = dst_tensor.permute((0, 3, 1, 2)).float()
    output = traced_script_module(dst_tensor)
    pred_label = torch.argmax(output, dim=1)
    pred_label = pred_label.view((img_size, img_size))
    pred_label = (255 * pred_label).cpu().numpy().astype(np.uint8)
    result = cv2.resize(pred_label, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('Examples_Results/pred_deploy.png', result)

    output = model(dst_tensor)
    pred_label = torch.argmax(output, dim=1)
    pred_label = pred_label.view((img_size, img_size))
    pred_label = (255 * pred_label).cpu().numpy().astype(np.uint8)
    result = cv2.resize(pred_label, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('Examples_Results/pred_torch.png', result)


    # from Utils.Augmentations import *

    # PRIOR_MEAN = torch.tensor([[[0.47008159783297326, 0.3302275149738268, 0.26258365359780844]]])
    # PRIOR_STD = torch.tensor([[[0.043578123562233576, 0.03262873283790422, 0.030290686596445255]]])

    # src1 = io.imread('../Dataset/21_cut.bmp')
    # src2 = io.imread('../Dataset/12_cut.bmp', as_gray=True)
    # label_src = io.imread('../Dataset/21_cut.bmp', as_gray=True)
    # src_h, src_w, c = src1.shape
    # nH = 8
    # nW = 28
    # dst_h = src_h - (src_h % nH)
    # dst_w = src_w - (src_w % nW)
    # nPatchH = dst_h // nH
    # nPatchW = dst_w // nW
    # print(nPatchH, nPatchW)
    # # 170 69
    # dst_rgb = cv2.resize(src1, (dst_w, dst_h))
    # dst_tensor = torch.from_numpy(dst_rgb)

    # # dst_tensor = dst_rgb[0:170, 0:69, :]
    # # dst_tensor = torch.from_numpy(dst_tensor)

    # dst_tensor = (dst_tensor / 255. - PRIOR_MEAN) / PRIOR_STD

    # dst_tensor = dst_tensor.view((nH, nPatchH, nW, nPatchW, 3))


    # dst_tensor = dst_tensor.permute((0, 2, 4, 1, 3))
    # dst_tensor = dst_tensor.contiguous().view((-1, 3, nPatchH, nPatchW))

    # # dst_tensor = dst_tensor.permute(2, 0, 1).unsqueeze(0)
    # dst_tensor = F.interpolate(dst_tensor, size=(160, 80))
    # output = traced_script_module(dst_tensor)

    # pred_label = torch.argmax(output, dim=1)  # (nH nW), 80, 40
    # pred_label = pred_label.view((nH, nW, 80, 40))
    # pred_label = pred_label.permute((0, 2, 1, 3))
    # pred_label = pred_label.contiguous().view((nH * 80), (nW * 40))
    # pred_label = (255 * pred_label).cpu().numpy().astype(np.uint8)
    # result = cv2.resize(pred_label, (src_w, src_h))
    # # output = pred_label[30:60, 20:30]
    # # print(output)

    # tmp = src1.copy()
    # src1[result == 255] = (255, 255, 255)
    # tmp = cv2.addWeighted(src1, 0.3, tmp, 0.7, 0)
    # tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('../Dataset/Results/dst_2.bmp', tmp)



    # dst = src1.copy()
    # height, width, _ = src1.shape
    # x_list = np.linspace(0, width, 29).astype(np.int)
    # y_list = np.linspace(0, height, 9).astype(np.int)
    # center_x_y = np.array((int(x_list[1] / 2.), int(y_list[1] / 2.)))
    #
    # n_times = 0
    # n_imgs = []
    # for idx_x in range(len(x_list) - 1):
    #     for idx_y in range(len(y_list) - 1):
    #         src_patch_1 = src1[y_list[idx_y]:y_list[idx_y + 1], x_list[idx_x]:x_list[idx_x + 1], :]
    #         src_patch_2 = src2[y_list[idx_y]:y_list[idx_y + 1], x_list[idx_x]:x_list[idx_x + 1]]
    #
    #         img, depth, label = transforms((src_patch_1, src_patch_2, src_patch_2))
    #         # depth = np.expand_dims(depth, axis=2)
    #         # img = np.concatenate([img, depth], axis=2)
    #
    #         img = torch.from_numpy(img).permute(2, 0, 1)
    #         img = img.unsqueeze(0)
    #         x = img
    #         output = traced_script_module(x)
    #
    #         pred_label = F.interpolate(output, src_patch_1.shape[0:2], mode='nearest')
    #         pred_label = torch.argmax(pred_label, 1)
    #         pred_label_src = pred_label.squeeze(0).cpu().numpy()
    #
    #         pred_label = (255 * pred_label_src).astype(np.uint8)
    #         kernel = np.ones((3, 3), np.uint8)
    #         pred_label = cv2.morphologyEx(pred_label, cv2.MORPH_CLOSE, kernel, iterations=2)
    #
    #         contours, hierarchy = cv2.findContours(pred_label, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #
    #         area = -np.inf
    #         id = 0
    #         for i in range(len(contours)):
    #             area_tmp = cv2.contourArea(contours[i])
    #             if area_tmp > area:
    #                 area = area_tmp
    #                 id = i
    #         cv2.drawContours(src_patch_1, contours, id, (255, 0, 0), 1, 8)
    #         # src_patch_1[pred_label_src == 255] = (255, 0, 0)
    #         dst[y_list[idx_y]:y_list[idx_y + 1], x_list[idx_x]:x_list[idx_x + 1], :] = src_patch_1
    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)

