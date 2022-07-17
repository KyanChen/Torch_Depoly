import torch
import argparse
import os
from utils import time_file_str, print_log
import torch.backends.cudnn as cudnn
from models.ZYNet import ZYNet
from torchvision import transforms
import time
from PIL import Image


parser = argparse.ArgumentParser(description='PyTorch CFNet Training')
parser.add_argument('--in_path', default='./data/SV2-02-PMS-102-812-L00000018271.jpg',help='path for predict')
parser.add_argument('--nclass', type=int, default=2, help='the number of class')
parser.add_argument('--resume', default=".\\checkpoints\\best.resnet.2022-07-10-4110.pth.tar", type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pattern_height', type=int, default=1024, help='image size when training')


args = parser.parse_args()

device = torch.device("cpu")

args.prefix = time_file_str()

pattern_height = args.pattern_height
pattern_width = args.pattern_height# args.pattern_width


def Generate_onnx():
    model_cfg = dict(
        backbone_cfg=dict(
            name='resnet18',
            num_classes=None,
            in_channels=3,
            pretrained=False,
            out_keys=('block2', 'block3', 'block4', 'block5')),
        neck_cfg=dict(
            in_channels=512,
            out_channels=512, ),
        head_cfg=dict(
            name='MattingNeck',
            in_channels=512,
            img_size=args.pattern_height,
            feat_channels=(512, 256, 128, 64, 64, 32),
            num_classes=args.nclass
        )
    )

    # create model
    model = ZYNet(**model_cfg)
    model=model.to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_pred = checkpoint['best_pred']
            model.load_state_dict(checkpoint['state_dict'])
            # print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at")

    cudnn.benchmark = True

    model.eval()

    print('Model has been loaded successfully!\n')
    input_transform = transforms.Compose([
        # transforms.Resize(size=(512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    if os.path.isdir(args.in_path):
        predict_files = os.listdir(args.in_path)
    if os.path.isfile(args.in_path):
        predict_files=[]
        name=args.in_path.split("/")
        predict_files.append(name[-1])

    for item in predict_files:
        image_type = '.jpg'
        if item.endswith(image_type):
            print("item:", item)
            print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            if os.path.isdir(args.in_path):
                image = Image.open(os.path.join(args.in_path, item)).convert('RGB')
            elif os.path.isfile(args.in_path):
                image = Image.open(os.path.join(args.in_path)).convert('RGB')

            # Seg
            tensor_in = input_transform(image).unsqueeze(0).to(device) * 255
            _, depth, height, width = tensor_in.shape
            # #
            img_transform = transforms.Resize((pattern_height, pattern_width), interpolation=Image.Resampling.NEAREST)
            tensor_in = img_transform(tensor_in)

            with torch.no_grad():
                input_names = ["img"]
                output_names = ["mask"]
                torch.onnx.export(model, tensor_in, "onnx_model.onnx", verbose=True,
                                  input_names=input_names,
                                  output_names=output_names,
                                  opset_version=12)


    print('Generate Onnx_model successfully!\n')


if __name__ == '__main__':
    Generate_onnx()
