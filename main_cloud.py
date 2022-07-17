import warnings
warnings.filterwarnings("ignore")
import argparse
from numpy import dstack, array
import os
# from utils import time_file_str
# import time
import onnxruntime
from PIL import Image
import numpy as np
from skimage import measure


parser = argparse.ArgumentParser(description='PyTorch CFNet Training')
parser.add_argument('--in_path', default='./SV2-02-PMS-102-812-L00000018271.jpg',help='path for predict')
parser.add_argument('--out_path', default='./data',help='path for output mask') #02-952-7-1
parser.add_argument('--nclass', type=int, default=2, help='the number of class')
parser.add_argument('--pattern_height', type=int, default=1024, help='image size when training')

parser.add_argument('--img_method', default='crop_resize',help='crop_resize')
parser.add_argument('--savemask', default=True,help='Save mask or not')

args = parser.parse_args()
# args.use_cuda = torch.cuda.is_available()
# device = torch.device("cpu")

# args.prefix = time_file_str()

pattern_height = args.pattern_height
pattern_width = args.pattern_height# args.pattern_width

color_map = array([[0, 0, 0],
                   [128, 128, 128],
                   # [0, 0, 128],
                   ])

categories_map = {0: 'background', 1: 'cloud',
                  # 2: 'snow'
                  }

class Evaluate():
    def __init__(self,mask):
        self.mask = mask

        self.h = self.mask.shape[0]
        self.w = self.mask.shape[1]
        self.area_all = self.h * self.w
        h_half = self.h//2
        w_half = self.w//2
        self.mask_1 = self.mask[0:h_half, 0:w_half]
        self.mask_2 = self.mask[0:h_half, w_half:]
        self.mask_3 = self.mask[h_half:, w_half:]
        self.mask_4 = self.mask[h_half:, 0:w_half]

        # plt.imshow(self.mask_3, cmap = plt.cm.jet)
        # plt.show()



    def cal_area(self):
        self.area_1 = self.mask_1.sum()/self.area_all * 400
        self.area_2 = self.mask_2.sum()/self.area_all * 400
        self.area_3 = self.mask_3.sum()/self.area_all * 400
        self.area_4 = self.mask_4.sum()/self.area_all * 400
        self.area = (self.area_1+self.area_2+self.area_3+self.area_4)/4
        return self.area,self.area_1,self.area_2,self.area_3,self.area_4

    def cal_frag(self,submask,subarea):
        edge = measure.perimeter(submask)/(2*self.h)
        extent = edge /(subarea+1e-5)
        # props = measure.regionprops(submask)
        # r_n = len(props)
        # if not props:
        #     extent = 0
        # else:
        #     var = []
        #     for prop in props:
        #         var.append((prop.area/(self.h*self.w*0.25) -subarea/r_n)**2)
        #     var = np.array(var).sum()
        #     extent = r_n*subarea/var
        return extent


    def cal_frag_all(self):
        self.extent_1 = self.cal_frag(self.mask_1,self.area_1)
        self.extent_2 = self.cal_frag(self.mask_2,self.area_2)
        self.extent_3 = self.cal_frag(self.mask_3,self.area_3)
        self.extent_4 = self.cal_frag(self.mask_4,self.area_4)
        self.extent = (self.extent_1+self.extent_2+self.extent_3+self.extent_4)/4
        return self.extent
    def evaluate(self):
        area, area_1, area_2, area_3, area_4 = self.cal_area()
        extent_img = 9 - min(self.cal_frag_all() / 0.94, 1) * 9  # 破碎度
        if area > 95:
            quality = 0
        elif area > 80:
            quality = 1
        elif area > 60:
            quality = 2
        elif area > 40:
            quality = 3
        elif area > 10:
            if extent_img < 5:
                quality = 4
            elif extent_img < 8:
                quality = 5
            else:
                quality = 6
        elif area > 5:
            quality = 7
        elif area < 1:
            quality = 9
        else:
            quality = 8
        return quality,area,extent_img,area_1, area_2,area_3,area_4

def save_image(nclass, array, path):
    r = array.copy()
    g = array.copy()
    b = array.copy()

    for i in range(nclass):
        r[array == i] = color_map[i][0]
        g[array == i] = color_map[i][1]
        b[array == i] = color_map[i][2]

    rgb = dstack((r, g, b))

    save_img = Image.fromarray(rgb.astype('uint8'))
    save_img.save(path)


def predict():
    # strat_time = time.time()
    session = onnxruntime.InferenceSession("onnx_model.onnx")

    # if not os.path.exists(args.out_path):
    #     os.mkdir(args.out_path)

    if os.path.isdir(args.in_path):
        args.out_path = args.in_path
        predict_files = os.listdir(args.in_path)

    if os.path.isfile(args.in_path):
        args.out_path = os.path.dirname(args.in_path)

        predict_files=[]
        # name=args.in_path.split("/")
        name = os.path.basename(args.in_path)
        predict_files.append(name)
    # file = open(args.out_path + "/" + 'Evaluate.txt', 'w+')
    # file.close()
    for item in predict_files:
        image_type = '.jpg'
        if item.endswith(image_type):
            # print("Load image successfully:", item)

            # print("Begin Time:", time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            if os.path.isdir(args.in_path):
                image = Image.open(os.path.join(args.in_path, item)).convert('RGB')
            elif os.path.isfile(args.in_path):
                image = Image.open(os.path.join(args.in_path)).convert('RGB')

            if args.img_method == "crop_resize":
                height = image.height
                width = image.width
                if (0.5<height/width<2):
                    # print("img_method: just resize")

                    image = image.resize((pattern_width, pattern_height), resample=Image.NEAREST)#Image.Resampling.NEAREST
                    tensor_in = np.expand_dims(np.array(image), axis=0)
                    tensor_in = np.transpose(tensor_in,(0,3, 1, 2)).astype('float32') #(1,3,1024,1024)

                    # model process
                    # t = time.time()
                    input_name = session.get_inputs()[0].name
                    # print("input name", input_name)
                    output_name = session.get_outputs()[0].name
                    # print("output name", output_name)
                    # forward model
                    res = session.run([output_name], {input_name: tensor_in})
                    pred_seg = np.array(res[0])
                    # print('Inference Time: ', time.time() - t)

                    # 输出预测结果
                    # pred_seg = output_seg#.data.cpu().numpy()
                    pred_seg = np.argmax(pred_seg, axis=1) #(1,1024,1024)
                    out = Image.fromarray(pred_seg[0].astype('uint8')).convert('L')
                    out = out.resize((width,height), resample=Image.NEAREST)
                    out = np.array(out) #(h,w)

                else:
                    # print("img_method: crop and resize")

                    out = (np.array(image) * 0)[:,:,0] # np.zeros(height,width) #image是（w,h），而经过numpy后成为out是（h,w）

                    if height > width:
                        patch_height = width
                        patch_width = width
                    else:
                        patch_height = height
                        patch_width = height

                    height_cut_num = height // patch_height
                    width_cut_num = width // patch_width
                    if (height % patch_height) != 0:
                        height_cut_num = height_cut_num + 1
                    if (width % patch_width) != 0:
                        width_cut_num = width_cut_num + 1
                    for j in range(height_cut_num):
                        for i in range(width_cut_num):
                            left_h = j * patch_height
                            left_w = i * patch_width
                            right_h = left_h + patch_height
                            right_w = left_w + patch_width
                            if right_h > height:
                                left_h = left_h - (right_h - height)
                                right_h = right_h - (right_h - height)
                            if right_w > width:
                                left_w = left_w - (right_w - width)
                                right_w = right_w - (right_w - width)
                            image_crop = image.crop((left_w,left_h,right_w,right_h))
                            image_crop = image_crop.resize((pattern_height, pattern_width),
                                                              resample=Image.NEAREST)
                            tensor_in_split = np.expand_dims(np.array(image_crop), axis=0)
                            tensor_in_split = np.transpose(tensor_in_split, (0, 3, 1, 2)).astype('float32')

                            # t = time.time()
                            input_name = session.get_inputs()[0].name
                            # print("input name", input_name)
                            output_name = session.get_outputs()[0].name
                            # print("output name", output_name)
                            # forward model
                            res = session.run([output_name], {input_name: tensor_in_split})
                            pred_seg = np.array(res[0])
                            # print('Inference Time: ', time.time() - t)

                            pred_seg = np.argmax(pred_seg, axis=1)
                            out_crop = Image.fromarray(pred_seg[0].astype('uint8')).convert('L')
                            out_crop = out_crop.resize((patch_height, patch_width), resample=Image.Resampling.NEAREST)
                            out_crop = np.array(out_crop)

                            out[left_h: right_h, left_w: right_w] = out_crop
                            # out = out

            cal = Evaluate(out)
            quality, area, extent_img, area_1, area_2, area_3, area_4 = cal.evaluate()
            # print('Process successfully!')
            # print('Total Processing Time: ', time.time() - strat_time)
            if args.savemask==True:
                save_image(args.nclass, out, os.path.join(args.out_path,item[:-4]+"_cloudmask.png") )
                # print('Mask has been saved, save dir:', args.out_path)

            # print('=================Out==================')
            # print('Quality |','Cloud Cover (%) |','Fragmentation |','Cloud Cover of sub_I~IV(%)')
            out_string = 'output:'+'\t'+str(quality)+ '\t'+ str(int(area))+ '\t'+ str(int(extent_img))+ '\t'+ str(int(area_1))+ '\t'+ str(int(area_2))+ '\t'+ str(int(area_3))+ '\t'+ str(int(area_4))
            print(out_string)
            # print('==================================================')

            #file = open(args.out_path + "/" + 'Evaluate.txt', 'a+')
            #file.write('File name:'+ item+":  ")
            #file.write(str(quality)+ '\t'+ str(int(area))+ '\t'+ str(int(extent_img))+ '\t'+ str(int(area_1))+ '\t'+ str(int(area_2))+ '\t'+
            #      str(int(area_3))+ '\t'+ str(int(area_4))+"\n")
    #file.close()
    # return out_string
    

if __name__ == '__main__':
    predict()
