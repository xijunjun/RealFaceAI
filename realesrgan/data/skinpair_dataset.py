# === 标准库 ===
import os
import os.path as osp
import io
import math
import random
import time
import hashlib

# === 第三方库 ===
import cv2
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFile
from pillow_lut import load_cube_file
# import albumentations as A
from accelerate.logging import get_logger

# === 项目内模块 ===
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY



# 取消最大像素限制
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 支持自动旋转
def imread_unicode(path):
    """
    使用 PIL 读取图片，自动根据 EXIF 旋转，并返回 OpenCV 格式的 BGR 图像。
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # ✅ 自动旋转
    img = img.convert("RGB")            # 确保是3通道
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_cv2



def read_index_list(txt_path):
    if txt_path is None:
        return None
    with open(txt_path, 'r') as f:
        indices = [int(line.strip()) for line in f if line.strip()]
    return indices


def get_all_files(root_dir, extensions=(".png", ".jpg", ".jpeg"),indtxt=None):
    files = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                files.append(os.path.join(root, filename))
    files=sorted(files)
    
    
    print('indtxt:',indtxt)
    goodindlist=read_index_list(indtxt)
    if goodindlist is None:
        return files
    
    files_filterd=[files[i] for i in goodindlist]
    # selected_files = [files[i] for i in goodindlist]
    
    return files_filterd


def read_txt_file(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@DATASET_REGISTRY.register()
class SkinPairDataset(data.Dataset):
    def __init__(self,opt, transform=None):
        
        self.opt = opt
        lq_dir=  self.opt['lq_dir']
        gt_dir= self.opt['gt_dir']
        
        self.whiteindtxt=self.opt['whiteindtxt']
        
        self.lq_images = self.load_source(lq_dir, (".png", ".jpg", ".jpeg"),self.whiteindtxt)
        self.gt_images = self.load_source(gt_dir, (".png", ".jpg", ".jpeg"),self.whiteindtxt)
        # self.mask_images = self.load_source(mask_dir, (".png", ".jpg", ".jpeg"),self.whiteindtxt)
        

        self.lq_images.sort()
        self.gt_images.sort()
        # self.mask_images.sort()

        self.transform = transform if transform else transforms.ToTensor()
        
    def load_source(self, dir_list, suffixes=(".png", ".jpg", ".jpeg"),whiteindtxt=None):
        if isinstance(dir_list, str):
            dir_list = [dir_list]

        if whiteindtxt is None:
            whiteindtxt=[  None for cudir in dir_list]
        
        if isinstance(whiteindtxt, str):
            whiteindtxt=[whiteindtxt]

        image_paths = []
        for i,directory in enumerate(dir_list):
            # for suffix in suffixes:
            curlist=get_all_files(directory, suffixes,whiteindtxt[i])
            print('>>>>>>>>>>>>>>>>>>>>>>>>>  curlist:',directory,':',len(curlist))
            image_paths.extend(curlist)
        # exit(0)
        return image_paths
    
    # def load_source(self, source, extensions):
    #     if os.path.isdir(source):
    #         return get_all_files(source, extensions)
    #     elif os.path.isfile(source) and source.endswith(".txt"):
    #         return read_txt_file(source)
    #     else:
    #         raise ValueError(f"Invalid source: {source}")

    def __len__(self):
        return len(self.lq_images)

    def __getitem__(self, idx):
        lq_path = self.lq_images[idx]
        gt_path = self.gt_images[idx]

        lq_image = cv2.cvtColor(imread_unicode(lq_path), cv2.COLOR_BGR2RGB)
        gt_image = cv2.cvtColor(imread_unicode(gt_path), cv2.COLOR_BGR2RGB)

        # #加入随机crop
        # if random.random()<0.9: 
        #     lq_image,gt_image,mask_image=random_crop_same_for_all([lq_image,gt_image,mask_image], min_scale=0.7, max_scale=1.0)


        # lq_image,gt_image=tuple(random_horizontal_flip_list([lq_image,gt_image], prob=0.5))

        resolution=1024
        # resolution=512
        # resolution=768

        lq_image=cv2.resize(lq_image,(resolution,resolution))
        gt_image=cv2.resize(gt_image,(resolution,resolution))
        # mask_image=cv2.resize(mask_image,(resolution,resolution))
                    
        lq_tensor = self.transform(lq_image)
        gt_tensor = self.transform(gt_image)
        # mask_tensor = self.transform(mask_image)
        
        # mask_tensor = mask_tensor[0].unsqueeze(0)
        
        return {"lq": lq_tensor, "gt": gt_tensor}