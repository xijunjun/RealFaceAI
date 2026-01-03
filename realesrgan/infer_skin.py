import os
import cv2
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
import re
from pathlib import Path

# flake8: noqa
import os.path as osp
import sys
sys.path.append(r'/root/autodl-fs/vcolor/BasicSR')
from basicsr.train import train_pipeline
sys.path.append('/root/autodl-fs/vcolor/Real-ESRGAN')
import realesrgan.archs
import realesrgan.data
import realesrgan.models
from basicsr.data import build_dataloader, build_dataset
from basicsr.utils.options import copy_opt_file, dict2str, parse_options
# from basicsr.data import DATASET_REGISTRY
import os
from torchvision.utils import save_image
from torch import cat
from tqdm import tqdm

from basicsr.archs import build_network
from utils_yaml import *



from pathlib import Path
import os
import torch
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import os
import random
from tqdm import tqdm
# from util.utils import load_pretrained
import shutil
# from util.utils import load_pretrained,load_pretrained_simple
from argparse import Namespace
Image.MAX_IMAGE_PIXELS = None

import cv2
# from  func_small2full  import  process_result_small2full




def extract_key(pth_path: str) -> str:
    path = Path(pth_path)
    match = re.search(r'experiments/([^/]+)/models/net_g_(\d+)\.pth', pth_path)
    if match:
        exp_name = match.group(1)  # e.g., train_flash_percep_rcropwd
        step = match.group(2)      # e.g., 120000
        return f"{exp_name}_{step}"
    else:
        raise ValueError("Path format does not match expected pattern.")


def extract_two_items_before_checkpoint(path_str):
    """
    提取路径中 checkpoint-xxx 前面的两个目录名
    """
    path = Path(path_str)
    parts = path.parts
    
    # 找到 checkpoint-xxx 的位置
    checkpoint_index = next(i for i, part in enumerate(parts) if part.startswith("checkpoint-"))
    
    if checkpoint_index < 2:
        raise ValueError("路径中 checkpoint 前没有足够的上层目录")
    
    return parts[checkpoint_index - 2]+'_'+ parts[checkpoint_index - 1]+'_'+ parts[checkpoint_index ]



def cv2_to_pil(cv2_img):
    """将 cv2 格式的图像（BGR）转换为 PIL 图像（RGB）"""
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)


def pil_to_cv2(pil_img):
    """将 PIL 图像（RGB）转换为 cv2 格式图像（BGR）"""
    rgb_array = np.array(pil_img)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)




def select_random_images(root_dir, num_images, save_dir=None):
    images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                images.append(os.path.join(dirpath, filename))

    if len(images) < num_images:
        print("Warning: Number of images in folder is less than required.")

    selected_images = random.sample(images, min(num_images, len(images)))
    
    if save_dir is not None:
        # Ensure the destination directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Copy selected images to the destination directory
        for image in selected_images:
            shutil.copy(image, save_dir)
    return selected_images


def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform



def makedir(tdir):
    if os.path.exists(tdir) ==False:
        os.makedirs(tdir)

def get_dirims(imgpath,sufs=['.jpg', '.jpeg', '.png']):
    imgpathlst = []
    for filename in os.listdir(imgpath):
        if os.path.splitext(filename)[1] in sufs :
            imgpathlst.append(os.path.join(imgpath, filename))
    return imgpathlst

def extract_key(pth_path: str) -> str:
    path = Path(pth_path)
    match = re.search(r'experiments/([^/]+)/models/net_g_(\d+)\.pth', pth_path)
    if match:
        exp_name = match.group(1)  # e.g., train_flash_percep_rcropwd
        step = match.group(2)      # e.g., 120000
        return f"{exp_name}_{step}"
    else:
        raise ValueError("Path format does not match expected pattern.")



def parse_args():
    parser = argparse.ArgumentParser(description='FLUX image generation with LoRA')
    parser.add_argument('--model_path', type=str, 
                        default="/data/vjuicefs_ai_camera_pgroup_ql/public_data/HuggingFaceModels/models--black-forest-labs--FLUX.1-dev",
                        help='Path to pretrained model')
    parser.add_argument('--image_path', type=str,
                        default="assets/our_test.png",
                        help='Input image path')
    
    parser.add_argument('--ref_path', type=str,
                        default="assets/our_test.png",
                        help='Input image path')
    
    parser.add_argument('--gt_path', type=str,
                        default="assets/our_test.png",
                        help='Input image path')



def sorted_image_list(path):
    return sorted([
        os.path.join(path, f) for f in os.listdir(path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])


def letterbox_pad(img, target_size):
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    return padded


def resize_height_limit_1080(image):
    h, w = image.shape[:2]
    if h <= 1080:
        return image  # 不缩放
    scale = 1080 / h
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, 1080), interpolation=cv2.INTER_AREA)
    return resized


transform= transforms.Compose(
            [
                # transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ]
        )



def get_imkey_ext(impath):
    imname=os.path.basename(impath)
    items=imname.split('.')
    ext='.'+items[-1]
    # imkey = imname.replace(ext,'')
    imkey=imname[0:-len(ext)]
    # ext='.'+ext
    return imkey,ext

def load_mamba_model(model_path,opt):

    model = build_network(opt['network_g'])
    print('''>>>>>>>>>>>>>>>load  network  opt['network_g']''',opt['network_g'])
    # state_dict = torch.load(model_path, map_location="cuda")  # 加载权重
    state_dict = torch.load(model_path) #, map_location="cuda")  # 加载权重

    # print("Keys in state_dict:")
    for key in state_dict.keys():
        print(key)
    print('>>>>>>>>>>>>>>>>>>')

    # if "module" in state_dict:
    state_dict = state_dict["params"] 
    model.load_state_dict(state_dict)  # 确保正确加载权重
    

    # state_dict={}
    # state_dict["params"]=model.state_dict()
    # torch.save(state_dict, '/data/vjuicefs_ai_camera_pgroup_ql/11103464/workspace_ql/train_proj_ql/colorgc/RealMambaST/experiments/mambasave/models/net_g_0.pth')
    # exit(0)
    
    # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # if missing_keys:
    #     print("[WARNING] Missing keys:", missing_keys)
    # if unexpected_keys:
    #     print("[WARNING] Unexpected keys:", unexpected_keys)
    # if not missing_keys and not unexpected_keys:
    #     print("[INFO] Model weights loaded successfully.")
    
    
    
    
    
    
    model.to(device="cuda")  # 移动到 GPU
    model.eval()  # 设置为推理模式
    return model



def eval(args,opt):
    
    
    image_path=args.image_path
    # ref_path=args.ref_path
    # gt_path=args.gt_path
    
    
    
    control_list = sorted_image_list(image_path)
    # ref_list = sorted_image_list(ref_path)
    # gt_list = sorted_image_list(gt_path)
    

    device='cuda:0'
    
    
    
    modelkey=extract_key(args.model_path) 
    
    output_dir=image_path+'_result_mambaresr_'+modelkey
    makedir(output_dir)
    
    
    
    network=load_mamba_model(args.model_path,opt)

    for i,control_path in enumerate(control_list):
        
        
        
        # if i<30:
        #     continue
            
            
        print('{}of{}'.format(i,len(control_list)))
        
        # process_images(im_input, masks_input[i],ims_ref[i],model, output_dir, resize_size)
    
        # ref_imagecv2=cv2.imread(ref_list[i])
        # gt_imagecv2=cv2.imread(gt_list[i])
        
        control_image_ori=cv2.imread(control_path)
        control_image_ori_1024=cv2.resize(control_image_ori,(1024,1024))


        condition_image=transform(cv2_to_pil(cv2.resize(control_image_ori,(1024,1024)))).unsqueeze(0).cuda()
        # ref_image=transform(cv2_to_pil(cv2.resize(ref_imagecv2,(1024,1024)))).unsqueeze(0).cuda()
        

        result= network(condition_image)
        result = (result.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
        
        
        
        resultfull=pil_to_cv2(result)

        imkey,ext=get_imkey_ext(control_path)
        
        resultfull=np.clip(resultfull,0,255).astype(np.uint8)
        # cv2.imwrite(args.output_path,resultfull)
        h, w = control_image_ori.shape[:2]

        dst_path_lq=os.path.join(output_dir,imkey+'_0lq'+ext)
        dst_path_gt=os.path.join(output_dir,imkey+'_1pred'+ext)
    
        cv2.imwrite(dst_path_lq,control_image_ori_1024)
        cv2.imwrite(dst_path_gt,resultfull)
    
    
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="IRStyle Inference Script")
    parser.add_argument("--image_path", required=True, help="Directory containing content images")
    # parser.add_argument("--ref_path", required=True, help="Directory containing content images")
    # parser.add_argument("--gt_path", required=True, help="Directory containing ref images")
    # parser.add_argument("--output_dir", required=True, help="Directory to save output images")
    parser.add_argument("--model_path", required=True, help="Path to IRStyle model checkpoint")
    parser.add_argument("--resize", type=str, default="256,256", help="Resize images to fixed size (format: width,height)")
    parser.add_argument("--trainopt", type=str, default="", help="Resize images to fixed size (format: width,height)")
    

    args = parser.parse_args()

    
    
    opt=yaml_load(args.trainopt)
    
    
    eval(args,opt)
    
    
    
