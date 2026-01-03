# flake8: noqa
import os.path as osp
import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../BasicSR-master')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.path.append(r'/root/autodl-fs/vcolor/BasicSR')
sys.path.append(r'/root/autodl-fs/vcolor/Real-ESRGAN')


from basicsr.train import train_pipeline


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


# return {"lq": lq_tensor, "gt": gt_tensor,'ref':ref_tensor,"lqmask": lqmask_tensor, 'refmask':refmask_tensor}


def save_concatenated_dataset(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
        lq = data["lq"]          # [C, H, W]
        gt = data["gt"]


        # 拼接图像（按宽度方向）
        concat_img = cat([lq, gt], dim=2)  # dim=2 是宽度方向

        save_path = os.path.join(output_dir, f"{idx:05d}_concat.jpg")
        save_image(concat_img, save_path)

# # 使用示例（你已有 train_set）：
# save_concatenated_dataset(train_set, "./concat_saved")


# 使用示例：
# 假设你已经创建了 CCDToneAugWithDRefDegradeDataset 实例：
# train_set = CCDToneAugWithDRefDegradeDataset(opt)





if __name__ == '__main__':
    
    # print("已注册的数据集名称：", DATASET_REGISTRY._obj_map.keys())

    # exit(0)
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    
    opt, args = parse_options(root_path, is_train=True)
    
    dataset_opt=opt['datasets']['train']
    
    train_set = build_dataset(dataset_opt)

    

    # for phase, dataset_opt in opt['datasets'].items():
    #     if phase == 'train':

    # for data in     train_set:
    #     print('data:',data)

    save_concatenated_dataset(train_set, "./saved_dataset")
    
    # train_pipeline(root_path)
    
    
    
