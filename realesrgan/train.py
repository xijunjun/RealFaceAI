# flake8: noqa
import os.path as osp
import sys

import os
# sys.path.append(r'./BasicSR-master')
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../BasicSR')))
# exit(0)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../BasicSR')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from basicsr.train import train_pipeline
import realesrgan.archs
import realesrgan.data
import realesrgan.models
# from basicsr.data import DATASET_REGISTRY

if __name__ == '__main__':

    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)



# # flake8: noqa
# import os.path as osp
# from basicsr.train import train_pipeline

# import realesrgan.archs
# import realesrgan.data
# import realesrgan.models

# if __name__ == '__main__':
#     root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
#     train_pipeline(root_path)
