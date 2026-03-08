import cv2
import math
import numpy as np
import random
from scipy import special

# -------------------------------------------------------------------- #
# --------------------------- 1. 核心模糊核算子 ------------------------- #
# -------------------------------------------------------------------- #

def sigma_matrix2(sig_x, sig_y, theta):
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

def mesh_grid(kernel_size):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size, 1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy

def pdf2(sigma_matrix, grid):
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel

def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    if grid is None: grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    return kernel / np.sum(kernel)

def bivariate_generalized_Gaussian(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    if grid is None: grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    return kernel / np.sum(kernel)

def bivariate_plateau(kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True):
    if grid is None: grid, _, _ = mesh_grid(kernel_size)
    sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]]) if isotropic else sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.reciprocal(np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    return kernel / np.sum(kernel)

def get_random_kernel(opt, phase):
    """根据该阶段独立的 kernel 配置生成随机核"""
    k_size = opt[f'kernel_size{phase}']
    k_list = opt[f'kernel_list{phase}']
    k_prob = opt[f'kernel_prob{phase}']
    sig_range = opt[f'blur_sigma{phase}']
    
    k_type = random.choices(k_list, k_prob)[0]
    sig_x = np.random.uniform(sig_range[0], sig_range[1])
    sig_y = np.random.uniform(sig_range[0], sig_range[1])
    rot = np.random.uniform(-math.pi, math.pi)
    
    if k_type == 'iso': return bivariate_Gaussian(k_size, sig_x, sig_x, 0, isotropic=True)
    if k_type == 'aniso': return bivariate_Gaussian(k_size, sig_x, sig_y, rot, isotropic=False)
    if 'generalized' in k_type:
        beta = np.random.uniform(opt[f'betag_range{phase}'][0], opt[f'betag_range{phase}'][1])
        return bivariate_generalized_Gaussian(k_size, sig_x, sig_y, rot, beta, isotropic=('iso' in k_type))
    if 'plateau' in k_type:
        beta = np.random.uniform(opt[f'betap_range{phase}'][0], opt[f'betap_range{phase}'][1])
        return bivariate_plateau(k_size, sig_x, sig_y, rot, beta, isotropic=('iso' in k_type))
    return np.ones((k_size, k_size)) / (k_size**2)


# --------------------------- 1. 核心模糊核算子 --------------------------- #
# (保持你提供的底层数学函数不变，此处略以节省篇幅，实际使用请保留完整函数)
# 包括 sigma_matrix2, mesh_grid, pdf2, bivariate_Gaussian, 
# bivariate_generalized_Gaussian, bivariate_plateau

def get_random_kernel(opt, phase):
    k_size = opt[f'kernel_size{phase}']
    k_list = opt[f'kernel_list{phase}']
    k_prob = opt[f'kernel_prob{phase}']
    sig_range = opt[f'blur_sigma{phase}']
    
    k_type = random.choices(k_list, k_prob)[0]
    sig_x = np.random.uniform(sig_range[0], sig_range[1])
    sig_y = np.random.uniform(sig_range[0], sig_range[1])
    rot = np.random.uniform(-math.pi, math.pi)
    
    if k_type == 'iso': return bivariate_Gaussian(k_size, sig_x, sig_x, 0, isotropic=True)
    if k_type == 'aniso': return bivariate_Gaussian(k_size, sig_x, sig_y, rot, isotropic=False)
    if 'generalized' in k_type:
        beta = np.random.uniform(opt[f'betag_range{phase}'][0], opt[f'betag_range{phase}'][1])
        return bivariate_generalized_Gaussian(k_size, sig_x, sig_y, rot, beta, isotropic=('iso' in k_type))
    if 'plateau' in k_type:
        beta = np.random.uniform(opt[f'betap_range{phase}'][0], opt[f'betap_range{phase}'][1])
        return bivariate_plateau(k_size, sig_x, sig_y, rot, beta, isotropic=('iso' in k_type))
    return np.ones((k_size, k_size)) / (k_size**2)

# --------------------------- 2. 退化流水线类 --------------------------- #

class DegradationPipeline:
    def __init__(self, opt):
        self.opt = opt
        self.op_map = {
            'blur': self._apply_blur,
            'resize': self._apply_resize,
            'noise': self._apply_noise,
            'jpeg': self._apply_jpeg
        }

    def _apply_blur(self, img, phase):
        if np.random.uniform() > self.opt.get(f'blur_prob{phase}', 1.0): return img
        kernel = get_random_kernel(self.opt, phase)
        return cv2.filter2D(img, -1, kernel)

    def _apply_resize(self, img, phase):
        h, w = img.shape[:2]
        if phase == '2' and self.opt.get('resize_to_final', True):
            target_h, target_w = self.orig_h // self.opt['scale'], self.orig_w // self.opt['scale']
        else:
            scale = np.random.uniform(self.opt[f'resize_range{phase}'][0], self.opt[f'resize_range{phase}'][1])
            target_h, target_w = int(h * scale), int(w * scale)
        
        mode = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        return cv2.resize(img, (target_w, target_h), interpolation=mode)

    def _apply_noise(self, img, phase):
        if np.random.uniform() > self.opt.get(f'noise_prob{phase}', 0.5): return img
        gray_p = self.opt.get(f'gray_noise_prob{phase}', 0.4)
        
        if np.random.uniform() < self.opt.get(f'poisson_prob{phase}', 0.5):
            scale = np.random.uniform(self.opt[f'poisson_scale{phase}'][0], self.opt[f'poisson_scale{phase}'][1])
            # 优化：对于 4K 图像，取唯一值的计算较慢，可改用简化 poisson
            noise = np.random.poisson(np.maximum(0, img) * 255.0) / 255.0 - img
            noise *= scale
        else:
            sigma = np.random.uniform(self.opt[f'gaussian_sigma{phase}'][0], self.opt[f'gaussian_sigma{phase}'][1])
            noise = np.random.randn(*img.shape).astype(np.float32) * (sigma / 255.)
            
        if np.random.uniform() < gray_p:
            noise = np.repeat(np.mean(noise, axis=-1, keepdims=True), img.shape[-1], axis=-1)
        return np.clip(img + noise, 0, 1)

    def _apply_jpeg(self, img, phase):
        q_range = self.opt[f'jpeg_range{phase}']
        quality = np.random.uniform(q_range[0], q_range[1])
        _, enc = cv2.imencode('.jpg', (img * 255.).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return cv2.imdecode(enc, 1).astype(np.float32) / 255.

    def process(self, img):
        """
        img: 输入图像，支持 uint8 或 float32 [0, 1]
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
            
        self.orig_h, self.orig_w = img.shape[:2]
        out = img.copy()

        # 处理第一阶段
        p1_seq = list(self.opt['phase1_sequence'])
        if self.opt.get('shuffle_phase1', False):
            random.shuffle(p1_seq)
        for op_name in p1_seq:
            out = self.op_map[op_name](out, '1')

        # 处理第二阶段
        p2_seq = list(self.opt['phase2_sequence'])
        if self.opt.get('shuffle_phase2', False):
            random.shuffle(p2_seq)
        for op_name in p2_seq:
            out = self.op_map[op_name](out, '2')
            
        return np.clip(out, 0, 1)

# --------------------------- 3. 训练集成配置 --------------------------- #

train_config = {
    'scale': 1,
    'shuffle_phase1': True,  # 训练时开启随机顺序
    'shuffle_phase2': True,
    
    # Phase 1
    'phase1_sequence': ['blur', 'resize', 'noise', 'jpeg'], 
    'kernel_size1': 31,
    'kernel_list1': ['iso', 'aniso', 'generalized_iso', 'plateau_iso'],
    'kernel_prob1': [0.4, 0.3, 0.2, 0.1],
    'blur_sigma1': (0.2, 3.0),
    'betag_range1': (0.5, 4.0), 'betap_range1': (1, 2.0),
    'resize_range1': (0.15, 0.9),
    'noise_prob1': 0.5, 'gaussian_sigma1': (1, 30), 'poisson_scale1': (0.05, 3.0),
    'gray_noise_prob1': 0.4, 'jpeg_range1': (30, 95),

    # Phase 2
    'phase2_sequence': ['noise', 'blur', 'resize', 'jpeg'],
    'kernel_size2': 31,
    'kernel_list2': ['iso', 'aniso'],
    'kernel_prob2': [0.5, 0.5],
    'blur_sigma2': (0.2, 1.5),
    'betag_range2': (0.5, 4.0), 'betap_range2': (1, 2.0),
    'resize_to_final': True, 
    'noise_prob2': 0.5, 'gaussian_sigma2': (1, 20), 'poisson_scale2': (0.05, 2.5),
    'gray_noise_prob2': 0.4, 'jpeg_range2': (40, 90)
}

# # --------------------------- 使用示例 --------------------------- #

# if __name__ == '__main__':
#     # 读取 4K 图像
#     hr_img = cv2.imread('DSC09982fix_0.jpg')
#     if hr_img is None:
#         hr_img = (np.random.rand(2160, 3840, 3) * 255).astype(np.uint8)

#     # 实例化
#     pipeline = DegradationPipeline(train_config)
    
#     # 获取退化后的 LQ 图像 (自动完成类型转换和顺序随机化)
#     lq_img_f32 = pipeline.process(hr_img)
    
#     # 转回 uint8 方便保存或可视化
#     lq_img_ui8 = (lq_img_f32 * 255.0).round().astype(np.uint8)
    
#     cv2.imwrite('lq_output.jpg', lq_img_ui8)
#     print(f"退化完成: {hr_img.shape} -> {lq_img_ui8.shape}")