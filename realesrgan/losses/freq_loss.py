import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import LOSS_REGISTRY
# version adaptation for PyTorch > 1.7.1
# IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) > (1, 7)
if IS_HIGH_VERSION:
    import torch.fft

def calc_fft(image):
    '''image is tensor, N*C*H*W'''
    if IS_HIGH_VERSION:
        # freq = torch.fft.fft2(image, norm='ortho')
        freq = torch.fft.fft2(image)
        freq = torch.stack([freq.real, freq.imag], -1)
    else:
        # freq = torch.rfft(image, 2, onesided=False, normalized=True)
        freq = torch.rfft(image, 2, onesided=False)
    fft_mag = torch.log(1 + torch.sqrt(freq[..., 0] ** 2 + freq[..., 1] ** 2 + 1e-8))
    return fft_mag


def fft_L1_loss(fake_image, real_image):
    criterion_L1 = torch.nn.L1Loss()

    fake_image_gray = fake_image[:,0]*0.299 + fake_image[:,1]*0.587 + fake_image[:,2]*0.114
    real_image_gray = real_image[:,0]*0.299 + real_image[:,1]*0.587 + real_image[:,2]*0.114

    fake_fft = calc_fft(fake_image_gray)
    real_fft = calc_fft(real_image_gray)
    loss = criterion_L1(fake_fft, real_fft)
    return loss


def fft_L1_loss_mask(fake_image, real_image, mask):
    criterion_L1 = torch.nn.L1Loss()

    fake_image_gray = fake_image[:, 0] * 0.299 + fake_image[:, 1] * 0.587 + fake_image[:, 2] * 0.114
    real_image_gray = real_image[:, 0] * 0.299 + real_image[:, 1] * 0.587 + real_image[:, 2] * 0.114

    fake_fft = calc_fft(fake_image_gray)
    real_fft = calc_fft(real_image_gray)
    loss = criterion_L1(fake_fft * mask, real_fft * mask)
    return loss


def fft_L1_loss_color(fake_image, real_image):
    criterion_L1 = torch.nn.L1Loss()

    fake_fft = calc_fft(fake_image)
    real_fft = calc_fft(real_image)
    loss = criterion_L1(fake_fft, real_fft)
    return loss

def fft_L1_loss_color_mask(fake_image, real_image, mask):
    criterion_L1 = torch.nn.L1Loss()

    fake_fft = calc_fft(fake_image)
    real_fft = calc_fft(real_image)
    loss = criterion_L1(fake_fft * mask, real_fft * mask)
    return loss

def decide_circle(N=4, L=256, r=96, size=256):
    x = torch.ones((N, L, L))
    for i in range(L):
        for j in range(L):
            if (i- L/2 + 0.5)**2 + (j- L/2 + 0.5)**2 < r **2:
                x[:,i,j]=0
    return x, torch.ones((N, L, L)) - x


def get_gaussian_blur(x, k, stride=1, padding=0):
    res = []
    x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    for xx in x.split(1, 1):
        res.append(F.conv2d(xx, k, stride=stride, padding=0))
    return torch.cat(res, 1)


def get_low_freq(im, gauss_kernel):
    padding = (gauss_kernel.shape[-1] - 1) // 2
    low_freq = get_gaussian_blur(im, gauss_kernel, padding=padding)
    return low_freq


def gaussian_blur(x, k, stride=1, padding=0):
    res = []
    x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    for xx in x.split(1, 1):
        res.append(F.conv2d(xx, k, stride=stride, padding=0))
    return torch.cat(res, 1)

def get_gaussian_kernel(size=3):
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
    return kernel


def find_fake_freq(im, gauss_kernel, index=None):
    padding = (gauss_kernel.shape[-1] - 1) // 2
    low_freq = gaussian_blur(im, gauss_kernel, padding=padding)
    im_gray = im[:, 0, ...] * 0.299 + im[:, 1, ...] * 0.587 + im[:, 2, ...] * 0.114
    im_gray = im_gray.unsqueeze_(dim=1).repeat(1, 3, 1, 1)
    low_gray = gaussian_blur(im_gray, gauss_kernel, padding=padding)
    return torch.cat((low_freq, im_gray - low_gray),1)

def find_fake_freq_mask(im, gauss_kernel, mask, index=None):
    padding = (gauss_kernel.shape[-1] - 1) // 2
    low_freq = gaussian_blur(im, gauss_kernel, padding=padding)
    im_gray = im[:, 0, ...] * 0.299 + im[:, 1, ...] * 0.587 + im[:, 2, ...] * 0.114
    im_gray = im_gray.unsqueeze_(dim=1).repeat(1, 3, 1, 1)
    low_gray = gaussian_blur(im_gray, gauss_kernel, padding=padding)
    return torch.cat((low_freq * mask, (im_gray - low_gray) * mask),1)


@LOSS_REGISTRY.register()
class FrequencyFeatureLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Frequency Domain Image Translation: More Photo-realistic, Better Identity-preserving. In ICCV 2021.
    <https://arxiv.org/pdf/2011.13611.pdf>
    github: https://github.com/mu-cai/frequency-domain-image-translation
    This is \mathcal{L}_{org} + \lambda_1 \mathcal{L}_{rec, pix} + \lambda_3 \mathcal{L}_{rec, fft} by reducing Translation Loss in Eq(10)

    Args:
        loss_weight (float): weight for overal frequency loss. Default: 1.0
        gauss_size (int): . Default: 21
        radius (int): . Default: 21
        batch_size (int): . Default: 2
        lambda_recon_blur (float): \lambda_1 to weight \mathcal{L}_{rec, pix}. Default: 1.
        lambda_recon_fft (float): \lambda_3 to weight \mathcal{L}_{rec, fft}. Default: 1.

        Default values are referred from https://github.com/mu-cai/frequency-domain-image-translation/blob/master/StarGANv2/main.py
    """
    def __init__(self, loss_weight:float=1.0, reduction:str='mean', gauss_size:int=21, lambda_recon_blur:float=1., lambda_recon_fft:float=1.):
        super(FrequencyFeatureLoss, self).__init__()
        self.gauss_kernel = get_gaussian_kernel(gauss_size).cuda()
        # # radius:int=21, batch_size:int=2,
        # self.mask_h, _ = decide_circle(r=radius, N=batch_size)
        # self.mask_h = self.mask_h.cuda()
        self.lambda_recon_blur = lambda_recon_blur
        self.lambda_recon_fft = lambda_recon_fft
        self.loss_weight = loss_weight
    def forward(self, pred, real):
        """Implementation is referred from https://github.com/mu-cai/frequency-domain-image-translation/blob/master/StarGANv2/core/solver.py#L291"""
        # Reconstruction loss in the pixel space
        loss_recon = F.l1_loss(pred, real) # `loss_org`` in the paper
        x_real_freq = find_fake_freq(real, self.gauss_kernel)  # , find=True, index=index
        x_pred_freq = find_fake_freq(pred, self.gauss_kernel)
        loss_rec_blur = F.l1_loss(x_pred_freq, x_real_freq)
        # Reconstruction loss in the Fourier space
        loss_recon_fft = fft_L1_loss_color(pred, real)
        return self.loss_weight * (loss_recon + self.lambda_recon_blur * loss_rec_blur + self.lambda_recon_fft * loss_recon_fft)



@LOSS_REGISTRY.register()
class FrequencyFeatureMergeL1Loss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Frequency Domain Image Translation: More Photo-realistic, Better Identity-preserving. In ICCV 2021.
    <https://arxiv.org/pdf/2011.13611.pdf>
    github: https://github.com/mu-cai/frequency-domain-image-translation
    This is \mathcal{L}_{org} + \lambda_1 \mathcal{L}_{rec, pix} + \lambda_3 \mathcal{L}_{rec, fft} by reducing Translation Loss in Eq(10)

    Args:
        loss_weight (float): weight for overal frequency loss. Default: 1.0
        gauss_size (int): . Default: 21
        radius (int): . Default: 21
        batch_size (int): . Default: 2
        lambda_recon_blur (float): \lambda_1 to weight \mathcal{L}_{rec, pix}. Default: 1.
        lambda_recon_fft (float): \lambda_3 to weight \mathcal{L}_{rec, fft}. Default: 1.

        Default values are referred from https://github.com/mu-cai/frequency-domain-image-translation/blob/master/StarGANv2/main.py
    """
    def __init__(self, loss_weight_l1:float=1.0, loss_weight_freq:float=1.0, reduction:str='mean', gauss_size:int=21, lambda_recon_blur:float=1., lambda_recon_fft:float=1.):
        super(FrequencyFeatureMergeL1Loss, self).__init__()
        self.gauss_kernel = get_gaussian_kernel(gauss_size).cuda()
        # # radius:int=21, batch_size:int=2,
        # self.mask_h, _ = decide_circle(r=radius, N=batch_size)
        # self.mask_h = self.mask_h.cuda()
        self.lambda_recon_blur = lambda_recon_blur
        self.lambda_recon_fft = lambda_recon_fft
        self.loss_weight_l1 = loss_weight_l1
        self.loss_weight_freq = loss_weight_freq
    def forward(self, pred, real):
        """Implementation is referred from https://github.com/mu-cai/frequency-domain-image-translation/blob/master/StarGANv2/core/solver.py#L291"""
        # Reconstruction loss in the pixel space
        loss_recon = F.l1_loss(pred, real) # `loss_org`` in the paper
        x_real_freq = find_fake_freq(real, self.gauss_kernel)  # , find=True, index=index
        x_pred_freq = find_fake_freq(pred, self.gauss_kernel)
        loss_rec_blur = F.l1_loss(x_pred_freq, x_real_freq)
        # Reconstruction loss in the Fourier space
        loss_recon_fft = fft_L1_loss_color(pred, real)
        return self.loss_weight_l1 * loss_recon + self.loss_weight_freq *  (self.lambda_recon_blur * loss_rec_blur + self.lambda_recon_fft * loss_recon_fft)
  

@LOSS_REGISTRY.register()
class FrequencyRegionFeatureLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Frequency Domain Image Translation: More Photo-realistic, Better Identity-preserving. In ICCV 2021.
    <https://arxiv.org/pdf/2011.13611.pdf>
    github: https://github.com/mu-cai/frequency-domain-image-translation
    This is \mathcal{L}_{org} + \lambda_1 \mathcal{L}_{rec, pix} + \lambda_3 \mathcal{L}_{rec, fft} by reducing Translation Loss in Eq(10)

    Args:
        loss_weight (float): weight for overal frequency loss. Default: 1.0
        gauss_size (int): . Default: 21
        radius (int): . Default: 21
        batch_size (int): . Default: 2
        lambda_recon_blur (float): \lambda_1 to weight \mathcal{L}_{rec, pix}. Default: 1.
        lambda_recon_fft (float): \lambda_3 to weight \mathcal{L}_{rec, fft}. Default: 1.

        Default values are referred from https://github.com/mu-cai/frequency-domain-image-translation/blob/master/StarGANv2/main.py
    """
    def __init__(self, l1_loss_weight:float=1.0, freq_loss_weight:float=1.0, reduction:str='mean', gauss_size:int=21, lambda_recon_blur:float=1., lambda_recon_fft:float=1.):
        super(FrequencyRegionFeatureLoss, self).__init__()
        self.gauss_kernel = get_gaussian_kernel(gauss_size).cuda()
        # # radius:int=21, batch_size:int=2,
        # self.mask_h, _ = decide_circle(r=radius, N=batch_size)
        # self.mask_h = self.mask_h.cuda()
        self.lambda_recon_blur = lambda_recon_blur
        self.lambda_recon_fft = lambda_recon_fft
        self.l1_loss_weight = l1_loss_weight
        self.freq_loss_weight = freq_loss_weight
    def forward(self, pred, real, mask):
        """Implementation is referred from https://github.com/mu-cai/frequency-domain-image-translation/blob/master/StarGANv2/core/solver.py#L291"""
        # Reconstruction loss in the pixel space
        pred_noface = pred * (1 - mask)
        real_noface = real * (1 - mask)
        loss_recon = F.l1_loss(pred_noface, real_noface) # `loss_org`` in the paper
        
        pred_face = pred * mask
        real_face = real * mask
        loss_recon_face = F.l1_loss(pred_face, real_face)
        x_real_freq = find_fake_freq(real_face, self.gauss_kernel) # , find=True, index=index
        x_pred_freq = find_fake_freq(pred_face, self.gauss_kernel)
        loss_rec_blur = F.l1_loss(x_pred_freq, x_real_freq)
        # Reconstruction loss in the Fourier space
        loss_recon_fft = fft_L1_loss_color(pred_face, real_face)
        
        loss1 = self.l1_loss_weight * loss_recon
        loss2 = self.freq_loss_weight * (loss_recon_face + self.lambda_recon_blur * loss_rec_blur + self.lambda_recon_fft * loss_recon_fft)
        return loss1 + loss2


@LOSS_REGISTRY.register()
class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>
    github: https://github.com/EndlessSora/focal-frequency-loss/blob/master/focal_frequency_loss/focal_frequency_loss.py

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, reduction:str='mean', alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

@LOSS_REGISTRY.register()
class FocalFrequencyLoss_l1(FocalFrequencyLoss):
    def __init__(self, lambda_ffl:float=1., loss_weight=1.0, reduction:str='mean', alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        """
        Notice:

            1. https://github.com/EndlessSora/focal-frequency-loss/blob/master/VanillaAE/models.py shows that self.criterion = nn.MSELoss()
            2. https://github.com/EndlessSora/focal-frequency-loss/blob/master/scripts/VanillaAE/train/celeba_recon_w_ffl.sh shows lambda_{ffl} (opt.ffl_w) = 100, while mse_w = 1.0.
        """
        super().__init__(loss_weight, reduction, alpha, patch_factor, ave_spectrum, log_matrix, batch_matrix)
        self.criterion = nn.L1Loss()
        self.lambda_ffl = lambda_ffl

    def forward(self, pred, real, matrix=None, **kwargs):
        pred_freq = self.tensor2freq(pred)
        real_freq = self.tensor2freq(real)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            real_freq = torch.mean(real_freq, 0, keepdim=True)
        loss_pix = self.criterion(pred, real)
        loss_ffl = self.loss_formulation(pred_freq, real_freq, matrix)

        # calculate focal frequency loss
        return  self.loss_weight * (loss_pix + self.lambda_ffl * loss_ffl)



@LOSS_REGISTRY.register()
class FrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Frequency Domain Image Translation: More Photo-realistic, Better Identity-preserving. In ICCV 2021.
    <https://arxiv.org/pdf/2011.13611.pdf>
    github: https://github.com/mu-cai/frequency-domain-image-translation
    This is \mathcal{L}_{org} + \lambda_1 \mathcal{L}_{rec, pix} + \lambda_3 \mathcal{L}_{rec, fft} by reducing Translation Loss in Eq(10)

    Args:
        loss_weight (float): weight for overal frequency loss. Default: 1.0
        gauss_size (int): . Default: 21
        radius (int): . Default: 21
        batch_size (int): . Default: 2
        lambda_recon_blur (float): \lambda_1 to weight \mathcal{L}_{rec, pix}. Default: 1.
        lambda_recon_fft (float): \lambda_3 to weight \mathcal{L}_{rec, fft}. Default: 1.

        Default values are referred from https://github.com/mu-cai/frequency-domain-image-translation/blob/master/StarGANv2/main.py
    """
    def __init__(self, loss_weight:float=1.0, reduction:str='mean', gauss_size:int=21, lambda_recon_blur:float=1., lambda_recon_fft:float=1.):
        super(FrequencyLoss, self).__init__()
        self.gauss_kernel = get_gaussian_kernel(gauss_size).cuda()
        # # radius:int=21, batch_size:int=2,
        # self.mask_h, _ = decide_circle(r=radius, N=batch_size)
        # self.mask_h = self.mask_h.cuda()
        self.lambda_recon_blur = lambda_recon_blur
        self.lambda_recon_fft = lambda_recon_fft
        self.loss_weight = loss_weight
    def forward(self, pred, real):
        """Implementation is referred from https://github.com/mu-cai/frequency-domain-image-translation/blob/master/StarGANv2/core/solver.py#L291"""
        # Reconstruction loss in the pixel space
        x_real_freq = find_fake_freq(real, self.gauss_kernel)  # , find=True, index=index
        x_pred_freq = find_fake_freq(pred, self.gauss_kernel)
        loss_rec_blur = F.l1_loss(x_pred_freq, x_real_freq)
        # Reconstruction loss in the Fourier space
        loss_recon_fft = fft_L1_loss_color(pred, real)
        return self.loss_weight * (self.lambda_recon_blur * loss_rec_blur + self.lambda_recon_fft * loss_recon_fft)



@LOSS_REGISTRY.register()
class WeightFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Frequency Domain Image Translation: More Photo-realistic, Better Identity-preserving. In ICCV 2021.
    <https://arxiv.org/pdf/2011.13611.pdf>
    github: https://github.com/mu-cai/frequency-domain-image-translation
    This is \mathcal{L}_{org} + \lambda_1 \mathcal{L}_{rec, pix} + \lambda_3 \mathcal{L}_{rec, fft} by reducing Translation Loss in Eq(10)

    Args:
        loss_weight (float): weight for overal frequency loss. Default: 1.0
        gauss_size (int): . Default: 21
        radius (int): . Default: 21
        batch_size (int): . Default: 2
        lambda_recon_blur (float): \lambda_1 to weight \mathcal{L}_{rec, pix}. Default: 1.
        lambda_recon_fft (float): \lambda_3 to weight \mathcal{L}_{rec, fft}. Default: 1.

        Default values are referred from https://github.com/mu-cai/frequency-domain-image-translation/blob/master/StarGANv2/main.py
    """
    def __init__(self, loss_weight:float=1.0, loss_weight2:float=0.1, weight:bool=False, reduction:str='mean', gauss_size:int=21, lambda_recon_blur:float=1., lambda_recon_fft:float=1.):
        super(WeightFrequencyLoss, self).__init__()
        self.gauss_kernel = get_gaussian_kernel(gauss_size).cuda()
        # # radius:int=21, batch_size:int=2,
        # self.mask_h, _ = decide_circle(r=radius, N=batch_size)
        # self.mask_h = self.mask_h.cuda()
        self.lambda_recon_blur = lambda_recon_blur
        self.lambda_recon_fft = lambda_recon_fft
        self.loss_weight = loss_weight
        self.loss_weight2 = loss_weight2
        self.weight = weight
    def forward(self, pred, real, weight_mask):
        """Implementation is referred from https://github.com/mu-cai/frequency-domain-image-translation/blob/master/StarGANv2/core/solver.py#L291"""
        # Reconstruction loss in the pixel space
        if not self.weight:
            weight_mask = None
            
        if weight_mask is None:
            x_real_freq = find_fake_freq(real, self.gauss_kernel)  # , find=True, index=index
            x_pred_freq = find_fake_freq(pred, self.gauss_kernel)
            loss_rec_blur = F.l1_loss(x_pred_freq, x_real_freq)
            # Reconstruction loss in the Fourier space
            loss_recon_fft = fft_L1_loss_color(pred, real)
            return self.loss_weight * (self.lambda_recon_blur * loss_rec_blur + self.lambda_recon_fft * loss_recon_fft)
        else:
            self.mask = self.loss_weight * weight_mask + self.loss_weight2 * (1 - weight_mask)
            x_real_freq = find_fake_freq_mask(real, self.gauss_kernel, self.mask)  # , find=True, index=index
            x_pred_freq = find_fake_freq_mask(pred, self.gauss_kernel, self.mask)

            loss_rec_blur = F.l1_loss(x_pred_freq, x_real_freq)
            # Reconstruction loss in the Fourier space
            loss_recon_fft = fft_L1_loss_color_mask(pred, real, self.mask)
            return self.lambda_recon_blur * loss_rec_blur + self.lambda_recon_fft * loss_recon_fft
        



@LOSS_REGISTRY.register()
class WeightFrequencyFeatureLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Frequency Domain Image Translation: More Photo-realistic, Better Identity-preserving. In ICCV 2021.
    <https://arxiv.org/pdf/2011.13611.pdf>
    github: https://github.com/mu-cai/frequency-domain-image-translation
    This is \mathcal{L}_{org} + \lambda_1 \mathcal{L}_{rec, pix} + \lambda_3 \mathcal{L}_{rec, fft} by reducing Translation Loss in Eq(10)

    Args:
        loss_weight (float): weight for overal frequency loss. Default: 1.0
        gauss_size (int): . Default: 21
        radius (int): . Default: 21
        batch_size (int): . Default: 2
        lambda_recon_blur (float): \lambda_1 to weight \mathcal{L}_{rec, pix}. Default: 1.
        lambda_recon_fft (float): \lambda_3 to weight \mathcal{L}_{rec, fft}. Default: 1.

        Default values are referred from https://github.com/mu-cai/frequency-domain-image-translation/blob/master/StarGANv2/main.py
    """
    def __init__(self, loss_weight:float=1.0, loss_weight2:float=0.1, weight:bool=False, reduction:str='mean', gauss_size:int=21, lambda_recon_blur:float=1., lambda_recon_fft:float=1.):
        super(WeightFrequencyFeatureLoss, self).__init__()
        self.gauss_kernel = get_gaussian_kernel(gauss_size).cuda()
        # # radius:int=21, batch_size:int=2,
        # self.mask_h, _ = decide_circle(r=radius, N=batch_size)
        # self.mask_h = self.mask_h.cuda()
        self.lambda_recon_blur = lambda_recon_blur
        self.lambda_recon_fft = lambda_recon_fft
        self.loss_weight = loss_weight
        self.loss_weight2 = loss_weight2
        self.weight = weight
    def forward(self, pred, real, weight_mask):
        """Implementation is referred from https://github.com/mu-cai/frequency-domain-image-translation/blob/master/StarGANv2/core/solver.py#L291"""
        # Reconstruction loss in the pixel space
        if not self.weight:
            weight_mask = None
            
        if weight_mask is None:
            loss_recon = F.l1_loss(pred, real) # `loss_org`` in the paper
            x_real_freq = find_fake_freq(real, self.gauss_kernel)  # , find=True, index=index
            x_pred_freq = find_fake_freq(pred, self.gauss_kernel)
            loss_rec_blur = F.l1_loss(x_pred_freq, x_real_freq)
            # Reconstruction loss in the Fourier space
            loss_recon_fft = fft_L1_loss_color(pred, real)
            return self.loss_weight * (loss_recon + self.lambda_recon_blur * loss_rec_blur + self.lambda_recon_fft * loss_recon_fft)
        else:
            self.mask = self.loss_weight * weight_mask + self.loss_weight2 * (1 - weight_mask)
            x_real_freq = find_fake_freq_mask(real, self.gauss_kernel, self.mask)  # , find=True, index=index
            x_pred_freq = find_fake_freq_mask(pred, self.gauss_kernel, self.mask)

            loss_rec_blur = F.l1_loss(x_pred_freq, x_real_freq)
            # Reconstruction loss in the Fourier space
            loss_recon_fft = fft_L1_loss_color_mask(pred, real, self.mask)
            
            loss_recon = F.l1_loss(pred * self.mask, real * self.mask)
            return loss_recon + self.lambda_recon_blur * loss_rec_blur + self.lambda_recon_fft * loss_recon_fft