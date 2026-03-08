import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from collections import OrderedDict
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.archs import build_network
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
# from .base_model import BaseModel



@MODEL_REGISTRY.register()
class SkinPairModel(SRModel):
    """RealESRNet Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(SkinPairModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # for paired training or validation
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(SkinPairModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True



@MODEL_REGISTRY.register()
class SkinPairLpipsModel(SRModel):
    """RealESRNet Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(SkinPairLpipsModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # for paired training or validation
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(SkinPairModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # # perceptual loss
        # if self.cri_perceptual:
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style

        if self.cri_perceptual:
            l_percep = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep


        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)




@MODEL_REGISTRY.register()
class SkinPairLpipsFreqModel(SRModel):
    """RealESRNet Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(SkinPairLpipsFreqModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # for paired training or validation
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('freq_opt'):
            self.cri_freq = build_loss(train_opt['freq_opt']).to(self.device)
        else:
            self.cri_freq = None
        
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(SkinPairModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # # perceptual loss
        # if self.cri_perceptual:
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style

        if self.cri_perceptual:
            l_percep = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep

        if self.cri_freq:
            l_freq = self.cri_freq(self.output, self.gt)
            if l_freq is not None:
                l_total += l_freq
                loss_dict['l_freq'] = l_freq

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)













@MODEL_REGISTRY.register()
class SkinPairLpipsGANModel(SRGANModel):
    """RealESRNet Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(SkinPairLpipsGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    def init_training_settings(self):
            train_opt = self.opt['train']
    
            self.ema_decay = train_opt.get('ema_decay', 0)
            if self.ema_decay > 0:
                logger = get_root_logger()
                logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
                # define network net_g with Exponential Moving Average (EMA)
                # net_g_ema is used only for testing on one GPU and saving
                # There is no need to wrap with DistributedDataParallel
                self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
                # load pretrained model
                load_path = self.opt['path'].get('pretrain_network_g', None)
                if load_path is not None:
                    self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
                else:
                    self.model_ema(0)  # copy net_g weight
                self.net_g_ema.eval()
    
            # define network net_d
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
    
            # load pretrained models
            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None:
                param_key = self.opt['path'].get('param_key_d', 'params')
                self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)
    
            self.net_g.train()
            self.net_d.train()
    
            # define losses
            if train_opt.get('pixel_opt'):
                self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            else:
                self.cri_pix = None
    
            if train_opt.get('ldl_opt'):
                self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)
            else:
                self.cri_ldl = None
    
            if train_opt.get('perceptual_opt'):
                self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
            else:
                self.cri_perceptual = None
    
            if train_opt.get('gan_opt'):
                self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

            if train_opt.get('freq_opt'):
                self.cri_freq = build_loss(train_opt['freq_opt']).to(self.device)
            else:
                self.cri_freq = None

        
            self.net_d_iters = train_opt.get('net_d_iters', 1)
            self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
    
            # set up optimizers and schedulers
            self.setup_optimizers()
            self.setup_schedulers()

    
    # @torch.no_grad()
    # def feed_data(self, data):
    #     """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
    #     """
    #     # for paired training or validation
    #     self.lq = data['lq'].to(self.device)
    #     self.gt = data['gt'].to(self.device)


    # def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
    #     # do not use the synthetic process during validation
    #     self.is_train = False
    #     super(SkinPairModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
    #     self.is_train = True

    # def optimize_parameters(self, current_iter):
    #     self.optimizer_g.zero_grad()
    #     self.output = self.net_g(self.lq)

    #     l_total = 0
    #     loss_dict = OrderedDict()
    #     # pixel loss
    #     if self.cri_pix:
    #         l_pix = self.cri_pix(self.output, self.gt)
    #         l_total += l_pix
    #         loss_dict['l_pix'] = l_pix
    #     # # perceptual loss
    #     # if self.cri_perceptual:
    #     #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
    #     #     if l_percep is not None:
    #     #         l_total += l_percep
    #     #         loss_dict['l_percep'] = l_percep
    #     #     if l_style is not None:
    #     #         l_total += l_style
    #     #         loss_dict['l_style'] = l_style

    #     if self.cri_perceptual:
    #         l_percep = self.cri_perceptual(self.output, self.gt)
    #         if l_percep is not None:
    #             l_total += l_percep
    #             loss_dict['l_percep'] = l_percep


    #     l_total.backward()
    #     self.optimizer_g.step()

    #     self.log_dict = self.reduce_loss_dict(loss_dict)

    #     if self.ema_decay > 0:
    #         self.model_ema(decay=self.ema_decay)



    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt
        percep_gt = self.gt
        gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            # if self.cri_perceptual:
            #     l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
            #     if l_g_percep is not None:
            #         l_g_total += l_g_percep
            #         loss_dict['l_g_percep'] = l_g_percep
            #     if l_g_style is not None:
            #         l_g_total += l_g_style
            #         loss_dict['l_g_style'] = l_g_style

            if self.cri_perceptual:
                l_percep = self.cri_perceptual(self.output, percep_gt)
                if l_percep is not None:
                    l_g_total += l_percep
                    loss_dict['l_g_percep'] = l_percep
                    
            if self.cri_freq:
                l_freq = self.cri_freq(self.output, self.gt)
                if l_freq is not None:
                    l_g_total += l_freq
                    loss_dict['l_g_freq'] = l_freq

            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)