import logging
from collections import OrderedDict
import random
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn.functional as F
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.base_model import BaseModel
from models.compressor import REALCOMP
from models.jpeg import DiffJPEG
from models.modules.loss import ReconstructionLoss
from models.modules.quantization import Quantization
import numpy as np

logger = logging.getLogger('base')


class TSAIN(BaseModel):
    def __init__(self, opt):
        super(TSAIN, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        if opt['scale']==2:
            self.net = networks.define(opt)[2].to(self.device)
        elif opt['scale']==4:
            self.net = networks.define(opt)[3].to(self.device)

        if opt['dist']:
            self.net = DistributedDataParallel(self.net, device_ids=[torch.cuda.current_device()])
        else:
            self.net = DataParallel(self.net)
        
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()
        

        if self.is_train:
            self.net.train()
            
            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            # optimizers
            wd = train_opt['weight_decay'] if train_opt['weight_decay'] else 0
            optim_params = []
            
            for k, v in self.net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            for name, param in self.net.named_parameters():
                print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
            self.optimizer = torch.optim.Adam(optim_params, lr=train_opt['lr'],
                                                weight_decay=wd,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer)
            
            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
    def diffcomp(self,x,qf):
        return DiffJPEG(differentiable=True, quality=qf)(x).to(x.device)  
    def realcomp(self,x,qf):
        return REALCOMP(format='JPEG', quality=qf)(x).to(x.device)
    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT
    def num_paras(self):
        total_parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        return total_parameters
    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        return l_forw_fit

    def loss_backward(self, out, x):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, x)
        return l_back_rec

    def optimize_parameters(self, step):
        self.optimizer.zero_grad()
        self.qf = random.choice([75])
        # forward downscaling
        self.input = self.real_H
        self.output, LR = self.net(x=self.input)

        zshape = self.output[:, 3:, :, :].shape
        LR_ref = self.ref_L.detach()

        l_forw_fit1 = self.loss_forward(LR, LR_ref)

        # backward upscaling
        LR = self.Quantization(LR)
        LR_ = self.diffcomp(LR,self.qf)

        gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1
        
        y_ = torch.cat((LR_, gaussian_scale * torch.zeros(zshape).cuda()), dim=1)
        x_samples, LR_recon = self.net(y_, rev=True)
        x_samples_recon = x_samples[:, :3, :, :]
        
        l_back_rec = self.loss_backward(x_samples_recon, self.real_H)

        LR_ref = self.realcomp(self.ref_L.detach(),self.qf)

        l_forw_fit2 = self.loss_forward(self.output[:, :3, :, :], LR_ref)

        l_reg = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(LR, LR_recon)
        l_rel = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(self.output[:, :3, :, :], self.realcomp(self.Quantization(LR),self.qf))


        loss = l_back_rec + (l_forw_fit1 + l_forw_fit2 + l_reg + l_rel) / 4 
        
        loss.backward()
        
        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.train_opt['gradient_clipping'])
        
        self.optimizer.step()

        # set log
        self.log_dict['l_forw_fit1'] = l_forw_fit1.item()
        self.log_dict['l_forw_fit2'] = l_forw_fit2.item()
        self.log_dict['l_reg'] = l_reg.item()
        self.log_dict['l_rel'] = l_rel.item()
        self.log_dict['l_back_rec2'] = l_back_rec.item()
        
    def test(self):
        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        self.input = self.real_H

        zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.net.eval()
        with torch.no_grad():
            self.forw_L = self.Quantization(self.net(x=self.input)[1])
            self.forw_L_hat = self.net(x=self.input)[0][:, :3, :, :]
            self.back_L = REALCOMP(format='JPEG', quality=90)(self.forw_L)
            y_forw = torch.cat((REALCOMP(format=self.train_opt['comp_format'], quality=90)(self.forw_L), gaussian_scale * torch.zeros(zshape).cuda()), dim=1)
            self.fake_H = self.net(x=y_forw, rev=True)[0][:, :3, :, :]

        self.net.train()

    def downscale(self, HR_img):
        self.net.eval()
        with torch.no_grad():
            LR_img = self.Quantization(self.net(x=HR_img)[1][:, :3, :, :])
        self.net.train()

        return LR_img

    def upscale(self, LR_img, scale,gaussian_scale=1):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale**2)- Lshape[1], Lshape[2], Lshape[3]] 

        self.net.eval()
        with torch.no_grad():
            y_ = torch.cat((LR_img, gaussian_scale * torch.zeros(zshape).cuda()), dim=1)
            HR_img = self.net(x=y_,rev=True)[0][:,:3,:,:]
        self.net.train()

        return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.net)
        if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.net.__class__.__name__,
                                             self.net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.net.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path = self.opt['path']['pretrain_model']
        if load_path is not None:
            logger.info('Loading model from [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.net, self.opt['path']['strict_load'])
        
    def save(self, iter_label):
        self.save_network(self.net, 'net', iter_label)

    
        
class TIRN(BaseModel):
    def __init__(self, opt):
        super(TIRN, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.train_opt = train_opt
        self.test_opt = test_opt
        if opt['scale']==2:
            self.net = networks.define(opt)[0].to(self.device)
        elif opt['scale']==4:
            self.net = networks.define(opt)[1].to(self.device)
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        logger.info(f"Total number of parameters: {total_params}")
        if opt['dist']:
            self.net = DistributedDataParallel(self.net, device_ids=[torch.cuda.current_device()])
        else:
            self.net = DataParallel(self.net)

        # print network
        self.print_network()
        self.load()
        self.quantori = Quantization()
        if self.is_train:
            self.net.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            # optimizers
            wd = train_opt['weight_decay'] if train_opt['weight_decay'] else 0
            optim_params = []
            for k, v in self.net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer = torch.optim.Adam(optim_params, lr=train_opt['lr'],
                                                weight_decay=wd,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer)
            
            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT
    def num_paras(self):
        total_parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        return total_parameters
        
    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        return l_forw_fit

    def loss_backward(self, out, x):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, x)
        return l_back_rec

    def optimize_parameters(self, step):
        self.optimizer.zero_grad()

        # forward downscaling
        self.input = self.real_H
        self.output,LR = self.net(x=self.input)

        zshape = self.output[:, 3:, :, :].shape#3->4
        LR_ref = self.ref_L.detach()
        
        # backward upscaling
        LR_=self.quantori(LR)#3-->4

        gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1
        y_ = torch.cat((LR_,gaussian_scale * torch.zeros(zshape).cuda()), dim=1)
        _,x_samples= self.net(y_,rev=True)
        
        x_samples_recon = x_samples[:, :3, :, :]
        l_back_rec = self.loss_backward(x_samples_recon, self.real_H)
        LR_ref = self.ref_L.detach()

        l_forw_fit = self.loss_forward(LR, LR_ref)

        loss = l_back_rec + l_forw_fit/4 

        loss.backward()
        
        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.train_opt['gradient_clipping'])
        
        self.optimizer.step()

        # set log

        self.log_dict['l_forw_fit2'] = l_forw_fit.item()
        self.log_dict['l_back_rec2'] = l_back_rec.item()
        
    def test(self):
        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        self.input = self.real_H
        zshape = [Lshape[0], input_dim * (self.opt['scale']**2) - Lshape[1], Lshape[2], Lshape[3]]
        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.net.eval()
        with torch.no_grad():
            self.forw_L_hat = self.quantori(self.net(x=self.input)[1][:, :3, :, :])
            y_forw = torch.cat((self.forw_L_hat, gaussian_scale * torch.zeros(zshape).cuda()), dim=1)
            self.fake_H = self.net(x=y_forw,rev=True)[1][:, :3, :, :]
        self.net.train()
        
    def downscale(self, HR_img):
        self.net.eval()
        with torch.no_grad():
            LR_img = self.quantori(self.net(x=HR_img)[1][:, :3, :, :])
        self.net.train()

        return LR_img

    def upscale(self, LR_img, scale,gaussian_scale=1):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale**2)- Lshape[1], Lshape[2], Lshape[3]] 

        self.net.eval()
        with torch.no_grad():
            y_ = torch.cat((LR_img, gaussian_scale * torch.zeros(zshape).cuda()), dim=1)
            HR_img = self.net(x=y_,rev=True)[1][:,:3,:,:]
        self.net.train()

        return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L_hat.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.net)
        if isinstance(self.net, nn.DataParallel) or isinstance(self.net, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.net.__class__.__name__,
                                             self.net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.net.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path = self.opt['path']['pretrain_model']
        if load_path is not None:
            logger.info('Loading model from [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.net, self.opt['path']['strict_load'])
        
    def save(self, iter_label):
        self.save_network(self.net, 'net', iter_label)


