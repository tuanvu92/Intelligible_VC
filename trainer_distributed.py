# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm
from progressbar import *

from losses_distributed import compute_d_loss, compute_g_loss, compute_sty_loss

import logging
from distributed import init_distributed, reduce_tensor, apply_gradient_allreduce

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Trainer(object):
    def __init__(self,
                 args,
                 model=None,
                 model_ema=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0,
                 fp16_run=False,
                 rank=0,
                 num_gpus=1
    ):
        self.args = args
        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = fp16_run
        # =====START: ADDED FOR DISTRIBUTED======
        self.rank = rank
        self.num_gpus = num_gpus
        # =====END:   ADDED FOR DISTRIBUTED======

    @torch.no_grad()
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        pass

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
            "model": {key: self.model[key].state_dict() for key in self.model}
        }
        if self.model_ema is not None:
            state_dict['model_ema'] = {key: self.model_ema[key].state_dict() for key in self.model_ema}

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        for key in self.model:
            self._load(state_dict["model"][key], self.model[key])

        if self.model_ema is not None:
            for key in self.model_ema:
                self._load(state_dict["model_ema"][key], self.model_ema[key])

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])

    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    if self.rank == 0:
                        self.logger.info("%s does not have same shape" % key)
                        print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                if self.rank == 0:
                    self.logger.info("not exist :%s" % key)
                    print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    @staticmethod
    def moving_average(model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.cpu().data, param_test.data, beta)

    def _train_epoch(self):
        self.epochs += 1
        self.train_dataloader.sampler.set_epoch(self.epochs)
        train_losses = defaultdict(list)
        _ = [self.model[k].train() for k in self.model]
        scaler = torch.cuda.amp.GradScaler() if self.fp16_run else None

        use_con_reg = (self.epochs >= self.args.con_reg_epoch)
        use_adv_cls = (self.epochs >= self.args.adv_cls_epoch)
        
        if self.rank == 0:
            widgets = [FormatLabel(''), Bar('#'), ' ', Percentage(format='%(percentage).1f%%'),
                       " (", Counter(), "|%d) " % len(self.train_dataloader), " ", Timer(), " ", ETA()]
            iterator = progressbar(self.train_dataloader, redirect_stdout=True, widgets=widgets)
        else:
            iterator = self.train_dataloader
        
        for batch in iterator:
            # load data
            batch = [b.cuda() for b in batch]
            x_real, y_org, x_ref, x_ref2, y_trg, z_trg, z_trg2 = batch
            
            # train the discriminator (by random reference)
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    d_loss, d_losses_latent = compute_d_loss(self.model, self.args.d_loss, x_real, y_org, y_trg,
                                                             z_trg=z_trg, use_adv_cls=use_adv_cls,
                                                             use_con_reg=use_con_reg)
                scaler.scale(d_loss).backward()
                for key in d_losses_latent:
                    d_losses_latent[key] = reduce_tensor(d_losses_latent[key], self.num_gpus).item()
            else:
                d_loss, d_losses_latent = compute_d_loss(self.model, self.args.d_loss, x_real, y_org, y_trg,
                                                         z_trg=z_trg, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg)
                d_loss.backward()
                for key in d_losses_latent:
                    d_losses_latent[key] = reduce_tensor(d_losses_latent[key], self.num_gpus).item()

            self.optimizer.step('discriminator', scaler=scaler)

            # train the discriminator (by target reference)
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    d_loss, _ = compute_d_loss(self.model, self.args.d_loss, x_real, y_org, y_trg,
                                               x_ref=x_ref, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg)
                scaler.scale(d_loss).backward()
            else:
                d_loss, _ = compute_d_loss(self.model, self.args.d_loss, x_real, y_org, y_trg, x_ref=x_ref,
                                           use_adv_cls=use_adv_cls, use_con_reg=use_con_reg)
                d_loss.backward()

            self.optimizer.step('discriminator', scaler=scaler)

            # train the generator (by random reference)
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    g_loss, g_losses_latent = compute_g_loss(
                        self.model, self.args.g_loss, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], use_adv_cls=use_adv_cls)
                # g_loss = reduce_tensor(g_loss, 2).item()
                for key in g_losses_latent:
                    g_losses_latent[key] = reduce_tensor(g_losses_latent[key], self.num_gpus).item()
                scaler.scale(g_loss).backward()
            else:
                g_loss, g_losses_latent = compute_g_loss(
                    self.model, self.args.g_loss, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], use_adv_cls=use_adv_cls)
                g_loss.backward()
                for key in g_losses_latent:
                    g_losses_latent[key] = reduce_tensor(g_losses_latent[key], self.num_gpus).item()
                
            self.optimizer.step('generator', scaler=scaler)
            self.optimizer.step('mapping_network', scaler=scaler)
            self.optimizer.step('style_encoder', scaler=scaler)

            # train the generator (by target reference)
            self.optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    g_loss, g_losses_ref = compute_g_loss(
                        self.model, self.args.g_loss, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2],
                        use_adv_cls=use_adv_cls)
                scaler.scale(g_loss).backward()
                for key in g_losses_ref:
                    g_losses_ref[key] = reduce_tensor(g_losses_ref[key], self.num_gpus).item()
            else:
                g_loss, g_losses_ref = compute_g_loss(
                    self.model, self.args.g_loss, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], use_adv_cls=use_adv_cls)
                g_loss.backward()
                for key in g_losses_ref:
                    g_losses_ref[key] = reduce_tensor(g_losses_ref[key], self.num_gpus).item()
            self.optimizer.step('generator', scaler=scaler)
            g_losses_latent['f0_sty'] = g_losses_ref['f0_sty']           

            if self.rank == 0 and self.model_ema is not None:
                # compute moving average of network parameters
                self.moving_average(self.model.generator, self.model_ema.generator, beta=0.999)
                self.moving_average(self.model.mapping_network, self.model_ema.mapping_network, beta=0.999)
                self.moving_average(self.model.style_encoder, self.model_ema.style_encoder, beta=0.999)

            self.optimizer.scheduler()
            for key in d_losses_latent:
                train_losses["train/%s" % key].append(d_losses_latent[key])
            for key in g_losses_latent:
                train_losses["train/%s" % key].append(g_losses_latent[key])
            
            if self.rank == 0:
                widgets[0] = FormatLabel("{:d}|{:d}: ".format(self.epochs, self.steps) +
                             ' '.join('{}={:.2e}'.format(k, g_losses_latent[k]) for k in g_losses_latent.keys()))
            self.steps += 1
                
        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        use_adv_cls = (self.epochs >= self.args.adv_cls_epoch)
        
        eval_losses = defaultdict(list)
        eval_images = defaultdict(list)
        _ = [self.model[k].eval() for k in self.model]
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):

            ### load data
            batch = [b.cuda() for b in batch]
            x_real, y_org, x_ref, x_ref2, y_trg, z_trg, z_trg2 = batch

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                self.model, self.args.d_loss, x_real, y_org, y_trg, z_trg=z_trg, use_r1_reg=False, use_adv_cls=use_adv_cls)
            d_loss, d_losses_ref = compute_d_loss(
                self.model, self.args.d_loss, x_real, y_org, y_trg, x_ref=x_ref, use_r1_reg=False, use_adv_cls=use_adv_cls)

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                self.model, self.args.g_loss, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], use_adv_cls=use_adv_cls)
            g_loss, g_losses_ref = compute_g_loss(
                self.model, self.args.g_loss, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], use_adv_cls=use_adv_cls)

            for key in d_losses_latent:
                eval_losses["eval/%s" % key].append(d_losses_latent[key].item())
            for key in g_losses_latent:
                eval_losses["eval/%s" % key].append(g_losses_latent[key].item())

            # if eval_steps_per_epoch % 10 == 0:
            #     # generate x_fake
            #     s_trg = self.model_ema.style_encoder(x_ref.cpu(), y_trg.cpu())
            #     F0 = self.model.f0_model.get_feature_GAN(x_real)
            #     x_fake = self.model_ema.generator(x_real.cpu(), s_trg.cpu(), masks=None, F0=F0.cpu())
            #     # generate x_recon
            #     s_real = self.model_ema.style_encoder(x_real.cpu(), y_org.cpu())
            #     F0_fake = self.model.f0_model.get_feature_GAN(x_fake.cuda())
            #     x_recon = self.model_ema.generator(x_fake.cpu(), s_real.cpu(), masks=None, F0=F0_fake.cpu())
            #
            #     eval_images['eval/image'].append(
            #         ([x_real[0, 0].cpu().numpy(),
            #         x_fake[0, 0].cpu().numpy(),
            #         x_recon[0, 0].cpu().numpy()]))

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_losses.update(eval_images)
        return eval_losses

        
class TrainerStyle(object):
    def __init__(self,
                 args,
                 style_net_student=None,
                 style_net_teacher=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0,
                 fp16_run=False,
                 rank=0,
                 num_gpus=1
    ):
        self.args = args
        self.steps = initial_steps
        self.epochs = initial_epochs
        self.style_net_student = style_net_student
        self.style_net_teacher = style_net_teacher
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = fp16_run
        # =====START: ADDED FOR DISTRIBUTED======
        self.rank = rank
        self.num_gpus = num_gpus
        # =====END:   ADDED FOR DISTRIBUTED======

    @torch.no_grad()
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        pass

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
            "model": self.style_net_student.state_dict()
        }
       
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.style_net_teacher.load_state_dict(state_dict["model_ema"]["style_encoder"])
        
#         if not load_only_params:
#             self.steps = state_dict["steps"]
#             self.epochs = state_dict["epochs"]
#             self.optimizer.load_state_dict(state_dict["optimizer"])
    
    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    def _train_epoch(self):
        self.epochs += 1
        self.train_dataloader.sampler.set_epoch(self.epochs)
        scaler = torch.cuda.amp.GradScaler() if self.fp16_run else None
        train_losses = defaultdict(list)
        if self.rank == 0:
            widgets = [FormatLabel(''), Bar('#'), ' ', Percentage(format='%(percentage).1f%%'),
                       " (", Counter(), "|%d) " % len(self.train_dataloader), " ", Timer(), " ", ETA()]
            iterator = progressbar(self.train_dataloader, redirect_stdout=True, widgets=widgets)
        else:
            iterator = self.train_dataloader
        
        for batch in iterator:
            # load data
            batch = [b.cuda() for b in batch]
            x_real, y_org, x_ref, x_ref2, y_trg, z_trg, z_trg2 = batch
            
            # train the style encoder (by random reference)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, _ = compute_sty_loss(self.style_net_student,
                                           self.style_net_teacher,
                                           x_real, 
                                           y_org, 
                                           y_trg,
                                           [x_ref, x_ref2])
            scaler.scale(loss).backward()
            loss = reduce_tensor(loss, self.num_gpus).item()
            train_losses["loss"].append(loss)
            self.optimizer.step('style_encoder', scaler=scaler)            
            self.optimizer.scheduler()                        
            if self.rank == 0:
                widgets[0] = FormatLabel("{:d}|{:d}: loss={:.2e}".format(self.epochs, self.steps, loss))
            self.steps += 1
                
        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses


class TrainerFinetune(object):
    def __init__(self,
                 args,
                 model=None,
                 style_encoder=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0,
                 fp16_run=False,
                 rank=0,
                 num_gpus=1
                 ):
        self.args = args
        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.style_encoder = style_encoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = fp16_run
        # =====START: ADDED FOR DISTRIBUTED======
        self.rank = rank
        self.num_gpus = num_gpus
        # =====END:   ADDED FOR DISTRIBUTED======

    @torch.no_grad()
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        pass

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
            "model": self.style_encoder.state_dict()
        }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model_ema"]
        self.model.generator.load_state_dict(state_dict["generator"])

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    def _train_epoch(self):
        self.epochs += 1
        self.train_dataloader.sampler.set_epoch(self.epochs)
        scaler = torch.cuda.amp.GradScaler() if self.fp16_run else None
        train_losses = defaultdict(list)
        if self.rank == 0:
            widgets = [FormatLabel(''), Bar('#'), ' ', Percentage(format='%(percentage).1f%%'),
                       " (", Counter(), "|%d) " % len(self.train_dataloader), " ", Timer(), " ", ETA()]
            iterator = progressbar(self.train_dataloader, redirect_stdout=True, widgets=widgets)
        else:
            iterator = self.train_dataloader

        for batch in iterator:
            # load data
            batch = [b.cuda() for b in batch]
            x_real, y_org, x_ref, x_ref2, y_trg, z_trg, z_trg2 = batch

            # train the style encoder (by random reference)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, _ = compute_sty_loss(self.style_net_student,
                                           self.style_net_teacher,
                                           x_real,
                                           y_org,
                                           y_trg,
                                           [x_ref, x_ref2])
            scaler.scale(loss).backward()
            loss = reduce_tensor(loss, self.num_gpus).item()
            train_losses["loss"].append(loss)
            self.optimizer.step('style_encoder', scaler=scaler)
            self.optimizer.scheduler()
            if self.rank == 0:
                widgets[0] = FormatLabel("{:d}|{:d}: loss={:.2e}".format(self.epochs, self.steps, loss))
            self.steps += 1

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses
