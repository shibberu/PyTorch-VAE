import os
import math
import torch
from torch import optim, Tensor
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

class VAEXperiment(pl.LightningModule):
    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(VAEXperiment, self).__init__()
#        self.automatic_optimization = False
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(
                *results,
                M_N=self.params['kld_weight'],  # e.g. batch_size/num_train_imgs
                batch_idx=batch_idx
                )
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

#        # Manual optimization:
#        opt1, opt2 = self.optimizers()  # Adjust if you have more or fewer optimizers
#        self.manual_backward(loss)
#
#        # Step each optimizer and zero gradients
#        opt1.step()
#        opt1.zero_grad()
#        opt2.step()
#        opt2.zero_grad()

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
                *results,
                M_N=1.0,  # e.g. 1.0 for full batch weight
                optimizer_idx=optimizer_idx,
                batch_idx=batch_idx
                )
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    # Renamed hook: previously "on_validation_end" is now "on_validation_epoch_end"
    def on_validation_epoch_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(
            recons.data,
            os.path.join(self.logger.log_dir,
                         "Reconstructions",
                         f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
            normalize=True,
            nrow=12
        )
        try:
            samples = self.model.sample(144, self.curr_device, labels=test_label)
            vutils.save_image(
                samples.cpu().data,
                os.path.join(self.logger.log_dir,
                             "Samples",
                             f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                normalize=True,
                nrow=12
            )
        except Warning:
            pass

    def configure_optimizers(self):
        optims = []
        scheds = []
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
        )
#        optims.append(optimizer)
#        # If a second optimizer is required (e.g. for adversarial training)
#        try:
#            if self.params['LR_2'] is not None:
#                optimizer2 = optim.Adam(
#                    getattr(self.model, self.params['submodel']).parameters(),
#                    lr=self.params['LR_2']
#                )
#                optims.append(optimizer2)
#        except:
#            pass
#        try:
#            if self.params['scheduler_gamma'] is not None:
#                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
#                                                               gamma=self.params['scheduler_gamma'])
#                scheds.append(scheduler)
#                try:
#                    if self.params['scheduler_gamma_2'] is not None:
#                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
#                                                                        gamma=self.params['scheduler_gamma_2'])
#                        scheds.append(scheduler2)
#                except:
#                    pass
#            return optims, scheds
#        except:
#            return optims

