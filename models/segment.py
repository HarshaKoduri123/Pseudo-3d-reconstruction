import torch
import torch.optim
import torch.nn as nn
import pytorch_lightning as pl

from models.unet3d_rec import MAELoss, random_mask


class Segment(pl.LightningModule):

    def __init__(self, model, optimizer=None, train_data=None, valid_data=None, tests_data=None,
                 loss=None, train_metrics=[], input_metrics=[], valid_metrics=[], tests_metrics=[],
                 lr_start=0.1, lr_param=1, schedule='flat', **kwargs):
        super().__init__()
        self.optimizer     = optimizer
        self.model         = model
        self.loss          = loss
        self.mae_loss      = MAELoss()         
        self.train_metrics = nn.ModuleList(train_metrics)
        self.valid_metrics = nn.ModuleList(valid_metrics)
        self.tests_metrics = nn.ModuleList(tests_metrics)
        self.train_data    = train_data
        self.valid_data    = valid_data
        self.lr_start      = lr_start
        self.lr_param      = lr_param
        self.schedule      = schedule

    def training_step(self, batch, batch_idx):

        dense = batch['tgt']   
        sparse, obs_mask = random_mask(dense, mask_ratio_range=(0.5, 0.9))

        pred = self.model(sparse, obs_mask)

        loss = self.mae_loss(pred, dense, obs_mask)

        with torch.no_grad():
            full_mse = torch.nn.functional.mse_loss(pred, dense)

        self.log('trn_loss',     loss,     prog_bar=True,  logger=True)
        self.log('trn_mse',      full_mse, prog_bar=False, logger=True)
        self.log('learn_rate',   self.trainer.optimizers[0].param_groups[0]['lr'],
                 prog_bar=True,  logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        sparse = batch['vol']  
        dense  = batch['tgt']   

        obs_mask = (sparse.mean(dim=[1,3,4], keepdim=True) > 0).float()
        obs_mask = obs_mask.expand_as(sparse)

        pred     = self.model(sparse, obs_mask)


        val_loss = self.mae_loss(pred, dense, obs_mask)

        full_mse = torch.nn.functional.mse_loss(pred, dense)

        with torch.no_grad():
            ssim_approx = 1.0 - val_loss  

        self.log('val_loss', val_loss,   prog_bar=True,  logger=True)
        self.log('val_mse',  full_mse,   prog_bar=True,  logger=True)
        self.log('val_ssim', ssim_approx, prog_bar=False, logger=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        sparse = batch['vol']
        dense  = batch['tgt']

        obs_mask = (sparse.mean(dim=[1,3,4], keepdim=True) > 0).float()
        obs_mask = obs_mask.expand_as(sparse)

        pred     = self.model(sparse, obs_mask)
        tst_loss = self.mae_loss(pred, dense, obs_mask)
        full_mse = torch.nn.functional.mse_loss(pred, dense)

        self.log('tst_loss', tst_loss, prog_bar=True, logger=True)
        self.log('tst_mse',  full_mse, prog_bar=True, logger=True)

        return tst_loss

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        def lr(step):
            if self.schedule == 'poly':
                return (1.0 - (step / self.trainer.max_steps)) ** self.lr_param
            elif self.schedule == 'step':
                return (0.1 ** (step // self.lr_param))
            else:
                return 1.0

        if self.schedule == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr)

        lr_scheduler = {
            'interval':  'epoch' if self.schedule == 'plateau' else 'step',
            'scheduler': scheduler,
            'monitor':   'val_loss',
        }
        return {'optimizer': self.optimizer, 'lr_scheduler': lr_scheduler}