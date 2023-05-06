import math
import os
import sys
from pathlib import Path
import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.optim as optim
from core.model import FuseModel
from torchvision import transforms
from torch.utils.data import DataLoader
from core.utils import load_config, debug, debug_color
from core.dataset.fusionDataset import FusionDataset
from core.loss import GeneratorLoss, DiscriminatorLoss


class Trainer:
    def __init__(self, project_name, config_path: str | Path, wandb_key: str):
        self.project_name = project_name
        wandb.login(key=wandb_key)  # wandb api key
        self.runs = wandb.init(project=self.project_name)
        self.gen_path = Path('weights') / self.project_name / 'generator'
        self.disc_path = Path('weights') / self.project_name / 'discriminator'
        self.gen_path.mkdir(parents=True, exist_ok=True)
        self.disc_path.mkdir(parents=True, exist_ok=True)

        self.config = load_config(config_path)
        self.fuse_model = FuseModel(self.config)
        self.datasets_config = self.config['Dataset']
        self.generator_config = self.config['Generator']
        self.discriminator_config = self.config['Discriminator']
        self.base_train_config = self.config['Train']['Base']
        self.generator_train_config = self.config['Train']['Generator']
        self.discriminator_train_config = self.config['Train']['Discriminator']
        self.train_dataset = FusionDataset(mode='train',
                                           config=self.datasets_config)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.base_train_config['batch_size'],
                                       shuffle=False)
        self.g_loss = GeneratorLoss(self.generator_config)
        self.d_loss = DiscriminatorLoss(self.discriminator_config)
        self.generator, self.discriminator = self.fuse_model.Generator.cuda(), self.fuse_model.Discriminator.cuda() \
            if torch.cuda.is_available() else (self.fuse_model.Generator, self.fuse_model.Discriminator)

        self.opt_generator = eval('optim.' + self.generator_train_config['opt'])(self.generator.parameters(),
                                                                                 self.generator_train_config['lr'])
        self.opt_discriminator = eval('optim.' + self.discriminator_train_config['opt'])(
            self.discriminator.parameters(),
            self.discriminator_train_config['lr'])

        #  设置学习率调整规则 - Warm up + Cosine Anneal
        self.gene_scheduler = LambdaLR(self.opt_generator, lr_lambda=self.get_learning_rate)

        self.epoch = 1
        self.disc_loss_ep = 0
        self.gene_loss_ep = 0
        self.disc_iter = 0
        self.gene_iter = 0

    def runer(self):
        for epoch in range(1, self.base_train_config['epoch'] + 1):
            self.epoch = epoch
            self.train_discriminator()
            self.train_generator()
            if epoch > 90:  #
                torch.save(self.generator.state_dict(), self.gen_path / f'generator_{self.epoch}.pth')
                torch.save(self.discriminator.state_dict(), self.disc_path / f'discriminator_{self.epoch}.pth')
            self.runs.log({
                'disc_loss': self.disc_loss_ep,
                'gene_loss': self.gene_loss_ep,
                'epoch': self.epoch,
                'lr_g': self.opt_generator.param_groups[0]['lr'],
                'lr_d': self.opt_discriminator.param_groups[0]['lr']
            })
        self.runs.finish()

    def train_discriminator(self):
        # Set requires_grad to False for Generator parameters
        for param in self.fuse_model.Generator.parameters():
            param.requires_grad = False

        # Set requires_grad to True for Discriminator parameters
        for param in self.fuse_model.Discriminator.parameters():
            param.requires_grad = True

        num_iter = len(self.train_loader)
        train_times_per_epoch = self.discriminator_train_config['train_times_per_epoch']
        min_loss_per_epoch = self.discriminator_train_config['min_loss_per_epoch']
        train_times = 0
        min_loss = min_loss_per_epoch
        while train_times < train_times_per_epoch and min_loss >= min_loss_per_epoch:
            all_loss = 0
            with tqdm(total=num_iter) as train_Discriminator_bar:
                for index, data in enumerate(self.train_loader):
                    if torch.cuda.is_available():
                        for i in data:
                            data[i] = data[i].cuda()
                    generator_feats, discriminator_feats, confidence = self.fuse_model(data)
                    # {'Generator_1': [2,3,512,512]}, {d_1:(4,), d_2:{4,}, {d_1:(4,), d_2:{4,}}
                    D_loss = self.d_loss(generator_feats, discriminator_feats, confidence)
                    self.opt_discriminator.zero_grad()
                    D_loss.backward()
                    self.opt_discriminator.step()
                    all_loss = all_loss + D_loss
                    train_Discriminator_bar.set_description('\tepoch:%s Train_D iter:%s loss:%.5f' %
                                                            (self.epoch, index, all_loss / num_iter))
                    train_Discriminator_bar.update(1)
                min_loss = all_loss / num_iter
                train_times += 1
                self.disc_iter += 1
                self.runs.log({
                    'disc_iter_loss': min_loss,
                    'disc_iter': self.disc_iter
                })
        self.disc_loss_ep = min_loss

    def train_generator(self):
        for param in self.fuse_model.Generator.parameters():
            param.requires_grad = True
        for param in self.fuse_model.Discriminator.parameters():
            param.requires_grad = False
        num_iter = len(self.train_loader)
        train_times_per_epoch = self.generator_train_config['train_times_per_epoch']
        min_loss_per_epoch = self.generator_train_config['min_loss_per_epoch']
        train_times = 0
        min_loss = min_loss_per_epoch
        while train_times < train_times_per_epoch and min_loss >= min_loss_per_epoch:
            all_loss = 0
            img_record = 0
            with tqdm(total=num_iter) as train_Generator_bar:
                for index, data in enumerate(self.train_loader):
                    if torch.cuda.is_available():
                        for i in data:
                            data[i] = data[i].cuda()
                    generator_feats, discriminator_feats, confidence = self.fuse_model(data)
                    # {'Generator_1': [2,3,512,512]}, {d_1:(4,), d_2:{4,}, {d_1:(4,), d_2:{4,}}
                    G_loss = self.g_loss(data, generator_feats, discriminator_feats, confidence)
                    self.opt_generator.zero_grad()
                    G_loss.backward()
                    self.opt_generator.step()
                    all_loss = all_loss + G_loss

                    # 记录训练过程中的图像融合情况
                    if self.datasets_config['color']:
                        vis_i, ir_i, fuse_i = debug_color(self.epoch, data, generator_feats)
                        if img_record < 5:
                            vis = wandb.Image(vis_i, caption="epoch:{}".format(self.epoch))
                            ir = wandb.Image(ir_i, caption="epoch:{}".format(self.epoch))
                            fuse = wandb.Image(fuse_i, caption="epoch:{}".format(self.epoch))
                            self.runs.log({
                                "vis": vis,
                                "ir": ir,
                                "fuse": fuse
                            })
                            img_record += 1
                    else:
                        img = debug(self.epoch, data, generator_feats)
                        if img_record < 5:
                            Img = wandb.Image(img, caption="epoch:{}".format(self.epoch))
                            self.runs.log({
                                "fuse": Img
                            })
                            img_record += 1

                    train_Generator_bar.set_description('\tepoch:%s Train_G iter:%s loss:%.5f' %
                                                        (self.epoch, index, all_loss / num_iter))
                    train_Generator_bar.update(1)
                min_loss = all_loss / num_iter
                train_times += 1
                self.gene_iter += 1
                self.runs.log({
                    'gene_iter_loss': min_loss,
                    'gene_iter': self.gene_iter
                })
        self.gene_scheduler.step()
        self.gene_loss_ep = min_loss

    def get_learning_rate(self, cur_iter):
        warm_up_iter = self.generator_train_config['warm_up_epoch']  # 设置warm up的轮次
        t_max = self.base_train_config['epoch']  # 周期
        lr_max = 0.1  # 最大值
        lr_min = 1e-5  # 最小值
        if cur_iter < warm_up_iter:
            return cur_iter / warm_up_iter
        else:
            cosine_decay = 0.5 * (1.0 + math.cos((cur_iter - warm_up_iter) / (t_max - warm_up_iter) * math.pi))
            return (lr_min + (lr_max - lr_min) * cosine_decay) / 0.1


if __name__ == '__main__':
    trainer = Trainer(project_name='GAN_G1_D2_COLOR',
                      config_path='./config/GAN_G1_D2_color_s.yaml',
                      wandb_key=f'49deeeb7e29fb1acb9e77e00885bc52d739dee0f')
    trainer.runer()
