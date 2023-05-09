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
from kornia.metrics import AverageMeter


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
                                       shuffle=False,
                                       num_workers=0)
        self.g_loss_fn = GeneratorLoss(self.generator_config)
        self.d_loss_fn = DiscriminatorLoss(self.discriminator_config)
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
        self.disc_loss_ep = []  # 判别器loss [d_his.avg, dv_his.avg, di_his.avg]
        self.gene_loss_ep = []  # 生成器loss [g_his.avg, g_con_his.avg, g_adv_his.avg]
        self.disc_iter = 0  # 判别器训练轮数（一回合会有多轮训练）
        self.gene_iter = 0  # 生成器训练轮数（一回合会有多轮训练）

    def runer(self):
        for epoch in range(1, self.base_train_config['epoch'] + 1):
            self.epoch = epoch
            self.train_discriminator()
            self.train_generator()
            if epoch > 0:  # 保存训练的权重
                torch.save(self.generator.state_dict(), self.gen_path / f'generator_{self.epoch}.pth')
                torch.save(self.discriminator.state_dict(), self.disc_path / f'discriminator_{self.epoch}.pth')
            self.runs.log({
                'disc_total_loss': self.disc_loss_ep[0],
                'disc_vi_loss': self.disc_loss_ep[1],
                'disc_ir_loss': self.disc_loss_ep[2],
                'gene_total_loss': self.gene_loss_ep[0],
                'gene_con_loss': self.gene_loss_ep[1],
                'gene_adv_loss': self.gene_loss_ep[2],
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
        train_times_per_epoch = self.discriminator_train_config['train_times_per_epoch']  # 一回合最多训练轮数
        min_loss_per_epoch = self.discriminator_train_config['min_loss_per_epoch']  # 小于该loss则不再训练
        train_times = 0
        epoch_loss = min_loss_per_epoch
        d_his = AverageMeter()
        dv_his = AverageMeter()
        di_his = AverageMeter()
        while train_times < train_times_per_epoch and epoch_loss >= min_loss_per_epoch:
            d_history = AverageMeter()
            dv_history = AverageMeter()
            di_history = AverageMeter()
            with tqdm(total=num_iter) as train_Discriminator_bar:
                for index, data in enumerate(self.train_loader):
                    if torch.cuda.is_available():
                        for i in data:
                            data[i] = data[i].cuda()
                    generator_feats, discriminator_feats, confidence = self.fuse_model(data)
                    # {'Generator_1': [2,3,512,512]}, {d_1:(4,), d_2:{4,}, {d_1:(4,), d_2:{4,}}
                    d_loss, dv_loss, di_loss = self.d_loss_fn(generator_feats, discriminator_feats, confidence)
                    self.opt_discriminator.zero_grad()
                    d_loss.backward()
                    self.opt_discriminator.step()
                    d_history.update(d_loss.item())
                    dv_history.update(dv_loss.item())
                    di_history.update(di_loss.item())
                    train_Discriminator_bar.set_description('\tepoch:%s Train_D iter:%s loss:%.5f' %
                                                            (self.epoch, index, d_history.avg))
                    train_Discriminator_bar.update(1)

                train_times += 1
                self.disc_iter += 1
                self.runs.log({
                    'disc/total': d_history.avg,
                    'disc/vi': dv_history.avg,
                    'disc/ir': di_history.avg,
                    'disc/disc_iter': self.disc_iter
                })
                d_his.update(d_history.avg)
                dv_his.update(dv_history.avg)
                di_his.update(di_history.avg)
        self.disc_loss_ep = [d_his.avg, dv_his.avg, di_his.avg]

    def train_generator(self):
        for param in self.fuse_model.Generator.parameters():
            param.requires_grad = True
        for param in self.fuse_model.Discriminator.parameters():
            param.requires_grad = False
        num_iter = len(self.train_loader)
        train_times_per_epoch = self.generator_train_config['train_times_per_epoch']
        min_loss_per_epoch = self.generator_train_config['min_loss_per_epoch']
        train_times = 0
        epoch_loss = min_loss_per_epoch
        g_his = AverageMeter()
        g_con_his = AverageMeter()
        g_adv_his = AverageMeter()
        while train_times < train_times_per_epoch and epoch_loss >= min_loss_per_epoch:
            g_history = AverageMeter()
            g_con_history = AverageMeter()
            g_adv_history = AverageMeter()
            img_record = 0  # wandb保存融合结果标志位
            with tqdm(total=num_iter) as train_Generator_bar:
                for index, data in enumerate(self.train_loader):
                    if torch.cuda.is_available():
                        for i in data:
                            data[i] = data[i].cuda()
                    generator_feats, discriminator_feats, confidence = self.fuse_model(data)
                    # {'Generator_1': [2,3,512,512]}, {d_1:(4,), d_2:{4,}, {d_1:(4,), d_2:{4,}}
                    g_loss, g_con_loss, g_adv_loss = self.g_loss_fn(data, generator_feats, discriminator_feats, confidence)
                    self.opt_generator.zero_grad()
                    g_loss.backward()
                    self.opt_generator.step()
                    g_history.update(g_loss.item())
                    g_con_history.update(g_con_loss.item())
                    g_adv_history.update(g_adv_loss.item())

                    # 记录训练过程中的图像融合情况
                    if img_record < 5:
                        if self.datasets_config['color']:
                            vis_i, ir_i, fuse_i = debug_color(self.epoch, data, generator_feats)
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
                            Img = wandb.Image(img, caption="epoch:{}".format(self.epoch))
                            self.runs.log({
                                "fuse": Img
                            })
                            img_record += 1
                    train_Generator_bar.set_description('\tepoch:%s Train_G iter:%s loss:%.5f' %
                                                        (self.epoch, index, g_history.avg))
                    train_Generator_bar.update(1)

                train_times += 1
                self.gene_iter += 1
                self.runs.log({
                    'gene/total': g_history.avg,
                    'gene/con': g_con_history.avg,
                    'gene/adv': g_adv_history.avg,
                    'gene/gene_iter': self.gene_iter
                })
                g_his.update(g_history.avg)
                g_con_his.update(g_con_history.avg)
                g_adv_his.update(g_adv_history.avg)
        self.gene_scheduler.step()
        self.gene_loss_ep = [g_his.avg, g_con_his.avg, g_adv_his.avg]

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
    trainer = Trainer(project_name='GAN_G1_D2_COLOR_TST',
                      config_path='./config/GAN_G1_D2_color_spl_disc.yaml',
                      wandb_key=f'49deeeb7e29fb1acb9e77e00885bc52d739dee0f')
    trainer.runer()
