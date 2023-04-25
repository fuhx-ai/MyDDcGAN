import os
import sys
import torch
from tqdm import tqdm
import torch.optim as optim
from core.model import FuseModel
from torchvision import transforms
from torch.utils.data import DataLoader
from core.utils import load_config, debug
from core.dataset.fusionDataset import FusionDataset
from core.loss import GeneratorLoss, DiscriminatorLoss


def train_discriminator(epoch, opt, datasets_generator, fuse_model, disc_loss_fn, discriminator_train_config):
    for i in fuse_model.Generator.parameters():
        i.requires_grad = False
    for i in fuse_model.Discriminator.parameters():
        i.requires_grad = True
    num_iter = len(datasets_generator)
    train_times_per_epoch = discriminator_train_config['train_times_per_epoch']
    min_loss_per_epoch = discriminator_train_config['min_loss_per_epoch']
    train_times = 0
    min_loss = min_loss_per_epoch
    while train_times < train_times_per_epoch and min_loss >= min_loss_per_epoch:
        all_loss = 0
        with tqdm(total=num_iter) as train_Discriminator_bar:
            for index, data in enumerate(datasets_generator):
                if torch.cuda.is_available():
                    for i in data:
                        data[i] = data[i].cuda()
                generator_feats, discriminator_feats, confidence = fuse_model(
                    data)  # {'Generator_1': [2,3,512,512]}, {d_1:(4,), d_2:{4,}, {d_1:(4,), d_2:{4,}}
                D_loss = disc_loss_fn(generator_feats, discriminator_feats, confidence)
                opt.zero_grad()
                D_loss.backward()
                opt.step()
                all_loss = all_loss + D_loss
                train_Discriminator_bar.set_description('\tepoch:%s Train_D iter:%s loss:%.5f' %
                                                        (epoch, index, all_loss / num_iter))
                train_Discriminator_bar.update(1)
            min_loss = all_loss / num_iter
            train_times += 1


def train_generator(epoch, opt, datasets_generator, fuse_model, generator_loss, generator_train_config):
    for i in fuse_model.Generator.parameters():
        i.requires_grad = True
    for i in fuse_model.Discriminator.parameters():
        i.requires_grad = False
    num_iter = len(datasets_generator)
    train_times_per_epoch = generator_train_config['train_times_per_epoch']
    min_loss_per_epoch = generator_train_config['min_loss_per_epoch']
    train_times = 0
    min_loss = min_loss_per_epoch
    while train_times < train_times_per_epoch and min_loss >= min_loss_per_epoch:
        all_loss = 0
        with tqdm(total=num_iter) as train_Generator_bar:
            for index, data in enumerate(datasets_generator):
                if torch.cuda.is_available():
                    for i in data:
                        data[i] = data[i].cuda()
                generator_feats, discriminator_feats, confidence = fuse_model(data)
                G_loss = generator_loss(data, generator_feats, discriminator_feats, confidence)
                opt.zero_grad()
                G_loss.backward()
                opt.step()
                all_loss = all_loss + G_loss
                debug(epoch, data, generator_feats,
                      mean=generator_train_config['mean'], std=generator_train_config['std'])
                train_Generator_bar.set_description('\tepoch:%s Train_G iter:%s loss:%.5f' %
                                                    (epoch, index, all_loss / num_iter))
                train_Generator_bar.update(1)
            min_loss = all_loss / num_iter
            train_times += 1


def runner():
    project_name = 'GAN_G1_D2_ConvDesc'

    try:
        os.mkdir(f'./weights/{project_name}/')
        os.mkdir(f'./weights/{project_name}/generator/')
        os.mkdir(f'./weights/{project_name}/discriminator/')
    except:
        pass

    config = load_config(f'./config/{project_name}.yaml')
    fuse_model = FuseModel(config)

    datasets_config = config['Dataset']
    generator_config = config['Generator']
    discriminator_config = config['Discriminator']
    base_train_config = config['Train']['Base']
    generator_train_config = config['Train']['Generator']
    discriminator_train_config = config['Train']['Discriminator']

    mean, std = datasets_config['mean'], datasets_config['std']
    input_size = datasets_config['input_size']
    train_dataloader = FusionDataset(root_dir=datasets_config['root_dir'], sensors=datasets_config['sensors'],
                                     transform=transforms.Compose([transforms.Resize((input_size, input_size)),
                                                                   transforms.ToTensor(),
                                                                   transforms.Normalize(mean, std)]))
    train_loader = DataLoader(train_dataloader, batch_size=base_train_config['batch_size'], shuffle=False)

    g_loss = GeneratorLoss(generator_config)
    d_loss = DiscriminatorLoss(discriminator_config)

    generator, discriminator = fuse_model.Generator.cuda(), fuse_model.Discriminator.cuda() \
        if torch.cuda.is_available() else (fuse_model.Generator, fuse_model.Discriminator)

    opt_generator = eval('optim.' + generator_train_config['opt'])(generator.parameters(), generator_train_config['lr'])
    opt_discriminator = eval('optim.' + discriminator_train_config['opt'])(discriminator.parameters(),
                                                                           discriminator_train_config['lr'])

    for epoch in range(1, base_train_config['epoch'] + 1):
        # exit()
        train_discriminator(epoch, opt_discriminator, train_loader, fuse_model, d_loss, discriminator_train_config)
        train_generator(epoch, opt_generator, train_loader, fuse_model, g_loss, generator_train_config)
        torch.save(generator.state_dict(), f'./weights/{project_name}/generator/generator_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'./weights/{project_name}/discriminator/discriminator_{epoch}.pth')


if __name__ == '__main__':
    runner()
