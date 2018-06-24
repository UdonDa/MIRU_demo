from model import Generator, Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import sys

class Solver(object):
    def __init__(self, data_loader, config):
        self.test_iters = config.test_iters
        self.model_save_dir = config.model_save_dir
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        # Generator
        self.g_conv_dim = config.g_conv_dim
        self.c_dim = config.c_dim
        self.g_repeat_num = config.g_repeat_num
        self.image_size = config.image_size
        self.g_lr = config.g_lr
        # Discriminator
        self.d_conv_dim = config.d_conv_dim
        self.d_repeat_num = config.d_repeat_num
        self.d_lr = config.d_lr
        self.device = torch.cuda.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = data_loader

        self.build_model()
        
    
    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir,
                              '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir,
                              '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(
            G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(
            D_path, map_location=lambda storage, loc: storage))

    def build_model(self):
        """Create a generator and a discriminator..."""
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.G.to(self.device)
        self.D.to(self.device)

        # # TODO: Multiple GPU
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
    
    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        # Set data loader.
        data_loader = self.data_loader

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(
                    "./", '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()),
                           result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))