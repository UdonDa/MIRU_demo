import os
import argparse

import cv2
import torch
from src.data_loader import get_loader
from src.solver import Solver

def main(config):

    # Data Loader
    data_loader = get_loader(config.image_dir, None, None,
                                 config.image_crop_size, config.image_size, config.batch_size, config.num_workers)
    solver = Solver(data_loader, config)


    cap = cv2.VideoCapture(1)
    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            # To stop camera, press Escape key.
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--image_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')

    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--c_dim', type=int, default=10, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--d_lr', type=float, default=0.0001,help='learning rate for D')

    parser.add_argument('--rafd_image_dir', type=str, default='data/test')

    config = parser.parse_args()
    main(config)