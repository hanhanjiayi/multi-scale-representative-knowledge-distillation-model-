import os, random
import argparse
import torch
import numpy as np
import torch.nn as nn

from cl_all_in_one import CLAIO
from models.FFA import FFA
from data.datasets import get_trainloader, get_testloader
from utils.utils import create_dir

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--task_order', type=str, nargs='+', required=False, default=['haze', 'rain', 'snow'])
parser.add_argument('--data_path', type=str, default='C:\\dataset', help='data path')
parser.add_argument('--logger_path', type=str, default='./Log', help='save train logger path')
parser.add_argument('--save_model_dir', type=str, default='./checkpoints/', help='save train model dir.pk')
parser.add_argument('--net', type=str, default='ffa')
parser.add_argument('--gps', type=int, default=3, help='residual_groups for ffa')
parser.add_argument('--blocks', type=int, default=20, help='residual_blocks for ffa')
parser.add_argument('--bs', type=int, default=1, help='batch size')
parser.add_argument('--crop', type=str, default=True)
parser.add_argument('--crop_size', type=int, default=240, help='Takes effect when using --crop ')
parser.add_argument('--contrastloss', type=str, default=True, help='Contrastive Loss')
parser.add_argument('--projector', type=str, default=True, help='Projector Loss')
parser.add_argument('--no_lr_sche', type=str, default=False, help='no lr cos schedule')
parser.add_argument('--exp_name', type=str, default='haze_rain_snow', help='no lr cos schedule')
parser.add_argument('--h_channels', type=int, default=16)
args = parser.parse_args()
args.logger_path = os.path.join(args.logger_path, args.exp_name)

if __name__ == '__main__':

    create_dir(args.logger_path)
    trainLogger = open(os.path.join(args.logger_path, 'train.log'), 'a+')
    device = torch.device(args.device)
    net = FFA(gps=args.gps, blocks=args.blocks).to(device)

    trainloader = get_trainloader(args)
    testloader = get_testloader(args)



    model = CLAIO(net,  trainloader, testloader, device, trainLogger, args)

    task_id=len(args.task_order)-1

    model.test(task_id)













