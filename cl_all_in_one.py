import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
import time
import os
from PIL import Image
from models.FFA import FFA
from utils.metrics import psnr, ssim, rmse





class CLAIO():
    def __init__(self, net: FFA, trainloader, testloader, device, trainLogger, args) -> None:
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.trainLogger = trainLogger
       
        self.device = device
        self.args = args
        self.old_net = deepcopy(self.net)


    @torch.no_grad()
    def test(self, task_id):
        self.net.load_state_dict(torch.load(
            os.path.join(self.args.save_model_dir, self.args.exp_name, self.args.task_order[task_id],
                         'ours_snow_net_step500000.pth'),
            map_location=self.device))
        self.old_net.load_state_dict(torch.load(
            os.path.join(self.args.save_model_dir, self.args.exp_name, self.args.task_order[task_id-1],
                         'ours_rain_net_step500000.pth'),
            map_location=self.device))

        self.net.eval()

        ssim_eval = []
        psnr_eval = []
        rmse_eval = []
        print('')


        for t, loader in enumerate(self.testloader[:task_id + 1]):
            ori_ssims, ori_psnrs, ori_rmses = [], [], []
            ssims, psnrs, rmses = [], [], []
            ssims_stable, psnrs_stable, rmses_stable = [], [], []
            for idx, (inputs, targets) in enumerate(loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                pred = self.net(x1=inputs)
                pred1 = self.old_net(x1=inputs)


                ssim0 = ssim(inputs, targets).item()
                psnr0 = psnr(inputs, targets)
                rmse0 = rmse(inputs, targets)
                ori_ssims.append(ssim0)
                ori_psnrs.append(psnr0)
                ori_rmses.append(rmse0)

                ssim1 = ssim(pred, targets).item()
                psnr1 = psnr(pred, targets)
                rmse1 = rmse(pred, targets)
                ssims.append(ssim1)
                psnrs.append(psnr1)
                rmses.append(rmse1)

                ssim2 = ssim(pred1, targets).item()
                psnr2 = psnr(pred1, targets)
                rmse2 = rmse(pred1, targets)
                ssims_stable.append(ssim2)
                psnrs_stable.append(psnr2)
                rmses_stable.append(rmse2)

            print("===============ORIGINAL==================")
            print(f'psnr:{np.mean(ori_psnrs):.4f} |ssim:{np.mean(ori_ssims):.4f} |rmse:{np.mean(ori_rmses):.4f}')
            print(f'psnr:{np.mean(ori_psnrs):.4f}±{np.std(ori_psnrs):.4f} |ssim:{np.mean(ori_ssims):.4f}±{np.std(ori_ssims):.4f} |rmse:{np.mean(ori_rmses):.4f}±{np.std(ori_rmses):.4f}')
            print("===============PREDITION==================")
            print("---------main model-------")
            print(f'psnr:{np.mean(psnrs):.4f} |ssim:{np.mean(ssims):.4f} |rmse:{np.mean(rmses):.4f}')
            print(f'psnr:{np.mean(psnrs):.4f}±{np.std(psnrs):.4f} |ssim:{np.mean(ssims):.4f}±{np.std(ssims):.4f} |rmse:{np.mean(rmses):.4f}±{np.std(rmses):.4f}')
            print("--------stable model-------")
            print(
                f'psnr:{np.mean(psnrs_stable):.4f} |ssim:{np.mean(ssims_stable):.4f} |rmse:{np.mean(rmses_stable):.4f}')
            print(f'psnr:{np.mean(psnrs_stable):.4f}±{np.std(psnrs_stable):.4f} |ssim:{np.mean(ssims_stable):.4f}±{np.std(ssims_stable):.4f} |rmse:{np.mean(rmses_stable):.4f}±{np.std(rmses_stable):.4f}')

            ssim_eval.append(np.mean(ssims))
            psnr_eval.append(np.mean(psnrs))
            rmse_eval.append(np.mean(rmses))

        ssim_test = np.mean(ssim_eval)
        psnr_test = np.mean(psnr_eval)
        rmse_test = np.mean(rmse_eval)
        

        return ssim_test, psnr_test, rmse_test


