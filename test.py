import os
import sys
import time
import argparse
import numpy as np
import cv2
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from env import Env
from model import MyFcn
from pixel_wise_a2c import PixelWiseA2C
from utils import PSNR, SSIM, NMSE, DC, computePSNR, computeSSIM, computeNMSE

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MICCAI_data', type=str,
                        dest='dataset', help='which dataset to use')
    parser.add_argument('--gpu', default=[0, 1], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--model', type=str, help='pth file to load')

    return parser.parse_args()


def test(model, a2c, config, early_break=True, batch_size=None, verbose=False):
    if batch_size is None:
        batch_size = config.batch_size
    env = Env(config)
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if not os.path.exists('results/actions'):
        os.mkdir('results/actions')
    if not os.path.exists('results/action_distribution'):
        os.mkdir('results/action_distribution')


    from dataset import MRIDataset
    test_loader = torch.utils.data.DataLoader(
        dataset = MRIDataset(image_set='test', transform=(config.dataset=='fastMRI_data'), config=config),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    reward_sum = 0
    p_list = defaultdict(list)
    PSNR_dict = defaultdict(list)
    SSIM_dict = defaultdict(list)
    NMSE_dict = defaultdict(list)
    count = 0
    actions_prob = np.zeros((config.num_actions, config.episode_len))
    image_history = dict()

    for i, (ori_image, image, mask) in enumerate(test_loader):
        count += 1
        if early_break and count == 101: # test only part of the dataset
            break
        if count % 100 == 0:
            print('tested: ', count)

        ori_image = ori_image.numpy()
        image = image.numpy()
        previous_image = image.copy()
        env.reset(ori_image=ori_image, image=image) 

        for t in range(config.episode_len):
            if verbose:
                image_history[t] = image
            image_input = Variable(torch.from_numpy(image).cuda(), volatile=True)
            pi_out, v_out, p = model(image_input, flag_a2c=True)

            p = p.permute(1, 0).cpu().data.numpy()
            env.set_param(p)
            p_list[t].append(p)

            actions = a2c.act(pi_out, deterministic=True)
            #np.save('rebuttal/pi_out_' + str(j) + '.npy', pi_out.cpu().data.numpy())
            #np.save('rebuttal/actions_' + str(j) + '.npy', actions)
            last_image = image.copy()
            image, reward = env.step(actions)
            image = np.clip(image, 0, 1)

            reward_sum += np.mean(reward)

            actions = actions.astype(np.uint8)
            prob = pi_out.cpu().data.numpy()
            total = actions.size
            for n in range(config.num_actions):
                actions_prob[n, t] += np.sum(actions==n) / total

            for j in range(ori_image.shape[0]):
                if i > 0:
                    break
                for dd in ['results/actions/', 'results/action_distribution/']:
                    if not os.path.exists(dd+str(j)):
                        os.mkdir(dd+str(j))
                a = actions[j].astype(np.uint8)
                total = a.size
                canvas = last_image[j, 0].copy()
                unchanged_mask = np.abs(last_image[j, 0] - image[j, 0]) < (1 / 255)
                #np.save('rebuttal/unchanged_mask_' + str(j) + '_'+ str(j) + '.npy', unchanged_mask)
                for n in range(config.num_actions):
                    A = np.tile(canvas[..., np.newaxis], (1, 1, 3)) * 255
                    a_mask = (a==n) & (1 - unchanged_mask).astype(np.bool)
                    #print(A.shape, a_mask.dtype)
                    #print(a_mask.mean())
                    #input()
                    A[a_mask, 2] += 250
                    #print(A.shape, unchanged_mask.shape)
                    cv2.imwrite('results/actions/' + str(j) + '/' + str(n) + '_' + str(t) +'.bmp', A)
                cv2.imwrite('results/actions/' + str(t) + '_unchanged.jpg', np.abs(last_image[j, 0] - image[j, 0]) * 255 * 255)

        for j in range(image.shape[0]):
            if 'fastMRI' in config.dataset:
                mask_j = mask.numpy()[j]
                mask_j = np.tile(mask_j, (image.shape[2] ,1))
            else:
                mask_j = test_loader.dataset.mask
            image_with_DC = DC(ori_image[j, 0], image[j, 0], mask_j)
            image_with_DC = np.clip(image_with_DC, 0, 1)
            for k in range(2):
                key = ['wo', 'DC'][k]
                tmp_image = [image[j, 0], image_with_DC][k]
               # print(k, ori_image[j, 0].min(), previous_image[j, 0].min(), tmp_image.min())
                PSNR_dict[key].append(computePSNR(ori_image[j, 0], previous_image[j, 0], tmp_image)) 
                SSIM_dict[key].append(computeSSIM(ori_image[j, 0], previous_image[j, 0], tmp_image))
                NMSE_dict[key].append(computeNMSE(ori_image[j, 0], previous_image[j, 0], tmp_image))
                if verbose:
                    print(j, key, PSNR_dict[key][-1], SSIM_dict[key][-1], NMSE_dict[key][-1])

            cv2.imwrite('results/'+str(i)+'_'+str(j)+'.bmp', np.concatenate((ori_image[j, 0], mask_j, previous_image[j, 0], image[j, 0], image_with_DC, np.abs(ori_image[j, 0] - image[j, 0]) * 10), axis=1) * 255)
            if verbose:
                cv2.imwrite('results_time_steps/'+str(i)+'_'+str(j)+'.bmp', np.concatenate([image_history[jj][j, 0] for jj in range(config.episode_len)] + [image[j, 0], image_with_DC, ori_image[j, 0]], axis=1) * 255)

    print('actions_prob', actions_prob / count)

    for key in PSNR_dict.keys():
        PSNR_list, SSIM_list, NMSE_list = map(lambda x: x[key], [PSNR_dict, SSIM_dict, NMSE_dict])
        print('number of test images: ', len(PSNR_list))
        psnr_res = np.mean(np.array(PSNR_list), axis=0)
        ssim_res = np.mean(np.array(SSIM_list), axis=0)
        nmse_res = np.mean(np.array(NMSE_list), axis=0)
        
        print('PSNR', psnr_res)
        print('SSIM', ssim_res)
        print('NMSE', nmse_res)

    for t in range(config.episode_len):
        print('parameters at {}: '.format(t), np.mean(np.concatenate(p_list[t], axis=1), axis=1))

    avg_reward = reward_sum / (i + 1)
    print('test finished: reward ', avg_reward)

    return avg_reward, psnr_res, ssim_res


if __name__ == "__main__":
    args = parse()
    sys.path.append(args.dataset)
    from config import config

    torch.backends.cudnn.benchmark = True

    env = Env(config)
    model = MyFcn(config)
    model.load_state_dict(torch.load(args.model))
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()
    a2c = PixelWiseA2C(config)

    avg_reward, psnr_res, ssim_res = test(model, a2c, config, early_break=False, batch_size=50, verbose=True)
