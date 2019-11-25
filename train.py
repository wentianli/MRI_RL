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
from test import test

from utils import adjust_learning_rate
from utils import PSNR, SSIM, NMSE, DC, computePSNR, computeSSIM, computeNMSE

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MICCAI', type=str,
                        dest='dataset', help='to use dataset.py and config.py in which directory')
    parser.add_argument('--gpu', default=[0, 1], nargs='+', type=int,
                        dest='gpu', help='the gpu used')

    return parser.parse_args()


def train():
    torch.backends.cudnn.benchmark = False

    # load config
    args = parse()
    sys.path.append(args.dataset)
    from config import config
    assert config.switch % config.iter_size == 0
    time_tuple = time.localtime(time.time())
    log_dir = './logs/' + '_'.join(map(lambda x: str(x), time_tuple[1:4]))
    print('log_dir: ', log_dir)
    writer = SummaryWriter(log_dir)
    if not os.path.exists('model/'):
        os.mkdir('model/')

    # dataset
    from dataset import MRIDataset
    train_loader = torch.utils.data.DataLoader(
        dataset = MRIDataset(image_set='train', transform=True, config=config),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=True)

    env = Env(config)
    a2c = PixelWiseA2C(config)

    episodes = 0
    model = MyFcn(config)
    if len(config.resume_model) > 0: # resume training
        model.load_state_dict(torch.load(config.resume_model))
        episodes = int(config.resume_model.split('.')[0].split('_')[-1])
        print('resume from episodes {}'.format(episodes))
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()

    # construct optimizers for a2c and ddpg 
    parameters_wo_p = [value for key, value in dict(model.module.named_parameters()).items() if '_p.' not in key]
    optimizer = torch.optim.Adam(parameters_wo_p, config.base_lr)

    parameters_p = [value for key, value in dict(model.module.named_parameters()).items() if '_p.' in key]
    #parameters_pi = [value for key, value in dict(model.module.named_parameters()).items() if '_pi.' in key]
    params = [
        {'params': parameters_p, 'lr': config.base_lr},
    ]
    optimizer_p = torch.optim.SGD(params, config.base_lr)

    # training
    flag_a2c = True # if True, use a2c; if False, use ddpg
    while episodes < config.num_episodes:

        for i, (ori_image, image, _) in enumerate(train_loader):
            # adjust learning rate
            learning_rate = adjust_learning_rate(optimizer, episodes, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)
            _ = adjust_learning_rate(optimizer_p, episodes, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)

            ori_image = ori_image.numpy()
            image = image.numpy()
            env.reset(ori_image=ori_image, image=image) 
            reward = np.zeros((1))

            # forward
            if not flag_a2c:
                v_out_dict = dict()
            for t in range(config.episode_len):
                image_input = Variable(torch.from_numpy(image).cuda())
                reward_input = Variable(torch.from_numpy(reward).cuda())
                pi_out, v_out, p = model(image_input, flag_a2c, add_noise=flag_a2c)
                if flag_a2c:
                    actions = a2c.act_and_train(pi_out, v_out, reward_input)
                else:
                    v_out_dict[t] = - v_out.mean()
                    actions = a2c.act(pi_out, deterministic=True)
 
                p = p.cpu().data.numpy().transpose(1, 0)
                env.set_param(p)
                previous_image = image
                image, reward = env.step(actions)

                if not(episodes % config.display):
                    print('\na2c: ', flag_a2c)
                    print('episode {}: reward@{} = {:.4f}'.format(episodes, t, np.mean(reward)))
                    for k, v in env.parameters.items(): 
                        print(k, ' parameters: ', v.mean())
                    print("PSNR: {:.5f} -> {:.5f}".format(*computePSNR(ori_image[0, 0], previous_image[0, 0], image[0, 0])))
                    print("SSIM: {:.5f} -> {:.5f}".format(*computeSSIM(ori_image[0, 0], previous_image[0, 0], image[0, 0])))

                image = np.clip(image, 0, 1)


            # compute loss and backpropagate
            if flag_a2c:
                losses = a2c.stop_episode_and_compute_loss(reward=Variable(torch.from_numpy(reward).cuda()), done=True)
                loss = sum(losses.values()) / config.iter_size
                loss.backward()
            else:
                loss = sum(v_out_dict.values()) * config.c_loss_coeff / config.iter_size
                loss.backward()

            if not(episodes % config.display):
                print('\na2c: ', flag_a2c)
                print('episode {}: loss = {}'.format(episodes, float(loss.data)))

            # update model and write into tensorboard
            if not(episodes % config.iter_size):
                if flag_a2c:
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizer_p.zero_grad()
                else:
                    optimizer_p.step()
                    optimizer_p.zero_grad()
                    optimizer.zero_grad()
                    for l in v_out_dict.keys():
                        writer.add_scalar('v_out_{}'.format(l), float(v_out_dict[l].cpu().data.numpy()), episodes)

                for l in losses.keys():
                    writer.add_scalar(l, float(losses[l].cpu().data.numpy()), episodes)
                writer.add_scalar('lr', float(learning_rate), episodes)
                for k, v in env.parameters.items():
                    writer.add_scalar(k, float(v.mean()), episodes)

                if not(episodes % config.switch):
                    flag_a2c = not flag_a2c
                    if episodes < config.warm_up_episodes:
                        flag_a2c = True

            episodes += 1

            # save model
            if not(episodes % config.save_episodes):
                torch.save(model.module.state_dict(), 'model/' + '_'.join(map(lambda x: str(x), time_tuple[1:4])) + '_' + str(episodes) + '.pth')
                print('model saved')

            # test model
            if not(episodes % config.test_episodes):
                avg_reward, psnr_res, ssim_res = test(model, a2c, config, batch_size=10)
                writer.add_scalar('test reward', avg_reward, episodes)
                writer.add_scalar('test psnr', psnr_res[1], episodes)
                writer.add_scalar('test ssim', ssim_res[1], episodes)

            if episodes == config.num_episodes:
                writer.close()
                break

if __name__ == "__main__":
    train()
