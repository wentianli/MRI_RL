import numpy as np
from numpy.random import randn
import torch
from torch.nn import Conv2d
import torch.nn.functional as F
from torch.autograd import Variable

class MyFcn(torch.nn.Module):
    def __init__(self, config):
        super(MyFcn, self).__init__()

        self.noise_scale = config.noise_scale
        self.num_parameters = len(config.parameters_scale)

        self.conv1 = Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = Conv2d(64, 64, kernel_size=3, stride=1, padding=4, dilation=4)

        self.conv5_pi = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_pi = Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_pi = Conv2d(64, config.num_actions, kernel_size=3, stride=1, padding=1)

        self.conv5_V = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_V = Conv2d(64 + self.num_parameters, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_V = Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        self.conv5_p = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_p = Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_p = Conv2d(64, self.num_parameters, kernel_size=3, stride=1, padding=1)

    def parse_p(self, u_out):
        p = torch.mean(u_out.view(u_out.shape[0], u_out.shape[1], -1), dim=2)
        return p

    def forward(self, x, flag_a2c=True, add_noise=False):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        if not flag_a2c:
            h = h.detach()

        # pi branch
        h_pi = F.relu(self.conv5_pi(h))
        h_pi = F.relu(self.conv6_pi(h_pi))
        pi_out = F.softmax(self.conv7_pi(h_pi), dim=1)

        # p branch
        p_out = F.relu(self.conv5_p(h))
        p_out = F.relu(self.conv6_p(p_out))
        p_out = self.conv7_p(p_out)
        if flag_a2c:
            if add_noise:
                p_out = p_out.data + torch.from_numpy(randn(p_out.shape[0], p_out.shape[1], 1, 1).astype(np.float32)).cuda() * self.noise_scale
                p_out = Variable(p_out)
            else:
                p_out = p_out.detach()
        p_out = F.sigmoid(p_out)

        # V branch
        h_v = F.relu(self.conv5_V(h))
        h_v = torch.cat((h_v, p_out), dim=1)
        h_v = F.relu(self.conv6_V(h_v))
        v_out = self.conv7_V(h_v)
       
        return pi_out, v_out, self.parse_p(p_out)
