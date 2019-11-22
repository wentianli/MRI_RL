import numpy as np
import sys
import cv2
import torch
from skimage.measure import compare_ssim

class Env():
    def __init__(self, config):
        self.image = None
        self.previous_image = None

        self.num_actions = config.num_actions
        self.actions = config.actions

        self.parameters_scale = config.parameters_scale
        self.parameters = dict()
        self.set_param([0.5] * len(self.parameters_scale))

        self.reward_method = config.reward_method 
    
    def reset(self, ori_image, image):
        self.ori_image = ori_image
        self.image = image
        self.previous_image = None
        return

    def set_param(self, p):
        for i, k in enumerate(sorted(self.parameters_scale.keys())):
            self.parameters[k] = p[i] * self.parameters_scale[k]
        return

    def step(self, act):
        self.previous_image = self.image.copy()

        canvas = [np.zeros(self.image.shape, self.image.dtype) for _ in range(self.num_actions + 1)]
        b, c, h, w = self.image.shape
        for i in range(b):
            canvas[0][i, 0] = self.image[i,0]
            canvas[self.actions['subtraction']][i, 0] = self.image[i,0] - 3. / 255

            if np.sum(act[i] == self.actions['box']) > 0:
                canvas[self.actions['box']][i, 0] = cv2.boxFilter(self.image[i,0], ddepth=-1, ksize=(5,5))

            if np.sum(act[i] == self.actions['bilateral']) > 0:
                canvas[self.actions['bilateral']][i, 0] = cv2.bilateralFilter(self.image[i,0], d=5, sigmaColor=0.1, sigmaSpace=5)

            if True:
                canvas[self.actions['Gaussian']][i, 0] = cv2.GaussianBlur(self.image[i,0], ksize=(5,5), sigmaX=0.5)

            if np.sum(act[i] == self.actions['median']) > 0:
                canvas[self.actions['median']][i, 0] = cv2.medianBlur(self.image[i,0], ksize=5)

            if np.sum(act[i] == self.actions['Laplace']) > 0:
                p = self.parameters['Laplace'][i]
                k = np.array([[0, -p, 0], [-p, 1 + 4 * p, -p], [0, -p, 0]])
                canvas[self.actions['Laplace']][i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)

            if np.sum(act[i] == self.actions['unsharp']) > 0:
                amount = self.parameters['unsharp'][i]
                canvas[self.actions['unsharp']][i, 0] = self.image[i, 0] * (1 + amount) - canvas[self.actions['Gaussian']][i, 0] * amount

            if np.sum(act[i] == self.actions['Sobel_v1']) > 0:
                p = self.parameters['Sobel_v1'][i]
                k = np.array([[p, 0, -p], [2 * p, 1, -2 * p], [p, 0, -p]])
                canvas[self.actions['Sobel_v1']][i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)

            if np.sum(act[i] == self.actions['Sobel_v2']) > 0:
                p = self.parameters['Sobel_v2'][i]
                k = np.array([[-p, 0, p], [-2 * p, 1, 2 * p], [-p, 0, p]])
                canvas[self.actions['Sobel_v2']][i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)

            if np.sum(act[i] == self.actions['Sobel_h1']) > 0:
                p = self.parameters['Sobel_h1'][i]
                k = np.array([[-p,-2 * p,-p], [0, 1, 0], [p, 2 * p, p]])
                canvas[self.actions['Sobel_h1']][i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)

            if np.sum(act[i] == self.actions['Sobel_h2']) > 0:
                p = self.parameters['Sobel_h2'][i]
                k = np.array([[p, 2 * p, p], [0, 1, 0], [-p, -2 * p, -p]])
                canvas[self.actions['Sobel_h2']][i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)

        for a in range(1, self.num_actions + 1):
            self.image = np.where(act[:,np.newaxis,:,:] == a, canvas[a], self.image)
        self.image = np.clip(self.image, 0, 1)

        if self.reward_method == 'square':
            reward = np.square(self.ori_image - self.previous_image) * 255 - np.square(self.ori_image - self.image) * 255
        elif self.reward_method == 'abs':
            reward = np.abs(self.ori_image - self.previous_image) * 255 - np.abs(self.ori_image - self.image) * 255

        return self.image, reward 
