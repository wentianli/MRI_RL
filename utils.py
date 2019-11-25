import numpy as np
import skimage.measure
import scipy


def fft_shift(x):
    fft = scipy.fftpack.fft2(x)
    fft = scipy.fftpack.fftshift(fft)
    return fft


def shift_ifft(fft):
    fft = scipy.fftpack.ifftshift(fft)
    x = scipy.fftpack.ifft2(fft)
    return x


def Downsample(x, mask):
    fft = scipy.fftpack.fft2(x)
    fft_good = scipy.fftpack.fftshift(fft)
    fft_bad = fft_good * mask
    fft = scipy.fftpack.ifftshift(fft_bad)
    x = scipy.fftpack.ifft2(fft)
#    x = np.abs(x)
    x = np.real(x)
    return x, fft_good, fft_bad


def SSIM(x_good, x_bad):
    assert len(x_good.shape) == 2
    ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
    return ssim_res


def PSNR(x_good, x_bad):
    assert len(x_good.shape) == 2
    psnr_res = skimage.measure.compare_psnr(x_good, x_bad)
    return psnr_res


def NMSE(x_good, x_bad):
    assert len(x_good.shape) == 2
    nmse_a_0_1 = np.sum((x_good - x_bad) ** 2)
    nmse_b_0_1 = np.sum(x_good ** 2)
    # this is DAGAN implementation, which is wrong
    nmse_a_0_1, nmse_b_0_1 = np.sqrt(nmse_a_0_1), np.sqrt(nmse_b_0_1)
    nmse_0_1 = nmse_a_0_1 / nmse_b_0_1
    return nmse_0_1


def computePSNR(o_, p_, i_):
    return PSNR(o_, p_), PSNR(o_, i_)


def computeSSIM(o_, p_, i_):
    return SSIM(o_, p_), SSIM(o_, i_)


def computeNMSE(o_, p_, i_):
    return NMSE(o_, p_), NMSE(o_, i_)


def DC(x_good, x_rec, mask):
    fft_good = fft_shift(x_good)
    fft_rec = fft_shift(x_rec)
    fft = fft_good * mask + fft_rec * (1 - mask)
    x = shift_ifft(fft)
    x = np.real(x)
    #x = np.abs(x)
    return x


def adjust_learning_rate(optimizer, iters, base_lr, policy_parameter, policy='step', multiple=[1]):
    '''
    source: https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/utils.py
    '''
    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['gamma'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['gamma'] * iters) ** (-policy_parameter['power']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['max_iter']) ** policy_parameter['power'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['gamma'] * (iters - policy_parameter['stepsize']))))
    elif policy == 'multistep-poly':
        lr = base_lr
        stepstart = 0
        stepend = policy_parameter['max_iter']
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
                stepstart = stepvalue
            else:
                stepend = stepvalue
                break
        lr = max(lr * policy_parameter['gamma'], lr * (1 - (iters - stepstart) * 1.0 / (stepend - stepstart)) ** policy_parameter['power'])

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr

if __name__ == "__main__":
    pass
