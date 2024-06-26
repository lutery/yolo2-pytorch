import numpy as np
import cv2


def imcv2_recolor(im, a=.1):
    '''
    对输入图像进行颜色扭曲和对比度调整，以达到图像增强的效果
    '''
    # t = [np.random.uniform()]
    # t += [np.random.uniform()]
    # t += [np.random.uniform()]
    # t = np.array(t) * 2. - 1.
    t = np.random.uniform(-1, 1, 3)

    # random amplify each channel
    # 进行颜色增强
    im = im.astype(np.float)
    im *= (1 + t * a)
    # 对比度调整
    mx = 255. * (1 + a)
    up = np.random.uniform(-1, 1)
    im = np.power(im / mx, 1. + up * .5)
    # return np.array(im * 255., np.uint8)
    return im


def imcv2_affine_trans(im):
    '''
    将图片进行所缩放、平移、翻转等操作，也就是图像增强

    param im: 原始图片数据
    '''
    # Scale and translate
    h, w, c = im.shape
    scale = np.random.uniform() / 10. + 1.
    max_offx = (scale - 1.) * w
    max_offy = (scale - 1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)

    im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
    im = im[offy: (offy + h), offx: (offx + w)]
    flip = np.random.uniform() > 0.5
    if flip:
        im = cv2.flip(im, 1)

    return im, [scale, [offx, offy], flip]
