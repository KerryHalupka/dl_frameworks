from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random
import numpy as np
import cv2
import os

def rand_range(a, b):
    '''
    a utility functio to generate random float values in desired range
    '''
    return np.random.rand() * (b - a) + a

def random_affine(im):
    '''
    wrapper of Affine transformation with random scale, rotation, shear and translation parameters
    '''
    tform = AffineTransform(scale=(rand_range(0.9, 1.1), rand_range(0.9, 1.1)),
                            rotation=rand_range(-0.25, 0.25),
                            shear=rand_range(-0.15, 0.15),
                            translation=(rand_range(-im.shape[0]//15, im.shape[0]//15), 
                                         rand_range(-im.shape[1]//15, im.shape[1]//15)))
    return warp(im, tform.inverse, mode='reflect')

def random_perspective(im):
    '''
    wrapper of Projective (or perspective) transform, from 4 random points selected from 4 corners of the image within a defined region.
    '''
    region = 1/6
    A = np.array([[0, 0], [0, im.shape[0]], [im.shape[1], im.shape[0]], [im.shape[1], 0]])
    B = np.array([[int(rand_range(0, im.shape[1] * region)), int(rand_range(0, im.shape[0] * region))], 
                  [int(rand_range(0, im.shape[1] * region)), int(rand_range(im.shape[0] * (1-region), im.shape[0]))], 
                  [int(rand_range(im.shape[1] * (1-region), im.shape[1])), int(rand_range(im.shape[0] * (1-region), im.shape[0]))], 
                  [int(rand_range(im.shape[1] * (1-region), im.shape[1])), int(rand_range(0, im.shape[0] * region))], 
                 ])

    pt = ProjectiveTransform()
    pt.estimate(A, B)
    return warp(im, pt, output_shape=im.shape[:2])

def random_crop(im):
    '''
    croping the image in the center from a random margin from the borders
    '''
    margin = 1/10
    start = [int(rand_range(0, im.shape[0] * margin)),
             int(rand_range(0, im.shape[1] * margin))]
    end = [int(rand_range(im.shape[0] * (1-margin), im.shape[0])), 
           int(rand_range(im.shape[1] * (1-margin), im.shape[1]))]
    return im[start[0]:end[0], start[1]:end[1]]

def random_intensity(im):
    '''
    rescales the intesity of the image to random interval of image intensity distribution
    '''
    return rescale_intensity(im,
                             in_range=tuple(np.percentile(im, (rand_range(0,10), rand_range(90,100)))),
                             out_range=tuple(np.percentile(im, (rand_range(0,10), rand_range(90,100)))))

def random_gamma(im):
    '''
    Gamma filter for contrast adjustment with random gamma value.
    '''
    return adjust_gamma(im, gamma=rand_range(0.5, 1.5))

def random_gaussian(im):
    '''
    Gaussian filter for bluring the image with random variance.
    '''
    return gaussian(im, sigma=rand_range(0, 5))
    
def random_filter(im):
    '''
    randomly selects an exposure filter from histogram equalizers, contrast adjustments, and intensity rescaler and applys it on the input image.
    filters include: equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid, gaussian
    '''
    filters = [equalize_adapthist, equalize_hist, adjust_log, adjust_sigmoid, random_gamma, random_gaussian, random_intensity]
    filt = random.choice(filters)
    return filt(im)
    
def resize(im, im_size):
    return cv2.resize(im, (im_size,im_size), interpolation=cv2.INTER_CUBIC)

def augment(im, im_size, random_affine_flag=0, random_perspective_flag=0, random_filter_flag=0, random_noise_flag=0, random_crop_flag=0, resize_flag=0):
    '''
    image augmentation by doing a sereis of transfomations on the image.
    '''
    im = im/255
    if random_affine_flag:
        im = random_affine(im)
    if random_perspective_flag:
        im = random_perspective(im)
    if random_filter_flag:
        im = random_filter(im)
    if random_noise_flag:
        im = random_noise(im)
    if random_crop_flag:
        im = random_crop(im)
    if resize_flag:
        im = resize(im, im_size)
    im = 2*(im/im.max())-1
    return im

if __name__ == "__main__":
    pass