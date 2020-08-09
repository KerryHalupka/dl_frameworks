from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random
import numpy as np
import cv2
import tensorflow as tf
import os
import torch

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

def augment(im, im_size, random_affine_flag=1, random_perspective_flag=1, random_filter_flag=1, random_noise_flag=1, random_crop_flag=1, resize_flag=1):
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
    # print(im.shape)
    # print(im.max())
    im = 2*(im/im.max())-1
    return im





### other code
class TensorflowDataGenerator(): #tf.keras.utils.Sequence
    def __init__(self, train_dir, batch_size, validation = False, shuffle=True):
        train_dir = train_dir.decode()

        train_dogs = [f'{train_dir}/{i}' for i in os.listdir(train_dir) if 'dog' in i]  #get dog images
        train_cats = [f'{train_dir}/{i}' for i in os.listdir(train_dir) if 'cat' in i]  #get cat images
        # Combine dog and cat images, then shuffle them
        num_im=500
        all_imgs = train_dogs[:num_im] + train_cats[:num_im]  # slice the dataset and use 2000 in each class
        random.shuffle(all_imgs)  # shuffle it randomly
        if validation:
            self.model_imgs = all_imgs[np.int(0.9*len(all_imgs)):]
            self.affine_flag=0
            self.perspective_flag=0
            self.filter_flag=0
            self.noise_flag=0
            self.crop_flag=0
            self.resize_flag=1
            self.type='validation'
        else:
            self.model_imgs = all_imgs[:np.int(0.9*len(all_imgs))]
            self.affine_flag=1
            self.perspective_flag=1
            self.filter_flag=1
            self.noise_flag=1
            self.crop_flag=1
            self.resize_flag=1
            self.type='train'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.index = 0

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.model_imgs)

    def __next__(self):
        print(f'{self.type} index: {self.index}')
        batch = self.model_imgs[self.index * self.batch_size:(self.index + 1) * self.batch_size]
        X, y = self.__get_data(batch)
        self.index +=1
        if self.index>=len(self.model_imgs) // self.batch_size:
            self.index = 0
            self.on_epoch_end()
            print('next epoch')
            raise StopIteration
        return X, y
        
    def __iter__(self):
        return self

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.model_imgs) // self.batch_size

    def __get_data(self, batch):
        X=[]
        y=[]
        for file_name in batch:
            im = cv2.imread(file_name, cv2.IMREAD_COLOR)
            im = augment(im, random_affine_flag=self.affine_flag, 
                                random_perspective_flag=self.perspective_flag, 
                                random_filter_flag=self.filter_flag, 
                                random_noise_flag=self.noise_flag, 
                                random_crop_flag=self.crop_flag, 
                                resize_flag=self.resize_flag)
            X.append(im)
            if 'dog' in file_name:
                y.append(1)
            elif 'cat' in file_name:
                y.append(0)
        return np.asarray(X), np.asarray(y)
            

# tmp = KerasDataGenerator(train_dir,batch_size)
# for count,i in enumerate(tmp):
#     if count>3:
#         break
# plt.imshow(i[0][0,...])
# plt.show()
# print(i[1][0,...])
# plt.imshow(i[0][1,...])
# plt.show()
# print(i[1][1,...])
# plt.imshow(i[0][2,...])
# plt.show()
# print(i[1][2,...])
# plt.imshow(i[0][3,...])
# plt.show()
# print(i[1][3,...])

class TestDataGenerator(tf.keras.utils.Sequence): #
    def __init__(self, train_dir, batch_size, validation = False, shuffle=True):
        train_dogs = [f'{train_dir}/{i}' for i in os.listdir(train_dir) if 'dog' in i]  #get dog images
        train_cats = [f'{train_dir}/{i}' for i in os.listdir(train_dir) if 'cat' in i]  #get cat images
        # Combine dog and cat images, then shuffle them
        num_im=1000
        all_imgs = train_dogs[:num_im] + train_cats[:num_im]  # slice the dataset and use 2000 in each class
        random.shuffle(all_imgs)  # shuffle it randomly
        if validation:
            self.model_imgs = all_imgs[np.int(0.8*len(all_imgs)):]
            random_affine_flag=0
            random_perspective_flag=0
            random_filter_flag=0
            random_noise_flag=0
            random_crop_flag=0
            resize_flag=1
        else:
            self.model_imgs = all_imgs[:np.int(0.8*len(all_imgs))]
            random_affine_flag=1
            random_perspective_flag=1
            random_filter_flag=1
            random_noise_flag=1
            random_crop_flag=1
            resize_flag=1
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.model_imgs)

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.model_imgs) // self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        batch = self.model_imgs[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batch)
        return X, y

    def __get_data(self, batch):
        X=[]
        y=[]
        for file_name in batch:
            im = cv2.imread(file_name, cv2.IMREAD_COLOR)
            im = augment(im, random_affine_flag=random_affine_flag, 
                                random_perspective_flag=random_perspective_flag, 
                                random_filter_flag=random_filter_flag, 
                                random_noise_flag=random_noise_flag, 
                                random_crop_flag=random_crop_flag, 
                                resize_flag=resize_flag)
            X.append(im)
            if 'dog' in file_name:
                y.append(1)
            elif 'cat' in file_name:
                y.append(0)
        return np.asarray(X), np.asarray(y)
    # def load_val(self):
    #     X=[]
    #     y=[]
    #     for file_name in self.val_imgs:
    #         im = cv2.imread(file_name, cv2.IMREAD_COLOR)
    #         im = augment(im, random_affine_flag=0, random_perspective_flag=0, random_filter_flag=0, random_noise_flag=0, random_crop_flag=0, resize_flag=1)
    #         X.append(im)
    #         if 'dog' in file_name:
    #             y.append(1)
    #         elif 'cat' in file_name:
    #             y.append(0)
    #     return (np.asarray(X),np.asarray(y))



def image_generator(file_list, batch_size, augment_flag=True):
    epoch = 0
    while epoch<100:
        num = 0
        while num < (len(file_list)-batch_size):
            X=[]
            y=[]
            for file_name in file_list[num:(num+batch_size)]:
                im = cv2.imread(file_name, cv2.IMREAD_COLOR)
                if augment_flag:
                    im = augment(im, random_affine_flag=1, random_perspective_flag=1, random_filter_flag=1, random_noise_flag=1, random_crop_flag=1, resize_flag=1)
                X.append(im)
                if 'dog' in file_name:
                    y.append(1)
                elif 'cat' in file_name:
                    y.append(0)
            yield (np.asarray(X),np.asarray(y))
            num += batch_size
        epoch+=1


def image_load(file_list):
    X=[]
    y=[]
    for file_name in file_list:
        im = cv2.imread(file_name, cv2.IMREAD_COLOR)
        im = augment(im, random_affine_flag=0, random_perspective_flag=0, random_filter_flag=0, random_noise_flag=0, random_crop_flag=0, resize_flag=1)
        X.append(im)
        if 'dog' in file_name:
            y.append(1)
        elif 'cat' in file_name:
            y.append(0)
    return (np.asarray(X),np.asarray(y))


if __name__ == "__main__":
    pass