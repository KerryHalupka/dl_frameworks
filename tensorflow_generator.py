import random
import numpy as np
import cv2
import tensorflow as tf
import os

# custom augmentation functions
from image_augmentation import *

#### tensorflow data generator
class TensorflowDataGenerator(tf.keras.utils.Sequence): #
    def __init__(self, data_dir, batch_size, im_size=200, num_im=None, shuffle=True):
        train_dogs = [f'{data_dir}/{i}' for i in os.listdir(data_dir) if 'dog' in i]  #get dog images
        train_cats = [f'{data_dir}/{i}' for i in os.listdir(data_dir) if 'cat' in i]  #get cat images
        # Combine dog and cat images, then shuffle them
        if num_im:
            all_imgs = train_dogs[:np.int(num_im/2)] + train_cats[:np.int(num_im/2)]  #train_dogs + train_cats # #, slice the dataset and use 2000 in each class
        else:
            all_imgs = train_dogs + train_cats
        random.seed(0)
        random.shuffle(all_imgs)  # shuffle it randomly
        self.train_imgs = all_imgs[:np.int(0.85*len(all_imgs))]
        self.val_imgs = all_imgs[np.int(0.85*len(all_imgs)):]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.im_size = im_size

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.train_imgs)
    
    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.train_imgs) // self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        batch = self.train_imgs[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batch)
        return X, y

    def __get_data(self, batch):
        X=[]
        y=[]
        for file_name in batch:
            im = cv2.imread(file_name, cv2.IMREAD_COLOR)
            im = augment(im, im_size = self.im_size, 
                                random_affine_flag=0, 
                                random_perspective_flag=0, 
                                random_filter_flag=0, 
                                random_noise_flag=0, 
                                random_crop_flag=1, 
                                resize_flag=1)
            X.append(im)
            if 'dog' in file_name:
                y.append(1)
            elif 'cat' in file_name:
                y.append(0)
        return np.asarray(X), np.asarray(y).astype('float32').reshape((-1,1))
    def load_val(self):
        X=[]
        y=[]
        for file_name in self.val_imgs:
            im = cv2.imread(file_name, cv2.IMREAD_COLOR)
            im = augment(im, self.im_size, 
                                random_affine_flag=0, 
                                random_perspective_flag=0, 
                                random_filter_flag=0, 
                                random_noise_flag=0, 
                                random_crop_flag=0, 
                                resize_flag=1)
            X.append(im)
            if 'dog' in file_name:
                y.append(1)
            elif 'cat' in file_name:
                y.append(0)
        return (np.asarray(X), np.asarray(y).astype('float32').reshape((-1,1)))


class TensorflowDataGenerator_Test(tf.keras.utils.Sequence): #
    def __init__(self, data_dir, batch_size, im_size=200, shuffle=False):
        all_imgs = [f'{data_dir}/{i}' for i in os.listdir(data_dir)]
        random.seed(0)
        random.shuffle(all_imgs)  # shuffle it randomly
        self.shuffle = shuffle
        self.test_imgs = all_imgs
        self.batch_size = batch_size
        self.on_epoch_end()
        self.im_size = im_size

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.test_imgs)
    
    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.test_imgs) // self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indices of the batch
        batch = self.test_imgs[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__get_data(batch)
        return X 

    def __get_data(self, batch):
        X=[]
        for file_name in batch:
            im = cv2.imread(file_name, cv2.IMREAD_COLOR)
            im = augment(im, im_size = self.im_size, 
                                random_affine_flag=0, 
                                random_perspective_flag=0, 
                                random_filter_flag=0, 
                                random_noise_flag=0, 
                                random_crop_flag=1, 
                                resize_flag=1)
            X.append(im)
        return np.asarray(X)

if __name__ == "__main__":
    pass