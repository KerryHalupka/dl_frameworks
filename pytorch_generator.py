import random
import numpy as np
import cv2
import torch
import torchvision

# custom augmentation functions
from image_augmentation import *

class PytorchDataGenerator(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, img_list, model_type = 'simple', validation = False, im_size=200):
        self.model_imgs = img_list
        print(f'num ims: {len(img_list)}')
        self.im_size = im_size
        self.model_type = model_type
        if validation:
            self.crop_flag=0
            self.resize_flag=1
            self.type='validation'
        else:
            self.crop_flag=1
            self.resize_flag=1
            self.type='train'
        
    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.model_imgs)

    def __getitem__(self, index):
        # Generate one sample of data
        file_name = self.model_imgs[index]
        im = cv2.imread(file_name, cv2.IMREAD_COLOR)
        X = augment(im, self.im_size, 
                    random_crop_flag=self.crop_flag, 
                    resize_flag=self.resize_flag)
        X = X.transpose((2, 0, 1)) #make image C x H x W
        X = torch.from_numpy(X)
        #normalise image only if using mobilenet
        if self.model_type=='mobilenet':
            X = torchvision.transforms.functional.normalize(X, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            pass
        # get labels
        if 'dog' in file_name:
            y=1
        elif 'cat' in file_name:
            y=0
        return X, torch.from_numpy(np.asarray(y))

class PytorchDataGenerator_Test(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, img_list, model_type = 'simple', im_size=200):
        self.model_imgs = img_list
        print(f'num ims: {len(img_list)}')
        self.im_size = im_size
        self.crop_flag=0
        self.resize_flag=1
        self.type='validation'
        
    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.model_imgs)

    def __getitem__(self, index):
        # Generate one sample of data
        file_name = self.model_imgs[index]
        im = cv2.imread(file_name, cv2.IMREAD_COLOR)
        X = augment(im, self.im_size, 
                    random_crop_flag=self.crop_flag, 
                    resize_flag=self.resize_flag)
        X = X.transpose((2, 0, 1)) #make image C x H x W
        X = torch.from_numpy(X)
        #normalise image only if using mobilenet
        if self.model_type=='mobilenet':
            X = torchvision.transforms.functional.normalize(X, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            pass
        return X

if __name__ == "__main__":
    pass