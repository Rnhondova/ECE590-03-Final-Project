from __future__ import print_function, division

import torch
import numpy as np
import torchvision
from torchvision import  transforms
import os
import copy
from PIL import Image


class LoadDataset(torch.utils.data.Dataset):

    def __init__(self,text_file,root_dir,
                  transform,train_or_val='train',
                  number_of_images_to_load=0):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
            train_or_val(string): classification into training 'T' or validation 'V'
        """
        if train_or_val == 'train':
          train_or_val_ = 'T'
        else:
          train_or_val_ = 'V'

        image_index = np.genfromtxt(text_file,dtype=['U50','U50','U50'], delimiter=',',names=True)
        class_lbls = image_index[image_index['subset']==train_or_val_]['class']

        if train_or_val == 'train' and number_of_images_to_load > 0 and len(class_lbls) > number_of_images_to_load:
          class_lbls = class_lbls[:number_of_images_to_load]
          self.classes = np.unique(image_index['class'][:number_of_images_to_load])
          self.name_frame = image_index[image_index['subset']==train_or_val_]['image'][:number_of_images_to_load]
        else:
          self.classes = np.unique(image_index['class'])
          self.name_frame = image_index[image_index['subset']==train_or_val_]['image']
        
        self.label_frame = np.zeros(len(class_lbls),dtype=np.int8)
        self.root_dir = root_dir
        self.transform = transform
        self.__reclassify_labels_to_int__(class_lbls)

    def __reclassify_labels_to_int__(self,labels):
        for class_ in self.classes:
          self.label_frame[labels==class_] = int(np.where(self.classes == class_)[0])
    
    def __reclassify_int_to_labels__():
        print('time')

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.name_frame[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')
        image = self.transform(image)
        labels = self.label_frame[idx]
        sample = {'image': image, 'labels': labels}

        return sample