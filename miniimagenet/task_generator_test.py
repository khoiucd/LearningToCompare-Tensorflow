# code is based on https://github.com/katerakelly/pytorch-maml and is modified from https://github.com/floodsung/LearningToCompare_FSL.git

import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def mini_imagenet_folders():
    train_folder = 'datas/miniImagenet/train'
    test_folder = 'datas/miniImagenet/test'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders


class MiniImagenetTask(object):

    def __init__(self, character_folders, num_classes, train_num,test_num):

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:

            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])
            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+test_num]

            self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
            self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]
        
    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])


class dataset():
    def __init__(self, task, num_per_class, split='train', shuffle=True):
        self.num_per_class = num_per_class
        self.task = task
        self.split = split
        self.shuffle = shuffle
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def getitem(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        label = np.zeros((self.task.num_classes,1))
        label[self.labels[idx],0] = 1
        return ((np.array(image)/255)-0.92206)/0.08426, label

    def generator(self):
        while(True):
            if self.split == 'train':
                num_inst = self.task.train_num
            else:
                num_inst = self.task.test_num

            if self.shuffle:
                batch = [[i+j*num_inst for i in np.random.permutation(num_inst)[:self.num_per_class]] for j in range(self.task.num_classes)]
            else:
                batch = [[i+j*num_inst for i in range(num_inst)[:self.num_per_class]] for j in range(self.task.num_classes)]
            batch = [item for sublist in batch for item in sublist]

            if self.shuffle:
                random.shuffle(batch)
            for idx in batch:
                yield self.getitem(idx)
    
