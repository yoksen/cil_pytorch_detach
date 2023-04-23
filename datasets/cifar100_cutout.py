from torchvision import datasets, transforms
from datasets.idata import iData
import os
import numpy as np
from datasets.augmentations.cut_out import *

class CIFAR100_Cutout(iData):
    '''
    Dataset Name:   CIFAR-100 dataset (Canadian Institute for Advanced Research, 100 classes)
    Source:         A subset of the Tiny Images dataset.
    Task:           Classification Task
    Data Format:    32x32 color images.
    Data Amount:    60000 (500 training images and 100 testing images per class)
    Class Num:      100 (grouped into 20 superclass).
    Label:          Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = False
        self.img_size = img_size if img_size != None else 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])

        self.train_trsf = [
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            normalize(mean, std),
            cutout(mask_size=int(img_size / 2),
                   p=1,
                   cutout_inside=False),
            to_tensor(),
        ]
        self.test_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        self.common_trsf = []
        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(os.environ['DATA'], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(os.environ['DATA'], train=False, download=True)
        
        self.class_to_idx = train_dataset.class_to_idx

        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)