from torchvision import datasets, transforms
from datasets.idata import iData
import os
import numpy as np

class CIFAR100_Default(iData):
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

        self.train_trsf =[
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

        self.test_trsf = [
            transforms.Resize((img_size, img_size)),
        ]
        self.common_trsf = [ transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)]
        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(os.environ['DATA'], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(os.environ['DATA'], train=False, download=True)
        
        self.class_to_idx = train_dataset.class_to_idx

        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)