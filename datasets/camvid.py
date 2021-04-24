import os
import torch.utils.data as data
from torch.utils.data.dataset import T_co
from PIL import Image
import datasets.utils as utils
from collections import OrderedDict


class CamVid(data.Dataset):
    # Training dataset folders
    train_folder = 'train'
    train_folder_labeled = 'train_labels'
    # Validation dataset folders
    val_folder = 'val'
    val_folder_labeled = 'val_labels'
    # Test dataset folders
    test_folder = 'test'
    test_folder_labeled = 'test_labels'
    # File containing color encoding for each class
    class_labels_file = 'class_dict.csv'

    # TODO use class_dict here
    class_encoding = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        ('road_marking', (255, 69, 0)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabeled', (0, 0, 0))
    ])

    def __init__(self, root_dir, mode='train', image_transform=None, label_transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = image_transform
        self.label_transform = label_transform

        if self.mode.lower() == 'train':
            train_dir = os.path.join(root_dir, self.train_folder)
            train_dir_labeled = os.path.join(root_dir, self.train_folder_labeled)
            self.train_data = utils.get_files(train_dir)
            self.train_labels = utils.get_files(train_dir_labeled)
        elif self.mode.lower() == 'val':
            val_dir = os.path.join(root_dir, self.val_folder)
            val_dir_labeled = os.path.join(root_dir, self.val_folder_labeled)
            self.val_data = utils.get_files(val_dir)
            self.val_labels = utils.get_files(val_dir_labeled)
        elif self.mode.lower() == 'test':
            test_dir = os.path.join(root_dir, self.test_folder)
            test_dir_labeled = os.path.join(root_dir, self.test_folder_labeled)
            self.test_data = utils.get_files(test_dir)
            self.test_labels = utils.get_files(test_dir_labeled)
            print(utils.get_files(test_dir))
            print(utils.get_files(test_dir_labeled))
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test.")

    def __getitem__(self, index) -> T_co:
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[index]
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test.")
        img = Image.open(data_path)
        label = Image.open(label_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return img, label

    def __len__(self):
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test.")
