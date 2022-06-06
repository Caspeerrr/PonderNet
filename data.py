from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import torch
import pytorch_lightning as pl
from augmentations import gaussian_noise_transform, gaussian_blur_transform, contrast_transform, jpeg_transform, rotation_transform


class DataModule(pl.LightningDataModule):
    '''
        DataModule to hold the MNIST dataset. Accepts different transforms for train and test to
        allow for extrapolation experiments.

        Parameters
        ----------
        data_dir : str
            Directory where MNIST will be downloaded or taken from.

        train_transform : [transform] 
            List of transformations for the training dataset. The same
            transformations are also applied to the validation dataset.

        test_transform : [transform] or [[transform]]
            List of transformations for the test dataset. Also accepts a list of
            lists to validate on multiple datasets with different transforms.

        batch_size : int
            Batch size for both all dataloaders.
    '''

    def __init__(self, data_dir='./', train_transform=None, test_transform=None, batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        mean = (0.491, 0.482, 0.447)
        std  = (0.247, 0.243, 0.262)

        self.train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32, 32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
        self.val_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data (train/val and test sets)
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None, validation_size=5000):
        '''
            Called on each GPU separately - stage defines if we are
            at fit, validate, test or predict step.
        '''
        # we set up only relevant datasets when stage is specified
        if stage in [None, 'fit', 'validate']:
            mnist_train = CIFAR10(self.data_dir, train=True,
                                transform=(self.train_transform))
            mnist_val = CIFAR10(self.data_dir, train=True,
                                transform=(self.val_transform))

            self.mnist_train, _ = random_split(mnist_train, 
                                    lengths=[len(mnist_train) - validation_size, validation_size],
                                    generator=torch.Generator().manual_seed(42))
            _, self.mnist_val = random_split(mnist_val, 
                                  lengths=[len(mnist_val) - validation_size, validation_size],
                                  generator=torch.Generator().manual_seed(42))

        if stage == 'test' or stage is None:
            if self.test_transform is None or isinstance(self.test_transform, transforms.Compose):
                self.mnist_test = CIFAR10(self.data_dir,
                                        train=False,
                                        transform=(self.test_transform or self.default_transform))
            else:
                self.mnist_test = [CIFAR10(self.data_dir,
                                         train=False,
                                         transform=test_transform)
                                   for test_transform in self.test_transform]

    def train_dataloader(self):
        '''returns training dataloader'''
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)
        return mnist_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
        return mnist_val

    def test_dataloader(self):
        '''returns test dataloader(s)'''
        if isinstance(self.mnist_test, CIFAR10):
            return DataLoader(self.mnist_test, batch_size=self.batch_size)

        mnist_test = [DataLoader(test_dataset,
                                 batch_size=self.batch_size)
                      for test_dataset in self.mnist_test]
        return mnist_test


def get_transforms():
    mean = (0.491, 0.482, 0.447)
    std  = (0.247, 0.243, 0.262)

    test_transform = [transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])]

    for s in range(1, 6):

        # define transformations
        transform_gnt = transforms.Compose([
            transforms.ToTensor(),
            gaussian_noise_transform(s),
            transforms.Normalize(mean, std)
        ])

        transform_gbt = transforms.Compose([
            transforms.ToTensor(),
            gaussian_blur_transform(s),
            transforms.Normalize(mean, std)
        ])

        transform_ct = transforms.Compose([
            transforms.ToTensor(),
            contrast_transform(s),
            transforms.Normalize(mean, std)
        ])

        transform_jt = transforms.Compose([
            transforms.ToTensor(),
            jpeg_transform(s),
            transforms.Normalize(mean, std)
        ])

        transform_rt = transforms.Compose([
            rotation_transform(s),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_transform.extend([transform_gnt, transform_gbt, transform_ct, transform_jt, transform_rt])

    train_transform = None

    return train_transform, test_transform
