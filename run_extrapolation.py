import pytorch_lightning as pl
from pytorch_lightning import Trainer
import argparse

from pondernet import PonderMNIST
from data import DataModule, get_transforms
from torchvision.datasets import MNIST, CIFAR10

from config import(
    BATCH_SIZE,
    EPOCHS,
    LR,
    GRAD_NORM_CLIP,
    N_HIDDEN,
    N_HIDDEN_CNN,
    N_HIDDEN_LIN,
    KERNEL_SIZE,
    MAX_STEPS,
    LAMBDA_P,
    BETA
)

def main(seed, data):
    # set seeds
    pl.seed_everything(seed)

    test_transform = get_transforms()

    # initialize datamodule and model
    mnist = DataModule(batch_size=BATCH_SIZE,
                            test_transform=test_transform,
                            dataset=data)
    model = PonderMNIST(n_hidden=N_HIDDEN,
                        n_hidden_cnn=N_HIDDEN_CNN,
                        n_hidden_lin=N_HIDDEN_LIN,
                        kernel_size=KERNEL_SIZE,
                        max_steps=MAX_STEPS,
                        lambda_p=LAMBDA_P,
                        beta=BETA,
                        lr=LR)


    trainer = Trainer(
        gpus=-1,                            # use all available GPU's
        max_epochs=EPOCHS,                  # maximum number of epochs
        gradient_clip_val=GRAD_NORM_CLIP,   # gradient clipping
        val_check_interval=0.25,            # validate 4 times per epoch
        deterministic=False)                 # for reproducibility
        # precision=16,                       # train in half precision

    # fit the model
    trainer.fit(model, datamodule=mnist)

    # evaluate on the test set
    trainer.test(model, datamodule=mnist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed')
    parser.add_argument('--data')
    args = parser.parse_args()

    if args.data == 'cifar10':
        data = CIFAR10
    elif args.data == 'MNIST':
        data = MNIST

    main(float(args.seed), data)
   