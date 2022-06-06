import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import argparse

from pondernet import PonderMNIST
from data import MNIST_DataModule, get_transforms
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

def main(beta, n, seed):
    # set seeds
    pl.seed_everything(seed)

    train_transform, test_transform = get_transforms()

    # initialize datamodule and model
    mnist = MNIST_DataModule(batch_size=BATCH_SIZE,
                            train_transform=train_transform,
                            test_transform=test_transform)
    model = PonderMNIST(n_hidden=N_HIDDEN,
                        n_hidden_cnn=N_HIDDEN_CNN,
                        n_hidden_lin=N_HIDDEN_LIN,
                        kernel_size=KERNEL_SIZE,
                        max_steps=MAX_STEPS,
                        lambda_p=n,
                        beta=beta,
                        lr=LR)

    # setup logger
    logger = WandbLogger(project='PonderNet', name='extrapolation', offline=False)
    logger.watch(model)

    trainer = Trainer(
        logger=logger,                      # W&B integration
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
    parser.add_argument('--beta')
    parser.add_argument('--n')
    parser.add_argument('--seed')
    args = parser.parse_args()

    main(float(args.beta), float(args.n), float(args.seed))
   