# Does the performance of PonderNet actually depend on pondering?
This repository contains the code used for the experiments in "Does the performance of PonderNet actually depend on pondering".

## Requirements

Make sure to install the required packages with the correct version before trying to reproduce our results. This information can be found in `requirements.txt`.

## Reproducing experiments
The experiment in the paper can be reproduced by running the following command. The argument specifies the number of random seeds you want to use. `analyze.py` will also be run, which analyzes the output of the experiment and creates the plots.

```
sh run_extrapolation.sh 5
```

To optimize training time make sure to choose the right amount of gpus in `run_extrapolation.py`.


### Transformations
The different transformations tested with 5 different severities are:
- Gaussian noise
- Gaussian blur
- Contrast transform
- JPEG transform
- Rotation transform

Additional transformations can be added by implementing them in `augmentations.py`.

### Hyperparameters
All the hyperparameters are configured in `config.py`.

### Dataset
To choose a different dataset instead of `CIFAR10`, make sure to edit the dataset name in `run_extrapolation.py`, and to choose the correct input size depending on the image size in `pondernet.py`.
## Citation
If you use this code to produce results for your scientific publication, or if you share a copy or fork, please cite us in the following way:

```
@inproceedings{VisbeekSmitPondering,
  title={Does the performance of PonderNet actually depend on pondering?},
  author={Visbeek, Samantha and Smit, Casper},
  year={2022},
  organization={UvA}
}
```


