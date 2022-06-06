# Does the performance of PonderNet actually depend on pondering?
This repository contains the code used for the experiments in "Does the performance of PonderNet actually depend on pondering".


# Reproducing experiments
The experiment in the paper can be reproduced by running the following command. The argument specifies the number of random seeds you want to use.

```
sh run_extrapolation.sh 5
```


### Transformations
The different transformations tested with 5 different severities are:
- Gaussian noise
- Gaussian blur
- Contrast transform
- JPEG transform
- Rotation transform

Additional transformations can be added by implementing them in `augmentations.py`.
# Citation
If you use this code to produce results for your scientific publication, or if you share a copy or fork, please cite us in the following way:

```
@inproceedings{VisbeekSmitPondering,
  title={Does the performance of PonderNet actually depend on pondering?},
  author={Visbeek, Samantha and Smit, Casper},
  year={2022},
  organization={UvA}
}
```


