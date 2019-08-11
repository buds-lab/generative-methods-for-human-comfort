# Outline

## Problem Statement
- Gathering thermal comfort related data (or in general user-generated data)
is costly and takes time.
- Hard to generalize over a big population of people, as well as obtain a similar
number of responses for each label/category (e.g. Thermal comfort labels, always
a predominance in the 'comfort' class and not the rest)

## Methods
- Try different generative models for augment datasets
    - GAN (and its variations: CGAN, WGAN, WCGAN, TGAN, TableGAN)
    - Autoencoders (and the variations: Adversarial, variational)

# Evaluation metrics
- Baseline: Original train and test set
- Train set and synthetic data as training set: Should increase the performance since classes would be more balanced
- Synthetic data as training set: Performance should be comparable with the baseline showing the synthetic set captures the same characteristics as the real train set
