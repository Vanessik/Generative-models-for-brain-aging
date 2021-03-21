# Generative-models-for-brain-aging
Exploring brain MRI: generative models for brain aging

## Prerequisites

Google Colab GPU is enough to run each experiment. Main prerequisites are:

1. pytorch
2. CUDA

## Repository structure




## Experiments

- ```./Autoencoder experiments/Autoencoder_morph.ipynb ``` -- __Autoencoder__ on morphometry data;
- ```./Autoencoder experiments/Autoencoder_mri.ipynb ``` -- __Autoencoder__ on full size MRI images;
- ```./GAN experiment/GAN_experiment_part1.ipynb ``` --  __GAN and Encoder__ on full size MRI images;
- ```./GAN experiment/GAN_experiment_part2.ipynb ``` --  __Classifier__ on syntetic images;

## Models


## Pretrained models

- ```./Autoencoder experiments/trained_models/autoencoder_morph ``` -- trained on morphometry data __Autoencoder__;
- ```./Autoencoder experiments/trained_models/autoencoder_mri ``` -- trained on MRI data __Autoencoder__;
- ```./GAN experiment/models/discriminator_checkpoint.pth ``` -- trained on MRI data __Discriminator__ model;
- ```./GAN experiment/models/encoder_checkpoint.pth``` -- trained on MRI data __Encoder__;
- ```./GAN experiment/models/gender_classifier_checkpoint.pth ``` -- trained on MRI data __Classifier__ ;
- ```./GAN experiment/models/generative_checkpoint.pth ``` -- trained on MRI data __Generator__;


## Data

In this project we use HCP data from https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release.

We work with __two types__ of data:

- Morphometric low dimensional data;
- Full size MRI images;

__MRI data__ could be downloaded from here https://drive.google.com/file/d/1vGPOyEaLsITt3xtjhPTAUV99-AvnCW6_/view and __labels for gender__ from https://drive.google.com/file/d/1u8gV_lzDYHKCim7F-7EHaSiNRVztKaQQ/view?usp=sharing.
__MRI labels for age__ you could find in ./data/age.csv
Preprocessed __morphometry data__ could be found in ./data/full_unrestricted.csv


## Results

### __Morphometry data__

- __Logistic Regression__ for gender

Logistic regression reached 93% accuracy on initial data and 86% accuracy on latent vectors, obtained after encoder with bottleneck 80;

- __Classifier__ for age

Attach scores on cross validation

- __PCA__ with number of components=2 works well for initial and obtained latent vectors for gender, but for age returns bad results;



### __Full size MRI images__

- __Autoencoder__ with bottleneck=1000

- __Logistic Regression__ for gender

Logistic regression reached 93% accuracy on initial data, reshaped to 2D and 93% accuracy on latent vectors, obtained after encoder with bottleneck 1000;

- __Logistic Regression__ for age
      
- __Generative adversarial model__ 

2a Trained GAN with 100 dimensional latent space.

Generated images and corresponding latent representations can be found by the following liks: https://drive.google.com/file/d/115XMMUh4U4OP_FB_cImRwVVDpzAikdh1/view?usp=sharing and https://drive.google.com/file/d/115xUwUxlgzJahS53WxOzRH-_wDcOWDtn/view?usp=sharing

2b Trained encoder on generated data to fit latent representations of size 100.

MSE on validation: 0.003 (std of generator's distribution approximately equals to 0.3)

2c Classification for gender (more complex model)

ResNet-like neural network reached 94.6% accuracy on validation without augmentation and 93% with augmentation by gaussian noise. However, classificators trained on neither augmented nor original dataset didn't show reasunable predictions for synthetic dataset. So, the continuation of the experiment was meaningless.










