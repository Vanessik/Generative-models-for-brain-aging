# Generative-models-for-brain-aging-and-gender-differences
Project devoted to studying interpretability of latent representations, derived from MRI brain data.

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
- ```./GAN experiment/models/gender_classifier_checkpoint.pth ``` -- trained on MRI data __Classifier__ ;
- ```./GAN experiment/models/generative_checkpoint.pth ``` -- trained on MRI data __Generator__;
- ```./GAN experiment/models/discriminator_checkpoint.pth ``` -- trained on MRI data __Discriminator__ model;
- ```./GAN experiment/models/encoder_checkpoint.pth``` -- trained on fake MRI data and latent representations __Encoder__;


## Data

In this project we use HCP data from https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release.

We work with __two types__ of data:

- Morphometric low dimensional data;
- Full size MRI images;

__MRI data__ could be downloaded from here https://drive.google.com/file/d/1vGPOyEaLsITt3xtjhPTAUV99-AvnCW6_/view and __labels for gender__ from https://drive.google.com/file/d/1u8gV_lzDYHKCim7F-7EHaSiNRVztKaQQ/view?usp=sharing.
__MRI labels for age__ you could find in ```./data/age.csv```.
Preprocessed __morphometry data__ could be found in ```./data/full_unrestricted.csv```.


## Results

### __Morphometry data__

- __Autoencoder__ with bottlenck = 80;

- __Logistic Regression__ for gender and age labels:

Build LR classifier both on initial data and obtained after encoder bottleneck vectors for gender and age labels.

| Label| Initial vectors | Bottlenck vectors |
|----------------|:---------:|----------------:|
| Gender | 0.919 | 0.856 |
| Age| 0.47 | 0.465 |

- __Cross validation__  on initial data and  bottleneck vectors for different models for age label:

__On initial data:__

| Method| Logistic Regression | Random Forest|K Nearest Neighbors| Decision Tree|
|----------------|:---------:|---------:|---------:|----------:|
| Multiclass | 0.475 | 0.479 |0.406|   0.408|
| OneVsRest|0.476| 0.48 |0.394 |     0.379|
| OneVsOne| 0.466 | 0.478 |0.4| 0.38 |

__On latent data:__

| Method| Logistic Regression | Random Forest|K Nearest Neighbors| Decision Tree|
|----------------|:---------:|--------:|---------:|---------:|
| Multiclass | 0.462 | 0.47 |0.396| 0.419|
| OneVsRest|0.467| 0.466 |0.404 |  0.418 |
| OneVsOne| 0.469 | 0.468 |0.4| 0.42 |

- __PCA__ with number of components=2 works well for initial and obtained latent vectors for gender, but for age returns bad results;



### __Full size MRI images__

- __Autoencoder__ with bottleneck=1000;

- __Logistic Regression__ for gender and age labels:

Logistic regression was trained on initial data, reshaped to 2D and on latent vectors, obtained after encoder with bottleneck 1000;

| Label| Initial vectors | Bottlenck vectors |
|----------------|:---------:|----------------:|
| Gender | 0.937 | 0.94|
| Age| 0.439 | 0.439 |


       

- __Generative adversarial model__ with 100 dimensional latent space.

![Alt-текст](https://github.com/Vanessik/Generative-models-for-brain-aging/blob/master/imgs/GAN_fake_slices.png "Generated brains")

Generated images and corresponding latent representations can be found by the following links: https://drive.google.com/file/d/115XMMUh4U4OP_FB_cImRwVVDpzAikdh1/view?usp=sharing and https://drive.google.com/file/d/115xUwUxlgzJahS53WxOzRH-_wDcOWDtn/view?usp=sharing


- __Encoder__ on generated data to fit latent representations of size 100.

MSE on validation: 0.003 (standard deviation of generator's distribution approximately equals to 0.3)

- __Classification for gender__ (more complex model)

ResNet-like neural network reached 94.6% accuracy on validation without augmentation and 93% with augmentation by gaussian noise. However, classificators trained on neither augmented nor original dataset didn't show reasonable predictions for synthetic dataset. So, the continuation of the experiment was meaningless.










