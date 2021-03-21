# Generative-models-for-brain-aging
Exploring brain MRI: generative models for brain aging

### Prerequisites

Google Colab GPU is enough to run each experiment. Main prerequisites are:

1. pytorch
2. CUDA

### Repository structure



### Experiments



### Models


### Pretrained models



### Data

In this project we use HCP data from https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release.

We work with two types of data:

- Morphometric low dimensional data;
- Full size MRI images;

MRI data could be downloaded from here https://drive.google.com/file/d/1vGPOyEaLsITt3xtjhPTAUV99-AvnCW6_/view and labels for gender from https://drive.google.com/file/d/1u8gV_lzDYHKCim7F-7EHaSiNRVztKaQQ/view?usp=sharing.
Labels for age you could find in ./data/age.csv
Preprocessed morphometry data could be found in ./data/restricted.csv


### Results

- Morphometry data

1a Classification for gender

Logistic regression reached 93% accuracy on initial data and 86% accuracy on latent vectors, obtained after encoder with bottleneck 80;

1b Classification for age


2 PCA with number of components=2 works well for initial and obtained latent vectors for gender, but for age returns bad results;

3 Feature importance for age and gender


- Full size MRI images

Autoencoder with bottleneck=1000

1a. Classification for gender

Logistic regression reached 93% accuracy on initial data, reshaped to 2D and 93% accuracy on latent vectors, obtained after encoder with bottleneck 1000;

1b. Classification for age











