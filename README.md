# Eigenface

Assignment 3 for Computer Vision, Winter, 2019, ZJU.

Eigenface algorithm implementation for face recognition in C++.

## Prerequisites

The code works only in Linux, here is my environment and dependencies:

- Ubuntu 16.04.4 LTS
- OpenCV 3.4.2
- cmake 3.5.1

To run the code, you should firstly compile and install OpenCV 3.4.2 (I use this version).

## Introduction

Eigenface recognition describes face images as feature vectors in a high dimension space, where the inter-dimensional correlations are eliminated with PCA method.

I calculate the corvariance matrix of a small training set of face images and then apply eigenvalue decomposition on it. The results of the decomposition, eigenvectors, are considered as "eigenfaces" which build up a subspace of "face". That means that all faces can be represented as a linear combination of these "eigenfaces" whose weights are the square root of the eigenvalues respectively.

We use the eigenfaces to implement a dimensionality reduction on training data, which compresses a face image to a vector with fewer dimensions. When there comes a new test image, we apply the same reduction process on it and then calculate the similarity between the test vector (from the test image) and all other training vectors compressed from training faces before.

### Dataset

- The ORL face database, Olivetti Research Laboratory in Cambridge, UK

The dataset provides with 40 people's faces images taken from 10 different angles in `pgm` format, with size of 92x112. I divide the dataset into two parts: training set and validation set, for eigenface training and testing. Previous 9 images of one person are added into training set and the last one is added into the validation set. In addition, I create a new person data of myself.

Each images is labeled with a `txt` file documenting the location of eyes. This file is used during the alignment process.

**About your own dataset:** The project is test on ORL dataset. If you want to construct your own face library, you may need to edit the source code to configure some parameters. Please refer to following steps:

1. Scale your images to 92x112.
2. Organize your dataset in this structure:
   - dataset_name/
     - train/
       - s1/
         - 1.png
         - 1.txt
         - ...
       - s2/
       - ...
     - val/
       - ...
  Make sure that each image is labeled by a eye location `txt` and they should be placed in the same folder.
3. In `utils.h`, change these settings:
   ```c++
   const 
   ```

### Usage

1. Clone this repo.
2. Build the source code with cmake. You may need to edit the `CMakeLists.txt` to hit your requirement. Make sure dependencies were installed correctly before.
3. Make the project and generate executable outputs `mytrain` and `mytest`.
4. Train a new model, here's an example:
   ```bash
   ./mytrain 0.99 model dataset/
   ```
5. A `json` model file will be generated. Refer to this example to apply recognition:
   ```bash
   ./mytest test.png model.json test_eye_location.txt
   ```