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

We calculate the corvariance matrix of a small training set of face images and then apply eigenvalue decomposition on it. The results of the decomposition, eigenvectors, are considered as "eigenfaces" which build up a subspace of "face". That means that all faces can be represented as a linear combination of these "eigenfaces" whose weights are the square root of the eigenvalues respectively.

...

### Dataset

### Codes
