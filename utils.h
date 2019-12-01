#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// Dataset parameters
const int PEOPLE_NUM = 40; // 40 people
const int TRAIN_IMG_NUM = 9; // 9 images to train for each person

// Mask parameters
const int MASK_WIDTH = 90; // Original width: 92
const int MASK_HEIGHT = 110; // Original height: 112
const double R_EYE_WIDTH_RATIO = 0.3, EYE_HEIGHT_RATIO = 0.4; // Relative locations of eyes of a masked face
const double PI = 3.14159265359;

// Functions
Mat cropForMask(const string &img_name, const string &location_name);
void readAndAlign(const string &dataset_path, vector< tuple<Mat, int, int> > &dst_vec);
Mat removeAvg(const Mat &src_mat, Mat &dst_mat);
int getBaseFacesNum(const Mat &eigenface_mat, const Mat &eigenvalue_mat, double energy_ratio);
void visualizeTopKFaces(const Mat &eigenface_mat);
void trainEigenface(const vector< tuple<Mat, int, int> > &train_img_vec, const string &model_save_name, double energy_ratio);

#endif // UTILS_H