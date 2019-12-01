#include "utils.h"

Mat cropForMask(const string &img_name, const string &location_name) {
    Mat masked_face;
    Mat img = imread(img_name, IMREAD_GRAYSCALE);

    // Check if image file is opened
    if (img.empty()) {
        cerr << "Failed to read image file " << img_name;
        exit(EXIT_FAILURE);
    }

    ifstream location_reader;
    location_reader.open(location_name);

    // Check if location file is opened
    if (!location_reader.is_open()) {
        cerr << "Failed to read eye-location file " << location_name << endl;
        exit(EXIT_FAILURE);
    }

    // Get eyes locations
    int lx, ly, rx, ry;
    location_reader >> lx >> ly >> rx >> ry;
    location_reader.close();

    // Get masked area
    // Rotate to the same direction
    double rot_angle = atan(1.0 * (ly - ry) / (lx - rx)) * 180 / PI;

    // Take the distance between two eyes as scale factor
    double scale_factor = MASK_WIDTH * (1.0 - 2 * R_EYE_WIDTH_RATIO) / 
                            sqrt(
                                pow((lx - rx), 2) + pow((ly - ry), 2)
                            );

    // Get the rotation matrix
    Mat rot_mat = getRotationMatrix2D(
        Point2f(static_cast<float>((lx + rx) / 2.0), static_cast<float>((ly + ry) / 2.0)),
        rot_angle,
        scale_factor
    );

    // Align the rotation center to the new center of the image
    rot_mat.at<double>(0, 2) += MASK_WIDTH * 0.5 - (lx + rx) / 2.0;
    rot_mat.at<double>(1, 2) += MASK_HEIGHT * EYE_HEIGHT_RATIO - (ly + ry) / 2.0;

    // Warp the image and equalize the histogram
    warpAffine(img, masked_face, rot_mat, Size(MASK_WIDTH, MASK_HEIGHT), INTER_CUBIC);
    equalizeHist(masked_face, masked_face);

    return masked_face;
}

void readAndAlign(const string &dataset_path, vector< pair<Mat, int> > &dst_vec) {
    stringstream sample_path;
    sample_path << dataset_path;
    for (int i = 1; i <= PEOPLE_NUM; ++i) {
        for (int j = 1; j <= TRAIN_IMG_NUM; ++j) {
            sample_path << "s" << i << '/' << j;
            string img_name = sample_path.str() + ".pgm"; // Image file
            string location_name = sample_path.str() + ".txt"; // Eye location file

            // Read an image and crop for masked area
            Mat masked_face;
            masked_face = cropForMask(img_name, location_name);
            pair<Mat, int> tmp(masked_face, i);
            dst_vec.push_back(tmp);

            // Reset path
            sample_path.str("");
            sample_path << dataset_path;
        }
    }
}

Mat removeAvg(const Mat &src_mat, Mat &dst_mat) {
    Mat s = Mat::zeros(1, src_mat.cols, CV_64FC1);
    for (int i = 0; i < src_mat.rows; ++i) {
        s += src_mat.row(i);
    }

    // Get average feature matrix
    s /= src_mat.rows;

    Mat avg_mat = Mat::zeros(src_mat.rows, src_mat.cols, CV_64FC1);
    for (int i = 0; i < src_mat.rows; ++i) {
        s.copyTo(avg_mat.row(i));
    }
    dst_mat = src_mat - avg_mat; // Remove average value

    return s;
}

int getBaseFacesNum(const Mat &eigenface_mat, const Mat &eigenvalue_mat, double energy_ratio) {
    double s = sum(eigenvalue_mat).val[0];
    for (int i = 0; i < eigenvalue_mat.rows; ++i) {
        double top_i_sum = sum(eigenvalue_mat.rowRange(0, i+1)).val[0];
        double percentage = top_i_sum / s;
        if (percentage >= energy_ratio) {
            return i + 1;
        }
    }
}

void visualizeTopKFaces(const Mat &eigenface_mat) {
    // Image reconstruction
    Mat all_recon_faces = Mat::zeros(2*MASK_HEIGHT, 5*MASK_WIDTH, CV_8UC1);
    for (int i = 0; i < 10; ++i) {
        // Reshape to image size
        Mat recon_face;
        eigenface_mat.col(i).copyTo(recon_face);
        recon_face = recon_face.reshape(1, MASK_HEIGHT);

        // Normalize into 0-255
        double max_value, min_value;
        minMaxLoc(recon_face, &min_value, &max_value);
        recon_face = (recon_face - min_value) / (max_value - min_value) * 255.0;

        // Convert to 8-bit grayscale and save eigenfaces
        recon_face.convertTo(recon_face, CV_8UC1);
        recon_face.copyTo(
            all_recon_faces(
                Range((i / 5) * MASK_HEIGHT, (i / 5) * MASK_HEIGHT + MASK_HEIGHT),
                Range((i % 5) * MASK_WIDTH, (i % 5) * MASK_WIDTH + MASK_WIDTH)
            )
        );
    }
    imwrite("top_10_eigenfaces.png", all_recon_faces);
    cout << "Top 10 eigenfaces are visualized" << endl;
}

void trainEigenface(const vector< pair<Mat, int> > &train_img_vec, const string &model_save_name, double energy_ratio) {
    // Stack into feature vector
    Mat all_features;
    Mat labels;
    for (int i = 0; i < train_img_vec.size(); ++i) {
        Mat ft = train_img_vec[i].first.reshape(1, 1);
        int label = train_img_vec[i].second;
        ft.convertTo(ft, CV_64FC1);
        ft /= 255.0; // Squeeze in [0,1];
        all_features.push_back(ft);
        labels.push_back(label);
    }

    // Calculate eigenvalues and eigenvectors, using improved algorithm
    Mat T_t;
    Mat avg_img = removeAvg(all_features, T_t); // Remove average value
    Mat T = T_t.t(); // Transpose
    Mat eigen_value_mat, eigen_vec_mat;
    eigen(T_t*T, eigen_value_mat, eigen_vec_mat);
    Mat eigenface_mat = T * eigen_vec_mat.t();

    // Get top-k eigenfaces
    int k = getBaseFacesNum(eigenface_mat, eigen_value_mat, energy_ratio);
    cout << "Model generated with " << k << " eigenfaces" << endl;
    
    // Visualize top-10 eigenfaces
    visualizeTopKFaces(eigenface_mat.colRange(0, k));

    // Compress training data into subspace
    Mat sub_faces = eigenface_mat.colRange(0, k).t() * T; // sub_faces: (9900, k)
    
    // Save model
    FileStorage model_writer(model_save_name+".json", FileStorage::WRITE);
    
    // Chech if the model file is opened
    if (!model_writer.isOpened()) {
        cerr << "Failed to create eigenface model file!" << endl;
        exit(EXIT_FAILURE);
    }

    model_writer << "avg_img" << avg_img;
    model_writer << "labels" << labels;
    model_writer << "sub_faces" << sub_faces;
    model_writer << "transform_mat" << eigenface_mat.colRange(0, k).t();
    model_writer.release();
    cout << "Eigenface model is saved as " << model_save_name << ".json" << endl;
}