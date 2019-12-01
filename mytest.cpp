#include "utils.h"

int main(int argc, char const *argv[]) {
    // Check arguments
    if (argc != 4) {
        cerr << "Wrong arguments!" << endl;
        cerr << "Usage: "<< argv[0] << " <test_image> <model_path> <eyes_location_file>" << endl << endl;
        cerr << "Description:" << endl;
        cerr << "<test_image>: Test image that you want to recognize." << endl;
        cerr << "<model_path>: Path of the pretrained eigenface model." << endl;
        cerr << "<eyes_location_file>: File of eyes location of test image." << endl << endl;
        cerr << "Example: "<< argv[0] <<" test.png model.json test_eye_location.txt" << endl;
        return 1;
    }

    // Load model
    String model_name = argv[2];
    FileStorage model_loader(model_name, FileStorage::READ);

    // Chech if the model file is opened
    if (!model_loader.isOpened()) {
        cerr << "Failed to open " << model_name << " !" << endl;
        return 1;
    }

    // Load model parameters
    Mat avg_img = model_loader["avg_img"].mat();
    Mat labels = model_loader["labels"].mat();
    Mat sub_faces = model_loader["sub_faces"].mat();
    Mat transform_mat = model_loader["transform_mat"].mat();
    model_loader.release();

    // Normalize test image
    string img_name = argv[1];
    string test_eye_location = argv[3];
    Mat masked_test_face = cropForMask(img_name, test_eye_location);
    masked_test_face.reshape(1, 1).convertTo(masked_test_face, CV_64FC1);
    masked_test_face /= 255.0; // Squeeze to [0,1]
    masked_test_face -= avg_img;

    // Project to subspace
    Mat test_feature = transform_mat * masked_test_face.t();

    // Calculate loss
    vector<double> loss_vec;
    for (int i = 0; i < sub_faces.cols; ++i) {
        Mat diff = sub_faces.col(i) - test_feature;
        double loss = norm(diff, NORM_L2); // Use l2 norm
        loss_vec.push_back(loss);
    }

    // Get labels
    auto smallest = min_element(begin(loss_vec), end(loss_vec));
    int id = distance(begin(loss_vec), smallest);

    // Print results
    cout << "--- Best match ---" << endl;
    cout << "L2 loss: " << loss_vec[id] << endl;
    cout << "Best matched person ID: " << labels.at<int>(id, 0) << endl;
    cout << "Best matched image ID: " << labels.at<int>(id, 1) << endl;

    // Save matching result
    stringstream ss;
    ss << "../dataset/train/s" << labels.at<int>(id, 0) << "/";
    ss << labels.at<int>(id, 1) << ".pgm";
    Mat matched_face = imread(ss.str(), IMREAD_GRAYSCALE); // Load matched face in face library
    Mat test_face = imread(img_name, IMREAD_GRAYSCALE); // Load test face
    Mat concat_face;
    hconcat(test_face, matched_face, concat_face); // Concatenate two faces as an image
    imwrite("best_match.png", concat_face);
    cout << "Best matched result is saved." << endl;
    cout << "--- End ---" << endl;

    return 0;
}
