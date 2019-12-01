#include "utils.h"

int main(int argc, char const *argv[]) {
    // Check arguments
    if (argc != 4) {
        cerr << "Wrong arguments!" << endl;
        cerr << "Usage: "<< argv[0] << " <model_path> <test_image> <eyes_location_file>" << endl << endl;
        cerr << "Description:" << endl;
        cerr << "<model_path>: Path of the pretrained eigenface model." << endl;
        cerr << "<test_image>: Test image that you want to recognize." << endl;
        cerr << "<eyes_location_file>: File of eyes location of test image." << endl << endl;
        cerr << "Example: "<< argv[0] <<" model.json test.png test_eye_location.txt" << endl;
        return 1;
    }

    // Load model
    String model_name = argv[1];
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
    string img_name = argv[2];
    string test_eye_location = argv[3];
    Mat masked_test_face = cropForMask(img_name, test_eye_location);
    masked_test_face.reshape(1, 1).convertTo(masked_test_face, CV_64FC1);
    masked_test_face /= 255.0; // Squeeze to [0,1]
    masked_test_face -= avg_img;

    // Project to subspace
    Mat test_feature = transform_mat * masked_test_face.t();

    // Calculate similarity
    // int id = getSimilarity();

    // Calculate loss
    vector<double> loss_mat;
    for (int i = 0; i < sub_faces.cols; ++i) {
        Mat diff = sub_faces.col(i) - test_feature;
        double loss = norm(diff, NORM_L2);
        loss_mat.push_back(loss);
    }

    auto smallest = min_element(begin(loss_mat), end(loss_mat));
    int id = distance(begin(loss_mat), smallest);
    cout << labels.row(id) << endl;
    return 0;
}
