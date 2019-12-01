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
    cout << "Best matched person ID: " << labels.row(id).col(0) << endl;
    cout << "Best matched image ID: " << labels.row(id).col(1) << endl;
    cout << "--- End ---" << endl;
    return 0;
}
