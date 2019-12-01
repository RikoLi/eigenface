#include "utils.h"

int main(int argc, char const *argv[]) {
    // Check arguments
    if (argc != 4) {
        cerr << "Wrong arguments!" << endl;
        cerr << "Usage: "<< argv[0] << " <energy_ratio> <model_save_name> <dataset_path>" << endl << endl;
        cerr << "Description:" << endl;
        cerr << "<energy_ratio>: Decide how many eigenfaces will be used." << endl;
        cerr << "<model_save_name>: Name of the output trained model." << endl;
        cerr << "<dataset_path>: Path of your face dataset." << endl << endl;
        cerr << "Example: "<< argv[0] <<" 0.95 trained_model ./dataset/" << endl;
        return 1;
    }
    double energy_ratio = atof(argv[1]);
    if (energy_ratio <= 0.0 || energy_ratio > 1.0) {
        cerr << "Invalid energy ratio, it should be a number in range (0.0, 1.0]!" << endl;
        return 1;
    }
    string model_save_path = argv[2];
    string dataset_path = argv[3];

    // Load images
    vector< pair<Mat, int> > train_vec;
    readAndAlign(dataset_path+"train/", train_vec);

    // Train eigenface
    cout << "--- Start training ---" << endl;
    trainEigenface(train_vec, model_save_path, energy_ratio);

    cout << "--- Training finished ---" << endl;
    return 0;
}
