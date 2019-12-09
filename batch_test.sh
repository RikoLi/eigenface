#!/bin/bash
# Description: Batch processsing script for multi-face validation.
# Author: Li Jiachen

if [ $# != 2 ]
then
    echo -e "Wrong arguments!\n"
    echo -e "Usage: ./batch_test.sh <dataset_root> <model_path>\n"
    echo "Description:"
    echo "<dataset_root>: Root path of dataset, including training and validation data."
    echo -e "<model_path>: Eigenface model path.\n"
    echo "Example: ./batch_test.sh dataset/ model.json"
else
    ROOT=$1
    MODEL=$2
    TEST_PATH="${ROOT}val/"
    for PERSON in `ls $TEST_PATH`
    do
        echo "Test for $PERSON:"
        ./mytest ${TEST_PATH}${PERSON}/10.pgm $MODEL ${TEST_PATH}${PERSON}/10.txt
    done
fi