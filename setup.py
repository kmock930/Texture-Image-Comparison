# from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt;
import os;
import shutil; # handle file operation: copying image files to another directory
import re; # regular expression library
import constants;
from collections import defaultdict;
import numpy as np;

# Function to recursively group files by image identifier
def group_files_by_identifier(directory):
    grouped_files = {};

    # Walk through all subdirectories and files, including 'set0', 'set1', 'set2'
    for root, _, files in os.walk(directory):
        for filename in files:
            if (filename.endswith(constants.ALLOWED_IMAGE_FORMATS)): # ensures it is an image
                if (filename.startswith(constants.CAL) or filename.startswith(constants.YOR)):
                    full_path = os.path.join(root, filename);
                    filename_split = re.split('_', re.split('-', filename)[0]);
                    category = filename_split[0];
                    id = filename_split[1];
                    if (grouped_files.__contains__(id) == False):
                        grouped_files[id] = [full_path];
                    else:
                        grouped_files[id].append(full_path);

    return grouped_files

def load_images(): 
    # initialize training sets and test sets for each category respectively
    yor_train_set = [];
    yor_train_labels = np.array([]);

    yor_test_set = [];
    yor_test_labels = np.array([]);

    cal_train_set = [];
    cal_train_labels = np.array([]);

    cal_test_set = [];
    cal_test_labels = np.array([]);

    # Group files for each category
    yor_groups = group_files_by_identifier(constants.YOR_DIR);
    cal_groups = group_files_by_identifier(constants.CAL_DIR);

    # Sort the groups to ensure consistency
    sorted_yor_ids = sorted(yor_groups.items(), key=lambda id: int(id[0]));
    sorted_cal_ids = sorted(cal_groups.items(), key=lambda id: int(id[0]));

    # put data into training and test sets respectively
    for item in sorted_yor_ids:
        if (int(item[0]) < constants.TEST_SIZE):
            yor_test_set += item[1];
        else:
            yor_train_set += item[1];
    
    for item in sorted_cal_ids:
        if (int(item[0]) < constants.TEST_SIZE):
            cal_test_set += item[1];
        else:
            cal_train_set += item[1];
    
    # Assign labels in all the sets
    yor_test_labels = np.full(len(yor_test_set), constants.YOR);
    yor_train_labels = np.full(len(yor_train_set), constants.YOR);
    cal_test_labels = np.full(len(cal_test_set), constants.CAL);
    cal_train_labels = np.full(len(cal_train_set), constants.CAL);

    return {
        # YOR
        "YOR_TEST_SAMPLES": yor_test_set,
        "YOR_TEST_LABELS": yor_test_labels,
        "YOR_TRAIN_SAMPLES": yor_train_set,
        "YOR_TRAIN_LABELS": yor_train_labels,
        # CAL
        "CAL_TEST_SAMPLES": cal_test_set,
        "CAL_TEST_LABELS": cal_test_labels,
        "CAL_TRAIN_SAMPLES": cal_train_set,
        "CAL_TRAIN_LABELS": cal_train_labels,
    };

def preprocess():
    # point processing

    # histogram equalization - Adaptive equalization

    # image sampling and reconstruction - https://medium.com/swlh/image-processing-with-python-digital-image-sampling-and-quantization-4d2c514e0f00

    # linear filtering python: https://scikit-image.org/skimage-tutorials/lectures/1_image_filters.html

    # cross correlation: https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_translation.html

    # box filter: https://scikit-image.org/docs/stable/api/skimage.filters.html

    # sobel edge filter: https://scikit-image.org/docs/stable/auto_examples/edges/plot_edge_filter.html

    # Gaussian blur: https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian

    # convolution: https://scikit-image.org/docs/stable/api/skimage.filters.html

    return

def save_fig(img_cat, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, img_cat + ".jpg")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout();
    plt.savefig(path, format='png', dpi=300);