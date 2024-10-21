# from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt;
import os;
import re; # regular expression library
import constants;
import numpy as np;
from skimage.io import imread;
from skimage import exposure;
from skimage.transform import resize;
from skimage.color import rgb2gray;
from skimage.filters import gaussian, sobel;
import random;
from matplotlib import pyplot as plt;
import joblib;

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

'''
Load Image Paths
'''
def load_images_paths(): 
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

'''
Load actual images into a numpy array.
'''
def load_images(paths_list: list, category: str):
    # Randomize an index in each category to display image
    randInd = random.randint(0,len(paths_list)-1);

    images = np.empty((0, constants.img_height, constants.img_width));
    currInd = 0;
    for path in paths_list:   
        img = imread(path);
        img_array = np.array(img);

        if (currInd == randInd):
            displayImage(img_array, category, isBefore=True);

        img_array = preprocess(img_array);

        if (currInd == randInd):
            displayImage(img_array, category, isBefore=False);
        
        images = np.append(images, [img_array], axis=0);
        currInd += 1; # next image
    return images;

def preprocess(img_array: np.ndarray):
    # Resizing to make the image smaller in resolution
    img_array = setResolution(img_array);

    # Grayscale
    img_array = rgb2gray(img_array);

    # Normalize to [0,1] range
    img_array = normalize(img_array);

    # point processing - Gamma Correction
    # to adjust the brightness
    img_array = exposure.adjust_gamma(img_array, gamma=constants.gamma);

    # histogram equalization - Adaptive equalization
    img_array = exposure.equalize_hist(img_array);

    # sobel edge filter: https://scikit-image.org/docs/stable/auto_examples/edges/plot_edge_filter.html
    # for edge detection
    img_array = sobel(img_array);

    # Gaussian blur: https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian
    # for smooting
    img_array = gaussian(img_array, sigma=constants.gaussianSigma); # sigma = standard deviation for the kernel

    return img_array;

def normalize(img_array: np.ndarray):
    return img_array / 255.0;  # Normalize to [0, 1] range

def setResolution(img_array: np.ndarray):
    return resize(img_array, (constants.img_height, constants.img_width));

def displayImage(img_array: np.ndarray, category: str, isBefore):
    if (isBefore == True):
        plt.title(f"Sample Image in {category} category BEFORE being processed");
        plt.imshow(img_array);
    else:
        plt.title(f"Sample Image in {category} category AFTER being processed");
        plt.imshow(img_array, cmap="gray");
    
    plt.show();

# Load Images from .pkl files
def unpack(filepath):
    with open(filepath, 'rb') as file:
        return joblib.load(file);

def convertPredArray(y_pred: np.ndarray):
    y_new_pred = np.array([]);
    if (y_pred.ndim > 1):
        for prediction in y_pred:
            y_new_pred = np.append(y_new_pred, prediction[0]);
    return y_new_pred;