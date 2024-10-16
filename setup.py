# from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import os
import shutil # handle file operation: copying image files to another directory

# Global variables
PROJECT_ROOT_DIR = "."
DATASET_DIR = os.path.join(PROJECT_ROOT_DIR, 'stoneflies')
YOR_DIR = os.path.join(DATASET_DIR, 'yor')
CAL_DIR = os.path.join(DATASET_DIR, 'cal')
TEST_SIZE = 50

def load_images(dataset_dir, yor_dir, cal_dir, test_size): 

    # Group files for each category
    yor_groups = group_files_by_identifier(yor_dir)
    cal_groups = group_files_by_identifier(cal_dir)

    # Sort the groups
    sorted_yor_ids = sorted(yor_groups.keys(), key=lambda x: int(x.split('_')[1]))
    sorted_cal_ids = sorted(cal_groups.keys(), key=lambda x: int(x.split('_')[1]))

    # Select the first 50 unique images for the test set
    yor_test_ids = sorted_yor_ids[:TEST_SIZE]
    cal_test_ids = sorted_cal_ids[:TEST_SIZE]

    # Define test and train directories - for model to access
    test_yor_dir = os.path.join(dataset_dir, 'test', 'yor')
    test_cal_dir = os.path.join(dataset_dir, 'test', 'cal')
    train_yor_dir = os.path.join(dataset_dir, 'train', 'yor')
    train_cal_dir = os.path.join(dataset_dir, 'train', 'cal')

    # Copy test set files
    copy_files(yor_groups, yor_test_ids, yor_dir, test_yor_dir)
    copy_files(cal_groups, cal_test_ids, cal_dir, test_cal_dir)

    # Assign remaining files for training
    yor_train_ids = sorted_yor_ids[test_size:]
    cal_train_ids = sorted_cal_ids[test_size:]

    # Copy training set files
    copy_files(yor_groups, yor_train_ids, YOR_DIR, train_yor_dir)
    copy_files(cal_groups, cal_train_ids, CAL_DIR, train_cal_dir)

    print("Data Splitting completed.")

    # point processing

    # histogram equalization - Adaptive equalization

    # image sampling and reconstruction - https://medium.com/swlh/image-processing-with-python-digital-image-sampling-and-quantization-4d2c514e0f00

    # linear filtering python: https://scikit-image.org/skimage-tutorials/lectures/1_image_filters.html

    # cross correlation: https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_translation.html

    # box filter: https://scikit-image.org/docs/stable/api/skimage.filters.html

    # sobel edge filter: https://scikit-image.org/docs/stable/auto_examples/edges/plot_edge_filter.html

    # Gaussian blur: https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian

    # convolution: https://scikit-image.org/docs/stable/api/skimage.filters.html

    # 

def save_fig(img_cat, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, img_cat + ".jpg")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def group_files_by_identifier(directory):
    import re
    
    grouped_files = {} # dict

    # matching filenames with a regular expression
    pattern = re.compile(r'(yor|cal)_(\d+)_\d+\.(jpg|jpeg|png)', re.IGNORECASE)
    
    for root, _, files in os.walk(directory):
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                category, id_num, ext = match.groups()
                identifier = f"{category}_{id_num}"
                grouped_files[identifier].append(filename)
        return grouped_files

def copy_files(file_groups, selected_ids, source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        # create directory if not exists
        os.makedirs(dest_dir)
    for identifier in selected_ids:
        for file in file_groups[identifier]:
            shutil.copy(os.path.join(source_dir, file), os.path.join(dest_dir, file))