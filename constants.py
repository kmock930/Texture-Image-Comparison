import os;

YOR = 'Yor';
CAL = 'Cal';
ALLOWED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png');

PROJECT_ROOT_DIR = ".";
DATASET_ROOT_DIR = 'stonefiles';
YOR_DIR = os.path.join(PROJECT_ROOT_DIR, DATASET_ROOT_DIR, YOR.lower());
CAL_DIR = os.path.join(PROJECT_ROOT_DIR, DATASET_ROOT_DIR, CAL.lower());
TEST_SIZE = 50;

# preprocessing parameters
img_height, img_width = 64, 64;
gamma = 0.4;
gaussianSigma = 1;