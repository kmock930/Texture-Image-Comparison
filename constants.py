import os;

YOR = 'Yor';
CAL = 'Cal';
ALLOWED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png');

PROJECT_ROOT_DIR = ".";
DATASET_ROOT_DIR = 'stonefiles';
YOR_DIR = os.path.join(PROJECT_ROOT_DIR, DATASET_ROOT_DIR, YOR.lower());
CAL_DIR = os.path.join(PROJECT_ROOT_DIR, DATASET_ROOT_DIR, CAL.lower());
TEST_SIZE = 50;
