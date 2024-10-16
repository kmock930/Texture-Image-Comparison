import os

YOR = 'Yor'
CAL = 'Cal'
ALLOWED_IMAGE_FORMATS = 'jpg|jpeg|png'

PROJECT_ROOT_DIR = "."
DATASET_ROOT_DIR = 'stonefiles'
YOR_DIR = os.path.join(DATASET_ROOT_DIR, YOR)
CAL_DIR = os.path.join(DATASET_ROOT_DIR, CAL)
TEST_SIZE = 50
FILENAMES_REGEX = rf'({YOR}|{CAL})_(\d+)_\d+\.({ALLOWED_IMAGE_FORMATS})'