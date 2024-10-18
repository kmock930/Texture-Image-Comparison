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

# MLP
num_classes = 100;
input_shape = (32, 32, 3);
weight_decay = 0.0001;
batch_size = 128;
num_epochs = 1;  # Recommended num_epochs = 50
dropout_rate = 0.2;
image_size = 64;  # We'll resize input images to this size.
patch_size = 8;  # Size of the patches to be extracted from the input images.
num_patches = (image_size // patch_size) ** 2;  # Size of the data array.
embedding_dim = 256;  # Number of hidden units.
num_blocks = 4;  # Number of blocks.