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
img_height, img_width = 64, 64; # assume equal (i.e., square image)
gamma = 0.4;
gaussianSigma = 1;
channels = 1; # grayscale

# MLP
num_classes = 2;
weight_decay = 0.0001;
batch_size = 128;
num_epochs = 50;
dropout_rate = 0.2;
patch_size = (img_height, img_width);  # Size of the patches to be extracted from the input images.
num_patches = int( (img_height // patch_size[0]) * ((img_width // patch_size[1])) );  # Size of the data array.
embedding_dim = 256;  # Number of hidden units.
num_blocks = 4;  # Number of blocks.
learning_rate = 0.005;
input_shape = (int(img_height),);

optimizer_mlp = 'adam';
loss_mlp = 'binary_crossentropy';
activaton_mlp_tensor1 = 'relu';
activaton_mlp_tensor2 = 'sigmoid';

# evaluation
metrics = ['accuracy'];