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

# MLP
num_classes = 2;
batch_size = 128;
num_epochs = 50;
mlp_learning_rate = 0.005;
input_shape_mlp = (int(img_height),);

optimizer_mlp = 'adam';
loss_mlp = 'binary_crossentropy';
activaton_mlp_tensor1 = 'relu';
activaton_mlp_tensor2 = 'sigmoid';

# CNN
input_shape_cnn = (img_height, img_width, 1); # for grayscale images
pool_size = 2;
strides = 2;
filter = 64;
cnn_kernel_size = (3,3);
activation1_cnn = "relu";
activation2_cnn = "sigmoid";
cnn_loss = "binary_crossentropy";
cnn_learning_rate=0.001;
cnn_steps_per_epoch = 8;
cnn_verbose = 1;

# evaluation
metrics = ['accuracy'];