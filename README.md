# Problem Statement
## Aim: The goal is to build a model which classifies the type of an unseen image as accurate as possible, by implementing, evaluating, and comparing amongst 2 different multi-layer perceptron Neural Networks.
## Datasets (from https://web.engr.oregonstate.edu/~tgd/bugid/stonefly9/)
1. YOR Dataset - containing 483 images of Yoraperla type.
2. CAL Dataset - containing 459 images of Calineuria type.

## Implementation Details
1. Please check my notebook with the name `CSI5341 Assignment 2 - Kelvin Mock 300453668` (in either `.pdf` or `.ipynb` format) to see the main flow of my analysis. 
2. **Multi-Layer Perceptron (MLP) model** - please check `MLP.py`.
3. **Convolutional Neural Network (CNN) model** - please check `CNN.py`.
4. Use of Keras API of Tensorflow
5. Involves some logic to laod images and to perform image preprocessing steps on them - please check `setup.py`.
6. Constants file which contains all kinds of configuration, such as the directories to the original datasets, size of the test set, and specific parameters for each neural network.
7. Models are exported into `.pkl files` during each stage of my modelling and analysis work. 
