# Problem Statement
## Aim: The goal is to build a model which classifies the type of an unseen image as accurate as possible, by implementing, evaluating, and comparing amongst 2 different multi-layer perceptron Neural Networks.
## Project Preview
![image](https://github.com/user-attachments/assets/9ef4036d-4d4f-426e-b990-74e674f037d7)
![image](https://github.com/user-attachments/assets/487d3ef6-df68-405f-869a-f1d3d1cf431a)
- The MLP model
![image](https://github.com/user-attachments/assets/bc3b0452-0971-4345-b197-15a324800496)
![image](https://github.com/user-attachments/assets/47652f9e-51d0-4beb-9f53-651c82ab85f0)
- The CNN model
![image](https://github.com/user-attachments/assets/ea7016b5-6daa-4ee2-b3cb-2af1f19e6e2a)
![image](https://github.com/user-attachments/assets/55937742-af44-47d1-bb2e-93f454675ba0)
![image](https://github.com/user-attachments/assets/19d84f60-edcb-4c94-8fbf-e692ec106871)

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
