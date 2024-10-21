import keras.optimizers;
import tensorflow as tf;
from tensorflow.keras.layers import Dense;
import keras;
import constants;
import numpy as np;
from sklearn.preprocessing import LabelEncoder;
import math;
import joblib;
import matplotlib.pyplot as plt;
from datetime import datetime; # for timestamp

class CNN:
    model: keras.Model;

    def __init__(self):
        self.model = self.build_classifier();

    # https://towardsdatascience.com/10-minutes-to-building-a-cnn-binary-image-classifier-in-tensorflow-4e216b2034aa
    def build_classifier(self):
        tf.keras.layers.Conv2D(constants.filter, constants.cnn_kernel_size, activation=constants.activation1_cnn, input_shape=constants.input_shape_cnn);
        tf.keras.layers.MaxPooling2D(constants.pool_size, constants.strides);


        self.model = tf.keras.models.Sequential();

        for _ in range(3):
            self.model.add(tf.keras.layers.Conv2D(constants.filter, constants.cnn_kernel_size, activation=constants.activation1_cnn, input_shape=constants.input_shape_cnn));
            self.model.add(tf.keras.layers.MaxPooling2D(constants.pool_size, constants.strides));
        
        self.model.add(tf.keras.layers.Flatten());
        
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
        self.model.add(tf.keras.layers.Dense(round(math.log(2, constants.num_classes)), activation=constants.activation2_cnn));
        
        # save pretrained model
        joblib.dump(self.model, "CNN_prebuilt");

        # build the model
        self.model.compile(
            loss=constants.cnn_loss,
            optimizer=keras.optimizers.RMSprop(learning_rate=constants.cnn_learning_rate),
            metrics=[
                keras.metrics.BinaryAccuracy(name="acc")
            ]
        );

        # save the post-trained model
        joblib.dump(self.model, 'CNN_pretrained.pkl');
        
        return self.model;

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        print("-----started training CNN model-----");
        startTime = datetime.now();

        history = self.model.fit(
            x=X_train,
            y=y_train,
            steps_per_epoch=constants.cnn_steps_per_epoch,
            epochs=constants.num_epochs,
            verbose= constants.cnn_verbose
        );

        # save the post-trained model
        joblib.dump(self.model, 'CNN_posttrained.pkl');

        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

        print("-----The CNN model is completedly trained-----");
        endTime = datetime.now();
        timeElapsed = endTime - startTime;
        print(f"Training Time Elapsed: {timeElapsed}");

        return history;

    def predict(self, X_test: np.ndarray):
        print("-----Started Predicting test samples-----");
        startTime = datetime.now();

        y_pred = self.model.predict(X_test);
    
        endTime = datetime.now();
        timeElapsed = endTime - startTime;
        print("-----Prediction completed-----");
        print(f"Prediction Time Elapsed: {timeElapsed}");
    
        return y_pred;


    def plotLearningCurve(history):
        # Plot training & validation accuracy values
        plt.plot(history.history['acc']);
        plt.title('Learning Curve - Model vs Accuracy');
        plt.ylabel('Accuracy');
        plt.xlabel('Epoch');
        plt.legend(['Train'], loc='upper left');
        plt.show();