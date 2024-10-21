from tensorflow.keras.layers import Dense;
import keras;
import constants;
import numpy as np;
from sklearn.preprocessing import LabelEncoder;
import math;
import joblib;
import matplotlib.pyplot as plt;
from datetime import datetime; # for timestamp

class MLP:
    model: keras.Model;

    def __init__(self):
        self.model = self.build_classifier();

    # https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
    def build_classifier(self):
        # baseline model
        model = keras.Sequential();
        # add tensors
        # Add a Flatten layer to convert (64, 64, 1) to (4096,)
        model.add(keras.layers.Flatten(input_shape=(constants.img_height, constants.img_width, 1)));

        layer1 = keras.layers.Dense(constants.img_height, input_shape=constants.input_shape_mlp, activation=constants.activaton_mlp_tensor1);
        model.add(layer1); #relu

        layer2 = keras.layers.Dense(round(math.log(2, constants.num_classes)), activation=constants.activaton_mlp_tensor2); #sigmoid
        model.add(layer2); #sigmoid

        #save the pretrained model (not built yet)
        joblib.dump(model, 'MLP-pretrained.pkl');

        model.build((None, constants.img_height, constants.img_width));
        
        # compile model
        model.compile(
            loss=constants.loss_mlp, 
            optimizer=constants.optimizer_mlp, 
            metrics=[
                keras.metrics.BinaryAccuracy(name="acc")
            ]
        );
        self.model = model;

        #save the compiled model
        joblib.dump(model, 'MLP_compiled.pkl');

        return model;
    
    # train the model
    # https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        print("-----started training MLP model-----");
        startTime = datetime.now();

        # Create Adam optimizer with weight decay.
        optimizer = keras.optimizers.Adam(
            learning_rate=constants.mlp_learning_rate
        )
        
        # Create a learning rate scheduler callback.
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5
        )
        # Create an early stopping callback.
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        # Fit the model.
        history = self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=constants.batch_size,
            epochs=constants.num_epochs,
            validation_split=0.1,
            callbacks=[early_stopping, reduce_lr],
            verbose=0,
        );

        # save the post-trained model
        joblib.dump(self.model, 'MLP_posttrained.pkl');

        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

        print("-----The MLP model is completedly trained-----");
        endTime = datetime.now();
        timeElapsed = endTime - startTime;
        print(f"Training Time Elapsed: {timeElapsed}");

        # Return history to plot learning curves.
        return history
     
    def normalize(self, X_train, X_test, y_train, y_test):
        le = LabelEncoder();  # convert categorical labels into numeric representation
        y_train = le.fit_transform(y_train);
        y_test = le.fit_transform(y_test);

        X_train = X_train / (constants.img_height - 1);  # normalize to the range [0,1]
        X_test = X_test / (constants.img_height - 1);  # normalize to the range [0,1]
        
        # Check if the image samples need to be reshaped to fit a 4D tensor
        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, axis=-1);
        if len(X_test.shape) == 3:
            X_test = np.expand_dims(X_test, axis=-1);

        return X_train, y_train, X_test, y_test;

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