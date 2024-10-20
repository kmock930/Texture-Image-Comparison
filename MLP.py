import keras;
from keras import layers;
from keras.layers import RandomFlip, RandomRotation
import constants;
from Patches import Patches;
from PositionEmbedding import PositionEmbedding;
from MLPMixerLayer import MLPMixerLayer;
import numpy as np;
from sklearn.preprocessing import LabelEncoder, StandardScaler;

class MLP:
    model: keras.Model;

    mlpmixer_blocks = keras.Sequential(
        [MLPMixerLayer(constants.num_patches, constants.embedding_dim, constants.dropout_rate) for _ in range(constants.num_blocks)]
    );

    def __init__(self):
        self.model = self.build_classifier(blocks=self.mlpmixer_blocks);

    data_augmentation = keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.1),
        ]
    );

    # https://keras.io/examples/vision/mlp_image_classification/    
    def build_classifier(self, blocks, positional_encoding=False):
        inputs = layers.Input(shape=constants.input_shape)
        # Augment data.
        augmented = self.data_augmentation(inputs)
        # Create patches.
        patches = Patches(constants.patch_size)(augmented)
        # Encode patches to generate a [batch_size, num_patches, embedding_dim] tensor.
        x = layers.Dense(units=constants.embedding_dim)(patches)
        if positional_encoding:
            x = x + PositionEmbedding(sequence_length=constants.num_patches)(x)
        # Process x using the module blocks.
        x = blocks(x)
        # Apply global average pooling to generate a [batch_size, embedding_dim] representation tensor.
        representation = layers.GlobalAveragePooling1D()(x)
        # Apply dropout.
        representation = layers.Dropout(rate=constants.dropout_rate)(representation)
        # Compute logits outputs.
        logits = layers.Dense(constants.num_classes)(representation)
        # Create the Keras model.
        return keras.Model(inputs=inputs, outputs=logits)
    
    # train the model
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        # Create Adam optimizer with weight decay.
        optimizer = keras.optimizers.AdamW(
            learning_rate=constants.learning_rate,
            weight_decay=constants.weight_decay,
        )
        # Compile the model.
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="acc"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
            ],
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
        )

        _, accuracy, top_5_accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

        # Return history to plot learning curves.
        return history
     
    def normalize(X_train, X_test, y_train, y_test):
        le = LabelEncoder(); # convert categorical labels into numeric representation
        y_train = le.fit_transform(y_train);
        y_test = le.fit_transform(y_test);

        X_train = X_train / (constants.img_height - 1); # normalize to the range [0,1]
        X_test = X_test / (constants.img_height - 1); # normalize to the range [0,1]
        
        # the shape of the image samples are not fitting a 4D tensor
        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, axis=-1);
        if len(X_test.shape) == 3:
            X_test = np.expand_dims(X_test, axis=-1);

        return X_train, y_train, X_test, y_test;