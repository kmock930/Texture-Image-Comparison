import keras;
from keras import layers;


class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches * hidden_units, activation="gelu"),
                layers.Dense(units=hidden_units),
                layers.Reshape((-1, hidden_units)),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=hidden_units, activation="gelu"),
                layers.Dense(units=hidden_units),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        # Apply layer normalization.
        x = self.normalize(inputs)
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
        x_channels = keras.ops.transpose(x, axes=(0, 2, 1))
        # Apply mlp1 on each channel independently.
        mlp1_outputs = self.mlp1(x_channels)
        # Transpose mlp1_outputs from [num_batches, hidden_dim, num_patches] to [num_batches, num_patches, hidden_units].
        mlp1_outputs = keras.ops.transpose(mlp1_outputs, axes=(0, 2, 1))
        # Ensure matching shapes for addition
        if keras.ops.shape(inputs)[1] != keras.ops.shape(mlp1_outputs)[1]:
            # Adjust the shape of inputs to match mlp1_outputs
            inputs = keras.ops.reshape(inputs, (-1, keras.ops.shape(mlp1_outputs)[1], keras.ops.shape(inputs)[2]))

        # Add skip connection (ensure tensors have matching shapes)
        outputs = keras.ops.add(inputs, mlp1_outputs)
        # Apply layer normalization.
        outputs_patches = self.normalize(outputs)
        # Apply mlp2 on each patch independtenly.
        mlp2_outputs = self.mlp2(outputs_patches)
        # Ensure matching shapes for addition between mlp2_outputs and the input or previous layer
        if keras.ops.shape(outputs)[1] != keras.ops.shape(mlp2_outputs)[1]:
            # Adjust the shape of outputs to match mlp2_outputs
            outputs = keras.ops.reshape(outputs, (-1, keras.ops.shape(mlp2_outputs)[1], keras.ops.shape(outputs)[2]))

        # Now you can safely add or process mlp2_outputs
        outputs = keras.ops.add(outputs, mlp2_outputs)
        return outputs