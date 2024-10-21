import tensorflow as tf
from tensorflow.keras.layers import Dense

def test_keras():
    layer = Dense(10)
    print("Keras import successful, created Dense layer:", layer)

test_keras()
