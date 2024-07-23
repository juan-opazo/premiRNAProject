"""
mfe - Minimum Free Energy kcal . mol ** -1
ss  - Secondary Structure
"""
import RNA
import numpy as np
import tensorflow as tf


def dot_bracket_to_pairing_matrix(dot_bracket):
    L = len(dot_bracket)
    matrix = np.zeros((L, L), dtype=int)
    stack = []

    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                matrix[i, j] = 1
                matrix[j, i] = 1

    return matrix


def one_hot_encode_sequence(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1]}
    one_hot = np.array([mapping[base] for base in sequence])
    return one_hot


def reshape_and_replicate_horizontally(one_hot):
    L, _ = one_hot.shape
    reshaped = one_hot.reshape(L, 1, 4)
    replicated = np.tile(reshaped, (1, L, 1))
    return replicated


def reshape_and_replicate_vertically(one_hot):
    L, _ = one_hot.shape
    reshaped = one_hot.reshape(1, L, 4)
    replicated = np.tile(reshaped, (L, 1, 1))
    return replicated

def sum_pooling(inputs):
    return tf.nn.pool(inputs, window_shape=[2, 2], pooling_type='AVG', strides=[2, 2], padding='VALID') * 4


def build_model(input_shape):
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    # Conv2D, ReLU, 2×2 sum-pooling
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.Lambda(sum_pooling))

    # Conv2D, ReLU, 2×2 max-pooling
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Conv2D, ReLU
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))

    # Downscaling to 4×4
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Reshape((1, 1, 128)))
    model.add(tf.keras.layers.UpSampling2D(size=(4, 4)))

    # Fully connected layer, sigmoid, output 256×1
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='sigmoid'))

    # Fully connected layer, output 2×1
    model.add(tf.keras.layers.Dense(2))

    # Softmax for classification
    model.add(tf.keras.layers.Softmax())

    return model


# seq = "AGACGACAAGGUUGAAUCGCACCCACAGUCUAUGAGUCGGUG"
# fc  = RNA.fold_compound(seq)
# (ss, mfe) = fc.mfe()

# print(f"{seq}\n{ss} ({mfe:6.2f})")


# Example sequence
sequence = "GCCCUUGGCA"
fc  = RNA.fold_compound(sequence)
(ss, mfe) = fc.mfe()
print(ss)

""" INPUT REPRESENTATION """

# One-hot encode the sequence
one_hot = one_hot_encode_sequence(sequence)

# Convert pairing matrix to L × L × 1
pairing_matrix_expanded = np.expand_dims(dot_bracket_to_pairing_matrix(ss), axis=2)

concatenated_matrix = np.concatenate(
    (pairing_matrix_expanded,
     reshape_and_replicate_horizontally(one_hot),
     reshape_and_replicate_vertically(one_hot)), axis=2)

print(concatenated_matrix)

""" NEURAL NETWORK """

# Example input shape: (L, L, 9)
L = concatenated_matrix.shape[0]  # You can change L to your specific value
input_shape = (L, L, 9)
model = build_model(input_shape)

model.summary()

print(concatenated_matrix.shape)