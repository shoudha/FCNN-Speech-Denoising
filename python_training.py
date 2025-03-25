# Import necessary libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from scipy.io import loadmat

# Reading the targets and predictors data into python variables
targets = loadmat('targets_for_training.mat')
predictors = loadmat('predictors_for_training.mat')

targets_mod = targets['targets']
predictors_mod = predictors['predictors']

# Making the targets and predictors zeros mean and unit variance. This is automatically done in MATLAB
# by its input layer. But in keras, it is done manually.
targets_zc = (targets_mod - targets_mod.mean())/targets_mod.std()
predictors_zc = (predictors_mod - predictors_mod.mean())/predictors_mod.std()

# Reshaping the variables for input into Fully convolutional network
targets_for_NN = targets_zc.reshape(targets_zc.shape[1], targets_zc.shape[0], 1, 1)
predictors_for_NN = predictors_zc.reshape(predictors_zc.shape[1], predictors_zc.shape[0], 1, 1)

# NN parameters
input_frame_size = 320
batch_size = 128
epochs = 25
n_filter = 80
filter_size = [30, 1]

fcnn_model = tf.keras.models.Sequential(
    [        
     
        # Layer 1
        tf.keras.layers.Conv2D(input_shape=[input_frame_size, 1, 1], filters=n_filter, kernel_size=filter_size, padding='same'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.Activation('relu'),
        
        # Layer 2
        tf.keras.layers.Conv2D(filters=n_filter, kernel_size=filter_size, padding='same'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.Activation('relu'),
        
        # Layer 3
        tf.keras.layers.Conv2D(filters=n_filter, kernel_size=filter_size, padding='same'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.Activation('relu'),
        
        # Layer 4
        tf.keras.layers.Conv2D(filters=n_filter, kernel_size=filter_size, padding='same'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.Activation('relu'),
        
        # Layer 5
        tf.keras.layers.Conv2D(filters=n_filter, kernel_size=filter_size, padding='same'),
        tf.keras.layers.BatchNormalization(epsilon=1e-5),
        tf.keras.layers.Activation('relu'),
        
        # Last convolutional layer
        tf.keras.layers.Conv2D(filters=1, kernel_size=[input_frame_size, 1], padding='same'),
     ]
)

# Optimizer function
adam = tf.keras.optimizers.Adam(lr = 4e-4)

fcnn_model.compile(
    optimizer=adam,
    loss='mean_squared_error',
)

# Run the training
fcnn_model.fit(
    predictors_for_NN, 
    targets_for_NN,
    epochs = epochs,
    batch_size = batch_size,
)

# Save the model for it to be loaded in MATLAB and test
# Network for babble noise is saved as babble_model.h5
# Network for machinery noise is saved as mach_model.net
fcnn_model.save('babble_model.h5', overwrite=True)























