# Convolutional Neural Network
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# * Applying 32 filters[feature detectors] of size 3*3
# * Converting input images to 64*64[limited CPU :p] and 3 channels [R, G, B] (using TF backend)
# * Using the ReLU function for activation of neurons[to remove the negative values and have non-linearity]
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# * Adding max pooling to reduce the size of feature maps and size of flattened layer
# With max pooling we reduce complexity make it less computation intensive and preserve spatial features
# Generally, as a rule of thumb we take a 2*2 window to avoid loss of features
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# * Using the Dense function to make a fully connected layer
# * Using a thumb rule of selecting number of neurons between i/p and o/p nodes. [Trial and Error]
classifier.add(Dense(output_dim = 128, activation = 'relu'))
# * Using the sigmoid function since we have only two categories
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
# * Using the stochastic gradient descent optimizer - adam
# * Using cross entropy as cost function. Binary because we have only two categories. 
# categorical_crossentropy if we have more than two outcomes. Using cross-entropy due to it's logarithmic
# component
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# * Using image data generator since we do not have too many training images. So we will generate
# images using Keras ImageDataGenerator package by fliping, rotating etc, them. This will give us
# diverse and a lot more images to train on. [also called Image Augmentation]

# Image Augmentation will allow us to enrich our dataset without having to add so many extra images.
# Code below copied from the official Keras documentation [https://keras.io/preprocessing/image/]
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, # so that pixels have value between 0 and 1
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32, # weights will be updated after every 32 images
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)