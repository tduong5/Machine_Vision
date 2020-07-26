# Building Convolutional Neural Networks to Classify the Dog and Cat Images. This is a Binary Classification Model i.e. 0 or 1
# Used Dataset -- a Subset (10,000) Images ==> (8,000 for training_set: 4,000 Dogs and 4,000 Cats) and (2,000 for test_set: 1,000 Dogs and 1,000 Cats of Original Dataset (25,000 images) of Dogs vs. Cats | Kaggle
# Original Dataset link ==> https://www.kaggle.com/c/dogs-vs-cats/data
# You might use 25 or more epochs and 8000 Samples per epoch

# Installing Theano
# Installing Tensorflow
# Installing Keras

# Part 1 - Building the ConvNet

# Importing the Keras libraries and packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Initializing the ConvNet
classifier = Sequential()

# Step 1 - Building the Convolution Layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Building the Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding The Second Convolutional Layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Building the Flattening Layer
classifier.add(Flatten())

# Step 4 - Building the Fully Connected Layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the ConvNet
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the ConvNet to the Images

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ..... Fill the Rest (a Few Lines of Code!)

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    # Rescale the tensors from values between 0 and 255 to values between 0 and 1, 
    # as neural networks prefer to deal with small input values.
    # shear_range is for randomly applying shearing transformations
        # Shear tool is used to shift one part of an image, a layer, a selection or a path to a direction
        #  and the other part to the opposite direction
    # zoom_range is for randomly zooming inside pictures
    # horizontal_flip is for randomly flipping half of the images horizontally
        #  --relevant when there are no assumptions of horizontal assymetry (e.g. real-world pictures)

# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
        'ConvNet_dataset/training_set',  # this is the target directory
        target_size=(64, 64),  # all images will be resized to 64x64
            # ** should be same as Step 1 where input_shape(64, 64, 3) otherwise "Incompatible shapes" error
        batch_size=32,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

val_data = test_datagen.flow_from_directory(
        'ConvNet_dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.summary() # View all the layers of the network

classifier.fit(train_data, steps_per_epoch=(8000/32), epochs=25, validation_data=val_data, validation_steps=(2000/32))
    # steps_per_epoch is used to define how many batches of samples to use in one epoch
    ### to fix 'error tensorflow:Your input ran out of data; interrupting training'
      # "Your problem stems from the fact that the parameters steps_per_epoch and
      # valiation_steps need to be equal to the total number of data point divided to the batch_size"
classifier.save_weights('save_weight1.h5')  # always save your weights after training or during training