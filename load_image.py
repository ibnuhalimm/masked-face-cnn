from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
import pathlib
import numpy as np

# Define our directories and the image size
test_case = 'cloth/full_face'
train_directory = 'datasets/v5/' + test_case + '/train'
validation_directory = 'datasets/v5/' + test_case + '/validation'

img_width = 224
img_height = 224
batch_size_train = 32
batch_size_val = 8

# Get the class names
data_dir = pathlib.Path(train_directory)
class_names = np.array(sorted(item.name for item in data_dir.glob('*')))

# Preprocessing the data (scaling/normalization)
train_datagen = ImageDataGenerator(rescale=1.0/255)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Import data from directories and turn in into batches
train_data = train_datagen.flow_from_directory(directory=train_directory,
                                               target_size=(img_width, img_height),
                                               class_mode="categorical",
                                               batch_size=batch_size_train,
                                               shuffle=True)

validation_data = validation_datagen.flow_from_directory(directory=validation_directory,
                                                         target_size=(img_width, img_height),
                                                         class_mode="categorical",
                                                         batch_size=batch_size_val,
                                                         shuffle=True)

"""
    Optimize model training dan testing with augmented data
    Augmented data is looking at the same image but from different perspective, such as
    - original
    - rotate
    - shift
    - zoom
"""

train_data_gen_augmented = ImageDataGenerator(rescale=1/255.,
                                              rotation_range=10,
                                              horizontal_flip=True,
                                            #   preprocessing_function=preprocess_input
                                              )

train_data_augmented = train_data_gen_augmented.flow_from_directory(directory=train_directory,
                                                                    target_size=(img_width, img_height),
                                                                    class_mode="categorical",
                                                                    batch_size=batch_size_train)