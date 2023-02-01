from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from load_image import *

def model_VGG16():
    # Load the VGG16 model
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    # Freeze all layers
    for layer in vgg16.layers:
        layer.trainable = False

    # Build new classifier
    x = vgg16.output
    x = Flatten()(x)
    x = Dense(512,
              activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(len(class_names), activation='softmax')(x)

    # Create new model
    return Model(inputs=vgg16.input, outputs=x)
