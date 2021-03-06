from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential, Model


import matplotlib.pyplot as plt
import numpy as np




SEED = 42



TRAIN_R = 0.6  # Train ratio

VAL_R = 0.2

TEST_R = 0.2



IMG_HEIGHT, IMG_WIDTH = (224, 224)

BATCH_SIZE = 32



DATA_DIR_PATH = "IMAGES_224x224"

OUTPUT_DIR = "IMAGES_224x224_SPLIT"





train_data_dir = f"{OUTPUT_DIR}/train"

valid_data_dir = f"{OUTPUT_DIR}/val"

test_data_dir = f"{OUTPUT_DIR}/test"



train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input)





train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

    class_mode="categorical")



valid_generator = train_datagen.flow_from_directory(

    valid_data_dir,

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=BATCH_SIZE,

    class_mode="categorical")





test_generator = test_datagen.flow_from_directory(

    test_data_dir,

    target_size=(IMG_HEIGHT, IMG_WIDTH),

    batch_size=1,

    class_mode="categorical")


EPOCHS = 100



base_model = ResNet50(include_top=False, weights="imagenet")



x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation="relu")(x)



predictions = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)



for layer in base_model.layers:

    layer.trainable = False

    

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["acc"])







history = model.fit(train_generator,

                    validation_data=valid_generator,

                    epochs=EPOCHS)
