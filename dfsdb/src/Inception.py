import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def inception_module(x, filters):
    branch_1x1 = layers.Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
    branch_3x3 = layers.Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    branch_3x3 = layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')(branch_3x3)
    branch_5x5 = layers.Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    branch_5x5 = layers.Conv2D(filters[4], (5, 5), padding='same', activation='relu')(branch_5x5)
    branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = layers.Conv2D(filters[5], (1, 1), padding='same', activation='relu')(branch_pool)
    outputs = [branch_1x1, branch_3x3, branch_5x5, branch_pool]
    return layers.Concatenate()(outputs)

def googlenet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = inception_module(x, [64, 96, 128, 16, 32, 32])
    x = inception_module(x, [128, 128, 192, 32, 96, 64])
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def preprocess_image(image_path, input_shape):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=input_shape)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

def predict_class(image_path, model):
    preprocessed_img = preprocess_image(image_path, input_shape[:2])
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions[0])
    return predicted_class_index

# Define paths to your dataset
train_dir = 'path/to/train_dataset'
val_dir = 'path/to/validation_dataset'
test_image_path = 'path/to/test_image.jpg'

# Parameters
input_shape = (224, 224, 3)
num_classes = 3
batch_size = 32
epochs = 10

# Data generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the GoogLeNet model
googlenet_model = googlenet(input_shape, num_classes)

# Compile the model
googlenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = googlenet_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Evaluate the model
loss, accuracy = googlenet_model.evaluate(validation_generator)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))

# Save the model
googlenet_model.save('googlenet_model.h5')

# Test the model on a sample image
predicted_class_index = predict_class(test_image_path, googlenet_model)
print("Predicted class index:", predicted_class_index)
