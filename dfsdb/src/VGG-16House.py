import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Step 1: Load images and labels
image_dir = '../images'
csv_file = '../data/houseData.csv'

# Load labels from CSV
class_dict = {0: 'stress', 1: 'introvert', 2: 'extrovert'}
labels_df = pd.read_csv(csv_file)

# Assuming 'id' column contains the image numbers and 'class' column contains the labels
image_ids = labels_df['id']
class_labels = labels_df['class']

# Create a list of labels using class_dict
labels = np.array([class_dict[label] for label in class_labels])

# Step 2: Preprocess the images
# You can use ImageDataGenerator for data augmentation and preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Rescale pixel values and split data into train/validation

# Load and preprocess images from directory
train_generator = datagen.flow_from_directory(
    image_dir,
    target_size=(224, 224),  # VGG16 input size
    batch_size=32,
    class_mode='categorical',
    subset='training',
    classes = ['house']
)

validation_generator = datagen.flow_from_directory(
    image_dir,
    target_size=(224, 224),  # VGG16 input size
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    classes = ['house']
)
# vgg16 model

def create_vgg16():
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='softmax'))  # 1000 classes for ImageNet
    
    return model

# Step 3: Define and compile the VGG16 model
vgg16_model = create_vgg16()  

# Compile the model
vgg16_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
checkpoint = ModelCheckpoint('vgg16_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
callbacks_list = [checkpoint]

history = vgg16_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks_list
)

# Step 5: Evaluate the model
loss, accuracy = vgg16_model.evaluate(validation_generator, verbose=1)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))

# Step 6: Use the trained model to classify unknown images
# Load the saved model
from tensorflow.keras.models import load_model

loaded_model = load_model('vgg16_model.h5')

# Load and preprocess an unknown image
from tensorflow.keras.preprocessing import image

img_path = '../images/house/300.png'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.  # Rescale pixel values

# Make predictions
prediction = loaded_model.predict(img_array)
predicted_class = class_dict[np.argmax(prediction)]

print("Predicted class:", predicted_class)
