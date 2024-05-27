import os
import pandas as pd
import numpy as np
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image

# Setting up log file
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')

# Number to class labels mapping for house image classifier
class_dict = {0: 'stress', 1: 'introvert', 2: 'extrovert'}

# Loading the data from the .csv file
try:
    labels_df = pd.read_csv('../data/houseData.csv')
except FileNotFoundError as e:
    logging.error(f"File not found error: {e}")
    exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred while loading data: {e}")
    exit(1)

# Convert 'id' column to strings and append '.png' to match the image filenames
labels_df['id'] = labels_df['id'].astype(str) + '.png'
# Convert 'class' column to strings
labels_df['class'] = labels_df['class'].astype(str)

# Prepare the data generator with augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Split data into training and validation sets
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=20
)

# Load and preprocess images from directory
train_generator = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory='../images/house/',
    x_col='id',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory='../images/house/',
    x_col='id',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define the VGG-16 model
def create_vgg16(num_classes=3):
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
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Output layer for num_classes classes
    
    return model

# Create the VGG-16 model
vgg16_model = create_vgg16(num_classes=3)

# Compile the model with the correct argument name
vgg16_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('vgg16_modelHouse.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
callbacks_list = [checkpoint]

# Train the model
history = vgg16_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks_list
)

# Evaluate the model
loss, accuracy = vgg16_model.evaluate(validation_generator, verbose=1)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))

# Generate classification report for the validation set
def generate_classification_report(model, generator):
    y_true = generator.classes
    y_pred = model.predict(generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    report = classification_report(y_true, y_pred_classes, target_names=list(class_dict.values()))
    print("Classification Report:")
    print(report)

generate_classification_report(vgg16_model, validation_generator)

# Load the saved model
loaded_model = vgg16_model

# Load and preprocess a single image
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.  # Rescale pixel values
    return img_array

# Function to get predictions for a single image
def get_prediction(model, image_array):
    prediction = model.predict(image_array)
    predicted_class = class_dict[np.argmax(prediction)]
    return predicted_class

# Load a fixed image
fixed_image_path = '../images/house/300.png'
fixed_image = load_and_preprocess_image(fixed_image_path)

# Get the prediction for the fixed image
prediction = get_prediction(loaded_model, fixed_image)

# Display the prediction
print(f'Prediction for the fixed image: {prediction}')
