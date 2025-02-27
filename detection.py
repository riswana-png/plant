import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# Set up the main directory containing subdirectories for each plant species
main_data_dir = 'SegmentedMedicinalLeafImages'
batch_size = 32
num_classes = len(os.listdir(main_data_dir))
epochs = 10

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# # List all subdirectories (class folders) in the main directory
# class_folders = os.listdir(main_data_dir)

# # Set the number of images per row
# images_per_row = 5

# # Calculate the number of rows needed
# num_rows = math.ceil(len(class_folders) / images_per_row)

# # Create a grid of subplots
# fig, axs = plt.subplots(num_rows, images_per_row, figsize=(15, 15))

# # Display images in rows with titles
# for i, class_folder in enumerate(class_folders):
#     # Get the first image file in the class folder
#     class_folder_path = os.path.join(main_data_dir, class_folder)
#     image_files = [f for f in os.listdir(class_folder_path) if f.endswith('.jpg')]
#     if image_files:
#         first_image_path = os.path.join(class_folder_path, image_files[0])
        
#         # Load the image
#         img = mpimg.imread(first_image_path)
        
#         # Calculate the row and column indices for the subplot
#         row = i // images_per_row
#         col = i % images_per_row
        
#         # Display the image in the corresponding subplot
#         axs[row, col].imshow(img)
#         axs[row, col].set_title(class_folder)
#         axs[row, col].axis('off')

# # Adjust layout for better spacing
# plt.tight_layout()
# plt.show()

# # Split ratio between training and validation data
# split_ratio = 0.8

# # Create ImageDataGenerator with data augmentation for training data
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest',
#     validation_split=1 - split_ratio  # Set validation split
# )

# # Load and preprocess training data using the generator
# train_generator = train_datagen.flow_from_directory(
#     main_data_dir,
#     target_size=(224, 224),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training'  # Specify training subset
# )

# # Load and preprocess validation data using the generator
# validation_generator = train_datagen.flow_from_directory(
#     main_data_dir,
#     target_size=(224, 224),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation'  # Specify validation subset
# )

# from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
# # Load MobileNetV2 base model
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Add custom classification head
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.5)(x)  # Adding dropout for regularization
# predictions = Dense(num_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# # Freeze the layers of the base model
# for layer in base_model.layers:
#     layer.trainable = False

# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size
# )

# model.save('plant_identification_model2.h5')

label_mapping = {i: label for i, label in enumerate(sorted(os.listdir(main_data_dir)))}

label_mapping
image_path = 'ttt.jpg'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and display the image
img = mpimg.imread(image_path)
    
plt.imshow(img)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model('plant_identification_model2.h5')

# Load and preprocess the image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(image_array)
    return preprocessed_image

# Perform prediction
def predict_plant(image_path, label_mapping):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    
    # Map model's numeric predictions to labels
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping[predicted_label_index]
    confidence = predictions[0][predicted_label_index]
    
    return predicted_label, confidence

# Provide the path to the image you want to classify
predicted_label, confidence = predict_plant(image_path, label_mapping)

# Print the prediction
print(f"Predicted Label: {predicted_label}")
print(f"Confidence: {confidence:.2f}")

## Results and Evaluation
### Training and Validation Curves
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

