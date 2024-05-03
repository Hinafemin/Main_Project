

# In[1]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


# Step 1: Define functions for loading and preprocessing data
def load_and_preprocess_data(folder_path, label, batch_size):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = img / 255.0
        data.append(img)
        labels.append(label)

        if len(data) == batch_size:
            yield np.array(data), np.array(labels)
            data = []
            labels = []

    if data:
        yield np.array(data), np.array(labels)

# Define function for loading and preprocessing video data
def load_and_preprocess_video(folder_path, label, batch_size):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        frame_path = os.path.join(folder_path, filename)
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (64, 64))
        frame = frame / 255.0
        data.append(frame)
        labels.append(label)

        if len(data) == batch_size:
            yield np.array(data), np.array(labels)
            data = []
            labels = []

    if data:
        yield np.array(data), np.array(labels)


# In[3]:


# Step 2: Load and preprocess image data in batches
image_folder_path = 'dataset'
batch_size_img = 100  # Adjust based on memory capacity
all_images = []
all_labels = []

for folder_name, label in [('Real', 0), ('Fake', 1)]:
    folder_path = os.path.join(image_folder_path, folder_name)
    for batch_images, batch_labels in load_and_preprocess_data(folder_path, label, batch_size_img):
        all_images.extend(batch_images)
        all_labels.extend(batch_labels)

all_images = np.array(all_images)
all_labels = np.array(all_labels)


# In[4]:


# Step 3: Load and preprocess video data in batches
video_folder_path = 'dataset'
batch_size_video = 100  # Adjust based on memory capacity
all_frames = []
all_video_labels = []

for folder_name, label in [('Real', 0), ('Fake', 1)]:
    folder_path = os.path.join(video_folder_path, folder_name)
    for batch_frames, batch_labels in load_and_preprocess_video(folder_path, label, batch_size_video):
        all_frames.extend(batch_frames)
        all_video_labels.extend(batch_labels)

all_frames = np.array(all_frames)
all_video_labels = np.array(all_video_labels)


# In[5]:


# Step 4: Split data into train and test sets
x_train_img, x_test_img, y_train_img, y_test_img = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
x_train_video, x_test_video, y_train_video, y_test_video = train_test_split(all_frames, all_video_labels, test_size=0.2, random_state=42)


# In[6]:


# Step 5: Define data generator with data augmentation
data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


# In[7]:


# Step 6: Define models
base_model_img = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
for layer in base_model_img.layers:
    layer.trainable = False
x_img = Flatten()(base_model_img.output)
x_img = Dense(256, activation='relu')(x_img)
output_img = Dense(1, activation='sigmoid')(x_img)
model_img = Model(inputs=base_model_img.input, outputs=output_img)
model_img.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

base_model_video = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
for layer in base_model_video.layers:
    layer.trainable = False
x_video = Flatten()(base_model_video.output)
x_video = Dense(256, activation='relu')(x_video)
output_video = Dense(1, activation='sigmoid')(x_video)
model_video = Model(inputs=base_model_video.input, outputs=output_video)
model_video.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])


# In[8]:


# Step 7: Train models using data generators
batch_size = 32
epochs = 30

# Image model training
image_train_generator = data_generator.flow(x_train_img, y_train_img, batch_size=batch_size)
image_validation_generator = data_generator.flow(x_test_img, y_test_img, batch_size=batch_size)

for epoch in range(epochs):
    image_history = model_img.fit(image_train_generator, 
                                  steps_per_epoch=len(x_train_img) // batch_size,
                                  validation_data=image_validation_generator,
                                  validation_steps=len(x_test_img) // batch_size,
                                  epochs=1)

# Video model training
video_train_generator = data_generator.flow(x_train_video, y_train_video, batch_size=batch_size)
video_validation_generator = data_generator.flow(x_test_video, y_test_video, batch_size=batch_size)

for epoch in range(epochs):
    video_history = model_video.fit(video_train_generator, 
                                    steps_per_epoch=len(x_train_video) // batch_size,
                                    validation_data=video_validation_generator,
                                    validation_steps=len(x_test_video) // batch_size,
                                    epochs=1)


# In[9]:


# Step 8: Save models
model_img.save('resnet_image_classifier_model.h5')
model_video.save('resnet_video_classifier_model.h5')

