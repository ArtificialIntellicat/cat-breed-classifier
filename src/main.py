import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from model import create_model_with_base

# Set up data paths
dataset_directory = '../data/images'
corrupt_images_directory = '../data/corrupt_images'

# Validate images and handle corrupt images
def validate_images(image_directory, target_directory):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    for subdir in os.listdir(image_directory):
        subdir_path = os.path.join(image_directory, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdir_path, file)
                    try:
                        # Verify the file format
                        Image.open(img_path).verify()
                    except (IOError, SyntaxError) as e:
                        print(f'Corrupt image detected: {img_path}')
                        os.rename(img_path, os.path.join(target_directory, file))

validate_images(dataset_directory, corrupt_images_directory)

# Training parameters
target_size = (224, 224)
batch_size = 32

# Image data generator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Training data generator
train_generator = datagen.flow_from_directory(
    dataset_directory,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
validation_generator = datagen.flow_from_directory(
    dataset_directory,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Model creation
num_classes = train_generator.num_classes
model = create_model_with_base(num_classes, input_shape=(224, 224, 3))

# Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10, # can be set to a higher number for improved accuracy at higher computational cost
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
)

# Model evaluation
loss, accuracy = model.evaluate(
    validation_generator,
    steps=validation_generator.samples // batch_size
)
print(f'Breed Prediction Accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save('../models/cat-breed-classifier.h5')
