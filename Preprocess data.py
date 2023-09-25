from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess the dataset
batch_size = 32
train_generator = datagen.flow_from_directory(
    'path_to_dataset_directory',
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='input',
    shuffle=True
)
