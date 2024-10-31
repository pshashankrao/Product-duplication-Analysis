import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd

# Paths to the parent folders containing subfolders
duplicate_folder = 'D:/Product duplication/duplicate/'
original_folder = 'D:/Product duplication/original/'

# Function to load images from nested folders
def load_images_from_nested_folders(parent_folder, label_value):
    images = []
    labels = []
    filenames = []
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))  # Resize to 64x64 for uniformity
                    images.append(img_to_array(img))
                    labels.append(label_value)
                    filenames.append(filename)
                else:
                    print(f"Failed to load image {img_path}")
    return np.array(images), np.array(labels), filenames

# Load images and labels from duplicate and original folders
duplicate_images, duplicate_labels, duplicate_filenames = load_images_from_nested_folders(duplicate_folder, 1)
original_images, original_labels, _ = load_images_from_nested_folders(original_folder, 0)

# Combine and split the data
X = np.concatenate((duplicate_images, original_images), axis=0)
y = np.concatenate((duplicate_labels, original_labels), axis=0)

if len(X) == 0 or len(y) == 0:
    raise ValueError("No images were loaded. Please check the file paths and ensure images exist in the specified folders.")

X = X.astype('float32') / 255.0  # Normalize the images

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten the images for Gaussian and KNN
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Separate original and duplicate test data for individual accuracy calculation
X_test_duplicate = X_test[y_test == 1]
y_test_duplicate = y_test[y_test == 1]
X_test_original = X_test[y_test == 0]
y_test_original = y_test[y_test == 0]

X_test_flat_duplicate = X_test_duplicate.reshape(X_test_duplicate.shape[0], -1)
X_test_flat_original = X_test_original.reshape(X_test_original.shape[0], -1)

# Gaussian Algorithm
gaussian_model = GaussianNB()
gaussian_model.fit(X_train_flat, y_train)
y_pred_gaussian = gaussian_model.predict(X_test_flat)
accuracy_gaussian = accuracy_score(y_test, y_pred_gaussian)

y_pred_gaussian_duplicate = gaussian_model.predict(X_test_flat_duplicate)
accuracy_gaussian_duplicate = accuracy_score(y_test_duplicate, y_pred_gaussian_duplicate)

y_pred_gaussian_original = gaussian_model.predict(X_test_flat_original)
accuracy_gaussian_original = accuracy_score(y_test_original, y_pred_gaussian_original)

# KNN Algorithm
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_flat, y_train)
y_pred_knn = knn_model.predict(X_test_flat)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

y_pred_knn_duplicate = knn_model.predict(X_test_flat_duplicate)
accuracy_knn_duplicate = accuracy_score(y_test_duplicate, y_pred_knn_duplicate)

y_pred_knn_original = knn_model.predict(X_test_flat_original)
accuracy_knn_original = accuracy_score(y_test_original, y_pred_knn_original)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)

# Create validation data manually
X_train_aug, X_val, y_train_aug, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# CNN Algorithm with data augmentation and dropout
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(datagen.flow(X_train_aug, y_train_aug, batch_size=32), epochs=30, validation_data=(X_val, y_val))

y_pred_cnn = np.argmax(cnn_model.predict(X_test), axis=-1)
accuracy_cnn = accuracy_score(y_test, y_pred_cnn)

y_pred_cnn_duplicate = np.argmax(cnn_model.predict(X_test_duplicate), axis=-1)
accuracy_cnn_duplicate = accuracy_score(y_test_duplicate, y_pred_cnn_duplicate)

y_pred_cnn_original = np.argmax(cnn_model.predict(X_test_original), axis=-1)
accuracy_cnn_original = accuracy_score(y_test_original, y_pred_cnn_original)

# Plotting the results
algorithms = ['Gaussian', 'KNN', 'CNN']
accuracies = [accuracy_gaussian, accuracy_knn, accuracy_cnn]
accuracies_duplicate = [accuracy_gaussian_duplicate, accuracy_knn_duplicate, accuracy_cnn_duplicate]
accuracies_original = [accuracy_gaussian_original, accuracy_knn_original, accuracy_cnn_original]

plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.bar(algorithms, accuracies, color=['blue', 'green', 'red'])
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Overall Accuracy')
plt.ylim([0, 1])
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')

plt.subplot(1, 3, 2)
plt.bar(algorithms, accuracies_duplicate, color=['blue', 'green', 'red'])
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Duplicate Images Accuracy')
plt.ylim([0, 1])
for i, v in enumerate(accuracies_duplicate):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')

plt.subplot(1, 3, 3)
plt.bar(algorithms, accuracies_original, color=['blue', 'green', 'red'])
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Original Images Accuracy')
plt.ylim([0, 1])
for i, v in enumerate(accuracies_original):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Display image names in duplicate folder in a table
duplicate_filenames_df = pd.DataFrame(duplicate_filenames, columns=['Image Name of duplicate folder'])
print(duplicate_filenames_df)


# Display the two images side by side
#img1_path = 'D:/Product duplication/duplicate/f_mechanical parts/break pad.jpg'
#img2_path = 'D:/Product duplication/original/o_mechanical parts/break pad.jpg'
# drive -> parent folder ->sub foldre ->mech-> img."jpg,jpeg, png"

# List of image paths
image_pairs = [
    
    ('D:/Product duplication/all product/f_break pad.jpg',
     'D:/Product duplication/all product/o_break pad.jpg'),
    
    ('D:/Product duplication/all product/f_cheque.jpg', 
     'D:/Product duplication/all product/o_cheque.jpg'),
    
    ('D:/Product duplication/all product/f_pepper.jpg', 
     'D:/Product duplication/all product/o_pepper.jpg'),
    
    ('D:/Product duplication/all product/f_holes.jpg', 
     'D:/Product duplication/all product/o_alamy.jpg')
]


# Displaying the images side by side
plt.figure(figsize=(10, len(image_pairs) * 5))  # Adjust the figure size according to the number of images

for i, (img1_path, img2_path) in enumerate(image_pairs):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Check if images were loaded successfully
    if img1 is None:
        print(f"Failed to load image at path: {img1_path}")
        continue
    if img2 is None:
        print(f"Failed to load image at path: {img2_path}")
        continue

    # Resize images for display
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    # Convert BGR (OpenCV default) to RGB for displaying with Matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.subplot(len(image_pairs), 2, 2*i + 1)
    plt.imshow(img1_rgb)
    plt.title(f'Duplicate - {os.path.basename(img1_path)}')
    plt.axis('off')

    plt.subplot(len(image_pairs), 2, 2*i + 2)
    plt.imshow(img2_rgb)
    plt.title(f'Original - {os.path.basename(img2_path)}')
    plt.axis('off')

plt.tight_layout()
plt.show()