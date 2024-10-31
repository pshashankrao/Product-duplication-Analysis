import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

# Paths to the parent folders containing subfolders
duplicate_folder = 'D:/Product duplication/duplicate/'
original_folder = 'D:/Product duplication/original/'

# Function to load images from nested folders
def load_images_from_nested_folders(parent_folder, label_value):
    images = []
    labels = []
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
                else:
                    print(f"Failed to load image {img_path}")
    return np.array(images), np.array(labels)

# Load images and labels from duplicate and original folders
duplicate_images, duplicate_labels = load_images_from_nested_folders(duplicate_folder, 1)
original_images, original_labels = load_images_from_nested_folders(original_folder, 0)

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

# KNN Algorithm
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_flat, y_train)
y_pred_knn = knn_model.predict(X_test_flat)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# CNN Algorithm
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

y_pred_cnn = np.argmax(cnn_model.predict(X_test), axis=-1)
accuracy_cnn = accuracy_score(y_test, y_pred_cnn)

# VGG16 Algorithm
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
for layer in vgg16_base.layers:
    layer.trainable = False  # Freeze the convolutional layers

vgg16_model = Sequential([
    vgg16_base,
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

vgg16_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
vgg16_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

y_pred_vgg16 = np.argmax(vgg16_model.predict(X_test), axis=-1)
accuracy_vgg16 = accuracy_score(y_test, y_pred_vgg16)

# EM Algorithm using Gaussian Mixture
em_model = GaussianMixture(n_components=2, random_state=42)
em_model.fit(X_train_flat)
y_pred_em = em_model.predict(X_test_flat)
accuracy_em = accuracy_score(y_test, y_pred_em)

# SVM Algorithm
svm_model = SVC()
svm_model.fit(X_train_flat, y_train)
y_pred_svm = svm_model.predict(X_test_flat)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Calculate accuracy for the duplicate and original test data
accuracy_gaussian_duplicate = accuracy_score(y_test_duplicate, gaussian_model.predict(X_test_flat_duplicate))
accuracy_gaussian_original = accuracy_score(y_test_original, gaussian_model.predict(X_test_flat_original))

accuracy_knn_duplicate = accuracy_score(y_test_duplicate, knn_model.predict(X_test_flat_duplicate))
accuracy_knn_original = accuracy_score(y_test_original, knn_model.predict(X_test_flat_original))

accuracy_cnn_duplicate = accuracy_score(y_test_duplicate, np.argmax(cnn_model.predict(X_test_duplicate), axis=-1))
accuracy_cnn_original = accuracy_score(y_test_original, np.argmax(cnn_model.predict(X_test_original), axis=-1))

accuracy_vgg16_duplicate = accuracy_score(y_test_duplicate, np.argmax(vgg16_model.predict(X_test_duplicate), axis=-1))
accuracy_vgg16_original = accuracy_score(y_test_original, np.argmax(vgg16_model.predict(X_test_original), axis=-1))

accuracy_em_duplicate = accuracy_score(y_test_duplicate, em_model.predict(X_test_flat_duplicate))
accuracy_em_original = accuracy_score(y_test_original, em_model.predict(X_test_flat_original))

accuracy_svm_duplicate = accuracy_score(y_test_duplicate, svm_model.predict(X_test_flat_duplicate))
accuracy_svm_original = accuracy_score(y_test_original, svm_model.predict(X_test_flat_original))

# Create a bar plot for the accuracies
algorithms = ['Gaussian', 'KNN', 'CNN', 'VGG16', 'EM', 'SVM']
duplicate_accuracies = [
    accuracy_gaussian_duplicate, accuracy_knn_duplicate, accuracy_cnn_duplicate,
    accuracy_vgg16_duplicate, accuracy_em_duplicate, accuracy_svm_duplicate
]
original_accuracies = [
    accuracy_gaussian_original, accuracy_knn_original, accuracy_cnn_original,
    accuracy_vgg16_original, accuracy_em_original, accuracy_svm_original
]

x = np.arange(len(algorithms))  # label locations
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, duplicate_accuracies, width, label='Duplicate')
bars2 = ax.bar(x + width/2, original_accuracies, width, label='Original')

# Add text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Algorithms')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of Different Algorithms for Duplicate and Original Images')
ax.set_xticks(x)
ax.set_xticklabels(algorithms)
ax.legend()

# Attach a text label above each bar in *rects*, displaying its height.
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bars1)
autolabel(bars2)

fig.tight_layout()
plt.show()
