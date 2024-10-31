import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and inspect the data
file_path = 'D:/Product duplication/books_paper.csv'
data = pd.read_excel(file_path)
print(data.head())

# Step 2: Identify fake products
fake_products = data[(data['O_Bar_Code'] != data['F_Bar_Code']) & data['F_Bar_Code'].notna()]
fake_products_list = fake_products[['Product', 'Licence_Number', 'O_Bar_Code', 'F_Bar_Code']]
print(fake_products_list)

# Create dummy data (replace with actual feature columns and target column)
X = np.random.rand(len(data), 10)  # Replace with your actual features
y = np.random.randint(2, size=len(data))  # Replace with your actual target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Gaussian Naive Bayes
model_gnb = GaussianNB()
model_gnb.fit(X_train, y_train)
predictions_gnb = model_gnb.predict(X_test)
accuracy_gnb = accuracy_score(y_test, predictions_gnb)
print(f'Gaussian Naive Bayes Accuracy: {accuracy_gnb}')

# Step 4: K-Nearest Neighbors
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)
predictions_knn = model_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, predictions_knn)
print(f'KNN Accuracy: {accuracy_knn}')

# Step 5: Support Vector Machine
model_svm = SVC()
model_svm.fit(X_train, y_train)
predictions_svm = model_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, predictions_svm)
print(f'SVM Accuracy: {accuracy_svm}')

# Step 6: Gradient Boosting Machines
model_gbm = GradientBoostingClassifier()
model_gbm.fit(X_train, y_train)
predictions_gbm = model_gbm.predict(X_test)
accuracy_gbm = accuracy_score(y_test, predictions_gbm)
print(f'GBM Accuracy: {accuracy_gbm}')

# Step 7: Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, predictions_rf)
print(f'Random Forest Accuracy: {accuracy_rf}')

# Step 8: Convolutional Neural Network (CNN)
# Assuming image data of shape (64, 64, 3) for CNN
X_image = np.random.rand(len(data), 64, 64, 3)
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(X_image, y, test_size=0.2, random_state=42)

model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn.fit(X_train_img, y_train_img, epochs=25, batch_size=32)
accuracy_cnn = model_cnn.evaluate(X_test_img, y_test_img)[1]
print(f'CNN Accuracy: {accuracy_cnn}')

# Step 9: VGG16 with local weights
vgg_weights_path = 'D:/Product duplication/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg = VGG16(include_top=False, input_tensor=Input(shape=(64, 64, 3)), weights=vgg_weights_path)
x = Flatten()(vgg.output)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model_vgg = Model(inputs=vgg.input, outputs=output)
model_vgg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_vgg.fit(X_train_img, y_train_img, epochs=25, batch_size=32)
accuracy_vgg = model_vgg.evaluate(X_test_img, y_test_img)[1]
print(f'VGG16 Accuracy: {accuracy_vgg}')

# Show inverse matrix
matrix = np.array([[1, 2], [3, 4]])
inverse_matrix = np.linalg.inv(matrix)
print("Inverse Matrix:")
print(inverse_matrix)

# Plot heatmaps
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Heatmap for original matrix
sns.heatmap(matrix, annot=True, cmap='coolwarm', ax=ax[0])
ax[0].set_title('Original Matrix')

# Heatmap for inverse matrix
sns.heatmap(inverse_matrix, annot=True, cmap='coolwarm', ax=ax[1])
ax[1].set_title('Inverse Matrix')

plt.show()
